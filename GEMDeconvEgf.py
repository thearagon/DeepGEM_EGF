#!/usr/bin/env python
# coding: utf-8


from deconvEgf_helpers import *
import argparse
print(torch.__version__)
print(scipy.__version__)

def main_function(args):

    ################################################ SET UP WEIGHTS ####################################################
    n_flow = 32
    affine = True
    data_weight = 1 / args.data_sigma ** 2

    seqfrac = args.seqfrac
    npix = args.stf_size
    npiy = 1

    if args.px_init_weight == None:
        # weight on init STF
        args.px_init_weight = (1/args.data_sigma)/6e-1
    if args.px_weight == None:
        # weight for priors on E step: list, [boundaries, TV]
        args.px_weight = [(1/args.data_sigma)/1e0,
                          (1/args.data_sigma)/2e0]
    if args.logdet_weight == None:
        # weight on q_theta
        args.logdet_weight = (1/args.data_sigma)/5e2
    if args.prior_phi_weight == None:
        # weight on init GF
        args.prior_phi_weight = (1/args.data_sigma)/1e2
    if args.kernel_norm_weight == None:
        # + weight on TV
        args.kernel_norm_weight = (1/args.data_sigma)/1e4
    if args.num_egf > 1:
        args.prior_phi_weight *= 1
        kernel_corrcoef_weight = 0
        # kernel_corrcoef_weight = 0
        # args.prior_phi_weight /= 2e0
    else:
        kernel_corrcoef_weight = 0.

        ################################################ SET UP DATA ####################################################
    try:
        st_trc = obspy.read("{}".format(args.trc))
        trc = np.concatenate([st_trc[k].data[:, None] for k in range(len(st_trc))], axis=1).T
    except TypeError:
        st_trc = None
        trc = np.load("{}".format(args.trc))

    try:
        st_gf = obspy.read("{}".format(args.egf))
        gf = np.concatenate([st_gf[k].data[:, None] for k in range(len(st_gf))], axis=1).T
        gf = gf.reshape(gf.shape[0]//3, 3, gf.shape[1])
    except TypeError:
        st_gf = None
        gf = np.load("{}".format(args.egf))
        gf = gf.reshape(gf.shape[0] // 3, 3, gf.shape[1])

    if len(args.stf0) > 0:
        try:
            st_stf0 = obspy.read("{}".format(args.stf0))
            stf0 = st_stf0[0].data
        except TypeError:
            st_stf0 = None
            stf0 = np.load("{}".format(args.stf0))
    else:
        ## STF init is a gaussian
        stf0 = np.exp(-np.power(np.arange(npix) - npix//2., 2.) / (2 * np.power(npix//30., 2.)))

    ## Normalize everything
    trc = trc/ np.amax(np.abs(trc))
    # gf = gf / np.amax(np.abs(gf))   # TO TEST
    for i in range(gf.shape[0]):
        gf[i] = gf[i]/ np.amax(np.abs(gf[i]))
    stf0 = stf0 / np.amax(stf0)

    ## If we know the truth
    if args.synthetics == True:
        try:
            st_gf_true = obspy.read("{}".format(args.gf_true))
            gf_true = np.concatenate([st_gf_true[k].data[:, None] for k in range(len(st_gf_true))], axis=1).T
            gf_true = gf_true.reshape(3, gf_true.shape[1])
        except TypeError:
            st_gf_true = None
            gf_true = np.load("{}".format(args.gf_true))
            gf_true = gf_true.reshape(3, gf_true.shape[1])
        stf_true = np.load("{}".format(args.stf_true))
        gf_true = gf_true/ np.amax(np.abs(gf_true))
        stf_true = stf_true / np.amax(stf_true)

    #
    # ############################################## MODEL SETUP #####################################################
    #
    trc_ext = np.concatenate(args.num_egf*[trc[None, :, :]], axis=0)
    trc = torch.Tensor(trc).to(device=args.device)
    trc_ext = torch.Tensor(trc_ext).to(device=args.device)
    gf = torch.Tensor(gf).to(device=args.device)
    stf0 = torch.Tensor(stf0).to(device=args.device)


    ##------------------------------------------------------------------------------
    # ---- INITIALIZATION

    # kernel init
    init = gf.detach().cpu().numpy()
    kernel_network = KNetwork(init,
                              num_layers=args.num_layers,
                              num_egf=args.num_egf
                              ).to(args.device)

    if args.reverse == True:
        print("Generating Reverse RealNVP Network")
        permute = 'reverse'
    else:
        print("Generating Random RealNVP Network")
        permute = 'random'
    realnvp = realnvpfc_model.RealNVP(npix, n_flow, seqfrac=seqfrac, affine=affine, permute=permute).to(args.device)
    stf_gen = stf_generator(realnvp).to(args.device)

    # True forward model (with init GF), used for priors
    FTrue = lambda x: trueForward(torch.unsqueeze(gf, dim=0), x, args.num_egf)

    print("Models Initialized")
    trc_ext = torch.cat(args.btsize * [trc_ext.unsqueeze(0)])

    # MULTIPLE GPUS
    if len(args.device_ids) > 1:
        stf_gen = nn.DataParallel(stf_gen, device_ids=args.device_ids)
        stf_gen.to(args.device)
        kernel_network = nn.DataParallel(kernel_network, device_ids=args.device_ids)
        kernel_network.to(args.device)

    ##------------------------------------------------------------------------------
    # ---- PRIORS

    ## Priors on M step
    if len(args.device_ids) > 1:
        ker_softl1 = lambda kernel_network: torch.abs(1 - torch.sum(kernel_network.module.generatekernel()))
    else:
        ker_softl1 = lambda kernel_network: torch.abs(1 - torch.sum(kernel_network.generatekernel()))
    f_phi_prior = lambda kernel: priorPhi(kernel, gf)  ## L1
    prior_TSV = lambda kernel, weight: weight * Loss_TSV(kernel, gf) if weight > 0 else 0 ## Total Variation
    phi_priors = [f_phi_prior, prior_TSV]  ## norms on init GF
    prior_correl_multiEGF = lambda kernel, weight: weight * Loss_multicorr(kernel) if weight > 0 else 0
    if args.num_egf > 1 :
        prior_k = prior_correl_multiEGF
        pk_weight = kernel_corrcoef_weight
    else:
        prior_k = 0
        pk_weight = 0

    ## Priors on E step (weight determined by px_weight)
    prior_xtrue_L2 = lambda x, weight: weight * torch.sqrt(torch.sum((stf0 - x) ** 2))  ## L2
    prior_boundary = lambda x, weight: weight * torch.sum(torch.abs(x[:, :, 0]) * torch.abs(x[:, :, -1]))
    prior_dtw = lambda x, weight: weight * Loss_DTW(x, stf0) if weight > 0 else 0
    prior_TV_stf = lambda x, weight: weight * Loss_TV(x) ## Total Variation

    flux = np.abs(np.sum(stf0.cpu().numpy()))
    logscale_factor = img_logscale(scale=flux / (0.8 * np.shape(stf0)[0])).to(args.device)
    logdet_weight = args.logdet_weight #flux/(npix*args.data_sigma)
    prior_x = prior_dtw
    prior_img = [prior_boundary, prior_TV_stf]  # prior on STF, can be a list

    ### DEFINE OPTIMIZERS
    Eoptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(stf_gen.parameters())
                                         + list(logscale_factor.parameters())), lr=args.Elr)
    Moptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(kernel_network.parameters())), lr=args.Mlr)

    #################################### TRAINING #########################################################

    Eloss_list = []
    Eloss_prior_list = []
    Eloss_mse_list = []
    Eloss_q_list = []

    Mloss_list = []
    Mloss_mse_list = []
    Mloss_phi_list = []
    Mloss_kernorm_list = []
    Mloss_phiprior_list = []

    z_sample = torch.randn(2, npiy * npix).to(device=args.device)

    img, logdet = GForward(z_sample, stf_gen, npix, npiy, logscale_factor,
                           device=args.device, device_ids=args.device_ids if len(args.device_ids)>1 else None)
    image = img.detach().cpu().numpy()
    y = FForward(img, kernel_network, args.data_sigma, args.device)
    image_blur = y.detach().cpu().numpy()

    # print("Check Initialization", image.max(), image.min(), image_blur.max(), image_blur.min())

    if len(args.device_ids) > 1:
        learned_kernel = kernel_network.module.generatekernel().detach().cpu().numpy()[0]
    else:
        learned_kernel = kernel_network.generatekernel().detach().cpu().numpy()[0]

    if args.output == True:
    # Plot init
        if args.synthetics == True:
            plot_res(0, 0, image, learned_kernel, stf0, gf, args, true_gf = gf_true, true_stf = stf_true)
        else:
            plot_res(0, 0, image, learned_kernel, stf0, gf, args)

    for k in range(args.num_epochs):

        ############################ E STEP Update Generator Network #######################
        # Solve for the STF

        for k_sub in range(args.num_subepochsE):
            z_sample = torch.randn(args.btsize, npix).to(device=args.device)

            Eloss, qloss, priorloss, mseloss = EStep(z_sample, args.device, trc_ext, stf_gen, kernel_network,
                                                     prior_x, prior_img, logdet_weight,
                                                     args.px_init_weight, args.px_weight, args.data_sigma, npix, npiy,
                                                     logscale_factor, data_weight,
                                                     args.device_ids if len(args.device_ids)>1 else None,
                                                     args.num_egf)
            #             print(qloss, priorloss, mseloss)
            Eloss_list.append(Eloss.detach().cpu().numpy())
            Eloss_prior_list.append(priorloss.detach().cpu().numpy())
            Eloss_q_list.append(qloss.detach().cpu().numpy())
            Eloss_mse_list.append(mseloss.detach().cpu().numpy())
            Eoptimizer.zero_grad()
            Eloss.backward()
            nn.utils.clip_grad_norm_(list(stf_gen.parameters()) + list(logscale_factor.parameters()), 1)
            Eoptimizer.step()

            if ((k_sub % args.save_every == 0) and args.EMFull) or ((k % args.save_every == 0) and not args.EMFull):

                z_sample = torch.randn(args.btsize, npix).to(device=args.device)
                img, logdet = GForward(z_sample, stf_gen, npix, npiy, logscale_factor,
                                       device=args.device, device_ids=args.device_ids if len(args.device_ids)>1 else None)
                image = img.detach().cpu().numpy()

                if args.output == True:
                    if k_sub != 0 and k % 10 == 0:
                        with torch.no_grad():
                            torch.save({
                                'epoch': k,
                                'model_state_dict': stf_gen.state_dict(),
                                'optimizer_state_dict': Eoptimizer.state_dict(),
                            }, '{}/{}{}_{}.pt'.format(args.PATH, "GeneratorNetwork", str(k).zfill(5), str(k_sub).zfill(5)))

            if ((k_sub % args.print_every == 0) and args.EMFull) or ((k % args.print_every == 0) and not args.EMFull):
                # print(f"epoch: {k:} {k_sub:}, E step losses: {Eloss_list[-1]:.5f}")
                print(f"epoch: {k:} {k_sub:}, E step losses (tot, prior, q, mse): ")
                print(''.join(f"{x:.2f}, " for x in [Eloss_list[-1], Eloss_prior_list[-1], Eloss_q_list[-1], Eloss_mse_list[-1]]) )

        ############################ M STEP Update Kernel Network #######################
        # update the kernel = EGF -> update the TRACE

        for k_sub in range(args.num_subepochsM):
            z_sample = torch.randn(args.btsize, npix).to(device=args.device)
            x_sample = torch.randn(args.btsize, npix).to(device=args.device).reshape((-1, npiy, npix))

            Mloss, philoss, mse, kernorm, priorphi = MStep(z_sample, x_sample, npix, npiy, args.device, trc_ext,
                                                           stf_gen, kernel_network, args.phi_weight,
                                                           FTrue, args.data_sigma, logscale_factor,
                                                           phi_priors, args.prior_phi_weight,
                                                           ker_softl1, args.kernel_norm_weight,
                                                           pk_weight, prior_k,
                                                           args.device_ids if len(args.device_ids)>1 else None,
                                                           args.num_egf)

            Mloss_list.append(Mloss.detach().cpu().numpy())
            Mloss_mse_list.append(mse.detach().cpu().numpy())
            Mloss_phi_list.append(philoss.detach().cpu().numpy())
            Mloss_phiprior_list.append(priorphi.detach().cpu().numpy())
            Mloss_kernorm_list.append(kernorm.detach().cpu().numpy())
            Moptimizer.zero_grad()
            Mloss.backward()
            nn.utils.clip_grad_norm_(list(kernel_network.parameters()), 1)
            Moptimizer.step()

            if ((k_sub % args.save_every == 0) and args.EMFull) or ((k % args.save_every == 0) and not args.EMFull):
                if len(args.device_ids) > 1:
                    learned_kernel = kernel_network.module.generatekernel().detach().cpu().numpy()[0]
                else:
                    learned_kernel = kernel_network.generatekernel().detach().cpu().numpy()[0]

                if args.output == True:
                    with torch.no_grad():
                        torch.save({
                            'epoch': k,
                            'model_state_dict': kernel_network.state_dict(),
                            'optimizer_state_dict': Moptimizer.state_dict(),
                        }, '{}/{}{}_{}.pt'.format(args.PATH, "KernelNetwork", str(k).zfill(5), str(k_sub).zfill(5)))
                    np.save("{}/Data/learned_kernel.npy".format(args.PATH), learned_kernel)

                    fig, ax = plt.subplots()
                    plt.plot(np.log10(Eloss_list), label="Estep")
                    plt.plot(np.log10(Eloss_prior_list), ":", label="p(x), E step priors")
                    plt.plot(np.log10(Eloss_mse_list), "--", label="Estep MSE")
                    plt.plot(np.log10(Eloss_q_list), ":", label='q')
                    plt.plot(np.log10(Mloss_list), label="Mstep")
                    # plt.plot(np.log10(Mloss_phi_list), ":", label="p(phi), MSE w. random STF")
                    plt.plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
                    plt.plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
                    plt.plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Priors")
                    plt.legend()
                    plt.savefig("{}/loss.png".format(args.PATH))
                    plt.close()

                    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
                    ax[0].plot(np.log10(Eloss_list), label="Estep")
                    ax[0].plot(np.log10(Eloss_mse_list), "--", label="Estep MSE")
                    ax[0].plot(np.log10(Eloss_prior_list), ":", label="Estep Priors")
                    ax[0].plot(np.log10(Eloss_q_list), ":", label="q")
                    ax[0].legend()
                    ax[1].plot(np.log10(Mloss_list), label="Mstep")
                    # ax[1].plot(np.log10(Mloss_phi_list), ":", label="p(phi), MSE w. random STF")
                    ax[1].plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
                    ax[1].plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
                    ax[1].plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Priors")
                    ax[1].legend()
                    plt.savefig("{}/SeparatedLoss.png".format(args.PATH))
                    plt.close()

            if ((k_sub % args.print_every == 0) and args.EMFull) or ((k % args.print_every == 0) and not args.EMFull):
                # print(f"epoch: {k:} {k_sub:}, M step losses: {Mloss_list[-1]:.5f}")
                print(f"epoch: {k:} {k_sub:}, M step losses (tot, phi_prior, norm, mse) : ")
                print(''.join(f"{x:.2f}, " for x in [Mloss_list[-1], Mloss_phiprior_list[-1], Mloss_kernorm_list[-1], Mloss_mse_list[-1]]))

        if args.output == True:
            # Plot results
            if args.synthetics == True:
                plot_res(k, k_sub,
                         image, learned_kernel,
                         stf0, gf, args,
                         true_gf = gf_true, true_stf = stf_true)
            else:
                plot_res(k, k_sub,
                         image, learned_kernel,
                         stf0, gf, args)

    if args.output == True:
        fig, ax = plt.subplots()
        plt.plot(np.log10(Eloss_list), label="Estep")
        plt.plot(np.log10(Eloss_prior_list), ":", label="p(x), E step priors")
        plt.plot(np.log10(Eloss_mse_list), "--", label="Estep MSE")
        plt.plot(np.log10(Eloss_q_list), ":", label='q')
        plt.plot(np.log10(Mloss_list), label="Mstep")
        # plt.plot(np.log10(Mloss_phi_list), ":", label="p(phi), MSE w. random STF")
        plt.plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
        plt.plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
        plt.plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Priors")
        plt.legend()
        plt.savefig("{}/loss.png".format(args.PATH))
        plt.close()

        fig, ax = plt.subplots(1, 2, figsize=(15, 4))
        ax[0].plot(np.log10(Eloss_list), label="Estep")
        ax[0].plot(np.log10(Eloss_mse_list), "--", label="Estep MSE")
        ax[0].plot(np.log10(Eloss_prior_list), ":", label="Estep Priors")
        ax[0].plot(np.log10(Eloss_q_list), ":", label="q")
        ax[0].legend()
        ax[1].plot(np.log10(Mloss_list), label="Mstep")
        # ax[1].plot(np.log10(Mloss_phi_list), ":", label="p(phi), MSE w. random STF")
        ax[1].plot(np.log10(Mloss_mse_list), ":", label="Mstep MSE")
        ax[1].plot(np.log10(Mloss_kernorm_list), ":", label="Mstep Kernel Norm")
        ax[1].plot(np.log10(Mloss_phiprior_list), ":", label="Mstep Priors")
        ax[1].legend()
        plt.savefig("{}/SeparatedLoss.png".format(args.PATH))
        plt.close()

    ############################################# GENERATE FIGURES ###########################################################

    if len(args.device_ids) > 1:
        learned_kernel = kernel_network.module.generatekernel().detach().cpu().numpy()[0]
    else:
        learned_kernel = kernel_network.generatekernel().detach().cpu().numpy()[0]

    z_sample = torch.randn(args.btsize, npix).to(device=args.device)
    img, logdet = GForward(z_sample, stf_gen, npix, npiy, logscale_factor,
                           device=args.device, device_ids=args.device_ids if len(args.device_ids)>1 else None)
    image = img.detach().cpu().numpy()
    y = FForward(img, kernel_network, args.data_sigma, args.device)
    image_blur = y.detach().cpu().numpy()

    # if args.output == True:
    np.save("{}/Data/reconSTF.npy".format(args.PATH), image)

    if st_trc is not None:
        st_trc_sd = st_trc.copy()
        st_trc_mn = st_trc.copy()
        for i in range(3):
            st_trc_mn[i].data = np.mean(image_blur, axis=(0,1))[i]
            st_trc_sd[i].data = np.std(image_blur, axis=(0,1))[i]
        st_trc_mn.write("{}/{}_out_mean.mseed".format(args.PATH, args.trc.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
        st_trc_sd.write("{}/{}_out_std.mseed".format(args.PATH, args.trc.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
    np.save("{}/Data/outTRC.npy".format(args.PATH), image_blur)

    if st_gf is not None:
        st_gf_out = st_gf.copy()
        lk = learned_kernel.reshape(args.num_egf*3, learned_kernel.shape[-1])
        for i in range(3):
            st_gf_out[i].data = lk[i, :]
        st_gf_out.write("{}/{}_out.mseed".format(args.PATH, args.egf.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
    np.save("{}/Data/outGF.npy".format(args.PATH), learned_kernel)

    # Plot results
    if args.synthetics == True and st_trc is None:
        plot_res(k, k_sub,
                 image, learned_kernel,
                 stf0, gf, args,
                 true_gf=gf_true, true_stf=stf_true)
        plot_trace(trc, image_blur, args)
    elif args.synthetics == True and st_gf is not None and st_trc is not None:
        plot_res(k, k_sub,
                 image, learned_kernel,
                 stf0, gf, args,
                 true_gf=gf_true, true_stf=stf_true)
        plot_st(st_trc, st_gf, image_blur, learned_kernel, image, args)
    elif st_gf is not None and st_trc is not None:
        plot_st(st_trc, st_gf, image_blur, learned_kernel, image, args)
    else:
        plot_res(k, k_sub,
                 image, learned_kernel,
                 stf0, gf, args)

        plot_trace(trc, image_blur, args)
        # plot_trace_diff(trc, image_blur, args)

    # calc_corr_wtrue(image_blur, image, learned_kernel, trc, stf0, gf)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='args')
    # Configurations
    parser.add_argument('--btsize', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--num_epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 3500)')
    parser.add_argument('--num_subepochsE', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--num_subepochsM', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--save_every', type=int, default=50, metavar='N',
                        help='checkpoint model (default: 50)')
    parser.add_argument('--print_every', type=int, default=50, metavar='N',
                        help='checkpoint model (default: 50)')
    parser.add_argument('--EMFull', action='store_true', default=True,
                        help='True: E to convergence, M to convergence False: alternate E, M every epoch (default: False)')
    parser.add_argument('--num_layers', type=int, default=7, metavar='N',
                        help='number of layers for kernel (default: 3)')
    parser.add_argument('--stf_size', type=int, default=40, metavar='N',
                        help='length of STF (default: 40)')
    parser.add_argument('--num_egf', type=int, default=1, metavar='N',
                        help='number of EGF (default: 1)')

    parser.add_argument('--dv', type=str, default='cpu',
                        help='which GPU to use, or cpu by default')
    parser.add_argument('--multidv', type=int, nargs='+', default=None,
                        help="use multiple gpus (default: 1) use -1 for all")
    parser.add_argument('--output', action='store_true', default=False,
                        help='Plot figures, store output')

    # User configurations
    parser.add_argument('-dir', '--dir', type=str, default="results",
                        help='output folder')
    parser.add_argument('--trc', type=str, default='',
                        help='trace file name')
    parser.add_argument('--egf', type=str, default='',
                        help='init trace file name')
    parser.add_argument('--stf0', type=str, default='',
                        help='init STF file name')
    parser.add_argument('--synthetics', action='store_true', default=False,
                        help='synthetic case, if we know the truth')
    parser.add_argument('--stf_true', type=str, default='',
                        help='synthetic case, true stf filename')
    parser.add_argument('--gf_true', type=str, default='',
                        help='synthetic case, true gf filename')

    # network setup
    parser.add_argument('--x_rand', action='store_true', default=True,
                        help='random x or from a certain sample')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='permute parameter, if False, random, if True, reverse')
    parser.add_argument('--seqfrac', type=int, default=2,
                        help='seqfrac (default:2), should be < to stf length')

    # parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--Elr', type=float, default=1e-3,
                        help='learning rate(default: 1e-4)')
    parser.add_argument('--Mlr', type=float, default=1e-3,
                        help='learning rate(default: 1e-4)')

    parser.add_argument('--data_sigma', type=float, default=5e-5,
                        help='data sigma (default: 5e-5)')
    parser.add_argument('--px_weight', type=float, nargs='+', default=None,
                        help='weight on E step priors, list (default None = function of data_sigma)')
    parser.add_argument('--logdet_weight', type=float, default=None,
                        help='weight on q_theta, E step prior (default None = function of data_sigma)')
    parser.add_argument('--kernel_norm_weight', type=float, default=None,
                        help='kernel norm weight + weight on TV, M step (default None = function of data_sigma)')
    parser.add_argument('--prior_phi_weight', type=float, default=None,
                        help='weight on init GF on M step (default None = function of data_sigma)')
    parser.add_argument('--px_init_weight', type=float, default=None,
                        help='weight on init STF on E step prior, default 0')
    parser.add_argument('--phi_weight', type=float, default=1e-1,
                        help='weight on MSE loss with random STF and init GF, M step, (default: 1e-1)')

    args = parser.parse_args()

    if os.uname().nodename == 'wouf':
        matplotlib.use('TkAgg')
        args.dir = '/home/thea/projet/EGF/deconvEgf_res/pala_test/'
        args.trc = "/home/thea/projet/EGF/cahuilla/semisynth/multi_semisy2_PALA_trace.npy"
        args.egf = "/home/thea/projet/EGF/cahuilla/data/38245496/PALA_multi_m2_trc.mseed"
        args.stf0 ="/home/thea/projet/EGF/cahuilla/semisynth/multi_semisy2_PALA_stf_true.npy"
        args.gf_true = "/home/thea/projet/EGF/cahuilla/semisynth/multi_semisy2_PALA_gf.npy"
        args.stf_true = "/home/thea/projet/EGF/cahuilla/semisynth/multi_semisy2_PALA_stf_true.npy"
        args.output = True
        args.synthetics = True
        args.num_egf = 3
        args.px_init_weight = 3e4
        args.btsize = 15
        args.num_subepochsE = 2
        args.num_subepochsM = 2


    if args.dir is not None:
        args.PATH = args.dir
    else:
        args.PATH = "./"

    if torch.cuda.is_available() and args.dv != 'cpu':
        args.device = args.dv
        dv = int(args.device[-1])
        if args.multidv == -1:
            arr = [i for i in range(torch.cuda.device_count())]
            args.device_ids = [dv] + arr[0:dv] + arr[dv+1:]
        elif args.multidv is None:
            args.device_ids = [dv]
        else:
            args.device_ids = [dv] + args.multidv
    else:
        args.device = 'cpu'
        args.device_ids = []

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(args)

    USE_GPU = True

    print("cuda available ", torch.cuda.is_available())
    print("---> num gpu", torch.cuda.device_count())
    print('---> using device:', args.device)
    if torch.cuda.is_available():
        print(" Using {} GPUS".format(len(args.device_ids)))

    if args.EMFull == True:
        args.num_epochs = args.num_epochs + 1
        args.num_subepochsE = args.num_subepochsE + 1
        args.num_subepochsM = args.num_subepochsM + 1
        print("Full EM w/ {} epochs and {} E subepochs {} M subepochs".format(args.num_epochs, args.num_subepochsE,
                                                                              args.num_subepochsM))
    else:
        args.num_epochs = args.num_epochs + 1
        args.num_subepochsE = 1
        args.num_subepochsM = 1
        print("Stochastic EM w/ {} epochs and {} E subepochs {} M subepochs".format(args.num_epochs,
                                                                                    args.num_subepochsE,
                                                                                    args.num_subepochsM))

    try:
        # Create target Directory
        os.mkdir(args.PATH)
        print("Directory ", args.PATH, " Created ")
    except FileExistsError:
        print("Directory ", args.PATH, " already exists")
    try:
        # Create target Directory
        os.mkdir(args.PATH + "/Data")
        print("Directory ", args.PATH + "/Data", " Created ")
    except FileExistsError:
        print("Directory ", args.PATH + "/Data", " already exists")
    with open("{}/args.json".format(args.PATH), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    main_function(args)
