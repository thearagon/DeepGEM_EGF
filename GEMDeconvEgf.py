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

    seqfrac = args.seqfrac
    npix = args.stf_size
    npiy = 1
    args.phi_weight = 1e-1

    if args.px_init_weight == None:
        # weight on init STF, 1e4 if stf0 (becomes 5e3 if no stf0)
        args.px_init_weight = (1/args.data_sigma)/2e2 #2e-1 #6e-1
    if args.px_weight == None:
        # weight for priors on E step: list, [boundaries, TV]
        args.px_weight = [(1/args.data_sigma)/1e0,
                          (1/args.data_sigma)/7e-1]
    if args.logdet_weight == None:
        # weight on q_theta
        # args.logdet_weight = (1/args.data_sigma)/4e1 # 5e2
        args.logdet_weight = 1
    if args.prior_phi_weight == None:
        # weight on init GF.
        args.prior_phi_weight = (1/args.data_sigma)/3e2 #2e3 #3e2
    if args.kernel_norm_weight == None:
        # + weight on TV
        args.kernel_norm_weight = (1/args.data_sigma)/1e6 #1e4
    if args.num_egf > 1:
        if args.egf_multi_weight == None:
            args.egf_multi_weight = 5e-1
        if args.egf_qual_weight == None:
            args.egf_qual_weight = np.ones(args.num_egf).tolist()
    else:
        args.egf_multi_weight = 0.
        args.egf_qual_weight = [1]
        #args.prior_phi_weight *= 5e0

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
        ## Order in stream is all traces E, all traces N, all traces Z
        gf = gf.reshape(gf.shape[0]//3, 3, gf.shape[1], order='F')
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
        if npix > len(stf0):
            stf_rs = np.zeros(npix)
            stf_rs[(len(stf_rs) - len(stf0)) // 2:-(len(stf_rs) - len(stf0)) // 2] = stf0
            stf0 = stf_rs
        elif npix < len(stf0):
            args.stf_size = len(stf0)
            npix = len(stf0)
            print('STF size set to length of STF0')

    else:
        ## STF init is a gaussian
        τc = npix//10. ## function of rate... M0 ?
        stf0 = 0.1*np.ones(npix) + 0.9*np.exp(-(np.arange(npix) - npix//2.)**2 / (2 * (τc/2)**2)) # npix//3
        args.px_init_weight /= 4.

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
        gf_true = gf_true / np.amax(np.abs(gf_true))
        stf_true = stf_true / np.amax(stf_true)
        if npix > len(stf_true):
            stf_rs = np.zeros(npix)
            stf_rs[(len(stf_rs) - len(stf_true)) // 2:-(len(stf_rs) - len(stf_true)) // 2] = stf_true
            stf_true = stf_rs
        elif npix < len(stf_true):
            args.stf_size = len(stf_true)
            npix = len(stf_true)

    ## Normalize everything
    if args.M0 is not None and args.M0_egf is not None:
        normalize = False
        norm = args.M0/args.M0_egf
    elif args.num_egf > 1:
        normalize = True
        norm = 1.
    else:
        normalize = True
        norm = 1.

    gf = gf / np.amax(np.abs(gf))
    gf = torch.Tensor(gf).to(device=args.device)
    stf0 = norm * stf0 / np.amax(stf0)
    stf0 = torch.Tensor(stf0).to(device=args.device)

    init_trc = trueForward(gf, stf0.view(1,1,-1), args.num_egf, normalize)
    trc /= np.amax(np.abs(trc))
    # test
    # if normalize is False:
    trc *= np.amax(np.abs(init_trc.detach().cpu().numpy()))
    trc = torch.Tensor(trc).to(device=args.device)
    trc_ext = torch.Tensor(trc).to(device=args.device)

    if args.synthetics == True:
        stf_true *= norm

    #
    # ############################################## MODEL SETUP #####################################################
    #

    ##------------------------------------------------------------------------------
    # ---- INITIALIZATION

    # kernel init
    kernel_network = [KNetwork(gf[i],
                              num_layers=args.num_layers,
                              num_egf=1,
                              device=args.device,
                              normalize=normalize
                              ).to(args.device) for i in range(args.num_egf)]

    if args.reverse == True:
        print("Generating Reverse RealNVP Network")
        permute = 'reverse'
    else:
        print("Generating Random RealNVP Network")
        permute = 'random'
    realnvp = realnvpfc_model.RealNVP(npix, n_flow, seqfrac=seqfrac, affine=affine, permute=permute).to(args.device)
    stf_gen = stf_generator(realnvp, norm).to(args.device)

    # True forward model (with init GF), used for priors
    FTrue = lambda x: trueForward(torch.unsqueeze(gf, dim=0), x, args.num_egf)

    print("Models Initialized")

    # MULTIPLE GPUS
    if len(args.device_ids) > 1:
        stf_gen = nn.DataParallel(stf_gen, device_ids=args.device_ids)
        stf_gen.to(args.device)
        for i in range(args.num_egf):
            kernel_network[i] = nn.DataParallel(kernel_network[i], device_ids=args.device_ids)
            kernel_network[i].to(args.device)

    ##------------------------------------------------------------------------------
    # ---- PRIORS

    ## Priors on M step
    if len(args.device_ids) > 1:
        ker_softl1 = lambda kernel_network: torch.abs(1 - torch.sum(kernel_network.module.generatekernel()))
    else:
        ker_softl1 = lambda kernel_network: torch.abs(1 - torch.sum(kernel_network.generatekernel()))
    f_phi_prior = lambda kernel: priorPhi(kernel, gf)
    if args.num_egf == 1:
        prior_L2 = lambda kernel, weight: weight * Loss_TSV(kernel, gf) if weight > 0 else 0 ## Total Variation
        #prior_L2 = lambda weight, kernel : weight * (2.5e-3 * Loss_DTW_Mstep(kernel, gf) + 1e-2 * Loss_L2(kernel, gf)) if weight > 0 else 0
        #prior_L1 = lambda kernel: Loss_L1(kernel, gf)
    else:
        prior_L2 = lambda weight, kernel, i : weight * (2.5e-3 * Loss_DTW_Mstep(kernel, gf[i].unsqueeze(0)) +  1e-2 * Loss_L2(kernel, gf[i].unsqueeze(0))) if weight > 0 else 0
        prior_L1 = lambda kernel, idx: Loss_L1(kernel, gf[idx])
    phi_priors = [f_phi_prior, prior_L2]  ## norms on init GF

    ## Priors on E step
    x_softl1 = lambda x: torch.abs(1 - torch.sum(x))
    prior_boundary = lambda x, weight: weight * torch.sum(torch.abs(x[:, :, 0]) * torch.abs(x[:, :, -1]))
    # prior_dtw = lambda x, weight: weight * (Loss_DTW(x, stf0)) if weight > 0 else 0
    prior_dtw = lambda x, weight: weight * (Loss_L2(x, stf0) + Loss_DTW(x, stf0)) if weight > 0 else 0
    prior_TV_stf = lambda x, weight: weight * Loss_TV(x)

    flux = torch.abs(torch.sum(stf0))
    logscale_factor = img_logscale(scale=flux / (0.8 * stf0.shape[0]), device=args.device).to(args.device)
    logdet_weight = args.logdet_weight #flux/(npix*args.data_sigma)
    prior_x = prior_dtw
    prior_img = [prior_boundary, prior_TV_stf]  # prior on STF, can be a list

    ### DEFINE OPTIMIZERS
    Eoptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(stf_gen.parameters())
                                         + list(logscale_factor.parameters())), lr=args.Elr)
    Moptimizer = [torch.optim.Adam(filter(lambda p: p.requires_grad, list(kernel_network[i].parameters())),
                                   lr=args.Mlr) for i in range(args.num_egf) ]

    #################################### TRAINING #########################################################

    Eloss_list = []
    Eloss_prior_list = []
    Eloss_mse_list = []
    Eloss_q_list = []

    Mloss_list = {}
    Mloss_mse_list = {}
    Mloss_multi_list = {}
    Mloss_kernorm_list = {}
    Mloss_phiprior_list = {}
    Mloss = {}

    for k_egf in range(args.num_egf):
        Mloss_list[k_egf] = []
        Mloss_mse_list[k_egf] = []
        Mloss_multi_list[k_egf] = []
        Mloss_kernorm_list[k_egf] = []
        Mloss_phiprior_list[k_egf] = []

    ## Initialize
    z_sample = torch.randn(args.btsize, npiy * npix).to(device=args.device)

    img, logdet = GForward(z_sample, stf_gen, npix, npiy, logscale_factor,
                           device=args.device, device_ids=args.device_ids if len(args.device_ids)>1 else None)
    image = img.detach().cpu().numpy()
    y = [FForward(img, kernel_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
    inferred_trace = [ y0.detach().cpu().numpy() for y0 in y ]


    learned_kernel = [kernel_network[i].module.generatekernel() for i in range(args.num_egf)] \
        if len(args.device_ids) > 1 else [kernel_network[i].generatekernel() for i in range(args.num_egf)]

    learned_kernel_np = [learned_kernel[i].detach().cpu().numpy()[0] for i in range(args.num_egf)]
    if args.output == True:
        # Plot init
        if args.synthetics == True:
            plot_res(0, 0, image, learned_kernel_np, inferred_trace, stf0, gf, trc, args, true_gf=gf_true, true_stf=stf_true)
        else:
            plot_res(0, 0, image, learned_kernel_np, inferred_trace, stf0, gf, trc, args)

    trc_ext = torch.cat(args.btsize * [trc_ext.unsqueeze(0)])

    # Save args
    with open("{}/args.json".format(args.PATH), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for k in range(args.num_epochs):

        ############################ E STEP Update STF Network #######################

        for k_sub in range(args.num_subepochsE):
            z_sample = torch.randn(args.btsize, npix).to(device=args.device)

            Eloss, qloss, priorloss, mseloss = EStep(z_sample, trc_ext, stf_gen, kernel_network,
                                                     prior_x, prior_img, logdet_weight,
                                                     npix, npiy, logscale_factor, args)

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
                print(f"epoch: {k:} {k_sub:}, E step losses (tot, prior, q, mse): ")
                print(''.join(f"{x:.2f}, " for x in [Eloss_list[-1], Eloss_prior_list[-1], Eloss_q_list[-1], Eloss_mse_list[-1]]) )

        ############################ M STEP Update Kernel Network #######################

        for k_sub in range(args.num_subepochsM):

            z_sample = torch.randn(args.btsize, npix).to(device=args.device)
            x_sample = torch.randn(args.btsize, npix).to(device=args.device).reshape((-1, npiy, npix))

            Mloss, mse, kernorm, priorphi, multiloss = MStep(z_sample, x_sample, npix, npiy,
                                                            trc_ext,
                                                            stf_gen, kernel_network,
                                                            FTrue, logscale_factor,
                                                            phi_priors, ker_softl1,
                                                            args)

            for k_egf in range(args.num_egf):
                Mloss_list[k_egf].append(Mloss[k_egf].detach().cpu().numpy())
                Mloss_mse_list[k_egf].append(mse[k_egf].detach().cpu().numpy())
                Mloss_multi_list[k_egf].append(multiloss.detach().cpu().numpy())
                Mloss_phiprior_list[k_egf].append(priorphi[k_egf].detach().cpu().numpy())
                Mloss_kernorm_list[k_egf].append(kernorm[k_egf].detach().cpu().numpy())
                Moptimizer[k_egf].zero_grad()
                Mloss[k_egf].backward(retain_graph=True)
                nn.utils.clip_grad_norm_(list(kernel_network[k_egf].parameters()), 1)
                Moptimizer[k_egf].step()

                if ((k_sub % args.print_every == 0) and args.EMFull) or (
                        (k % args.print_every == 0) and not args.EMFull):
                    print(f"epoch: {k:} {k_sub:}, M step EGF {k_egf:} losses (tot, phi_prior, norm, mse, multi) : ")
                    print(''.join(f"{x:.2f}, " for x in
                                  [Mloss_list[k_egf][-1], Mloss_phiprior_list[k_egf][-1],
                                   Mloss_kernorm_list[k_egf][-1], Mloss_mse_list[k_egf][-1],
                                   Mloss_multi_list[k_egf][-1]]))

                if args.output == True:
                    with torch.no_grad():
                        torch.save({
                            'epoch': k,
                            'model_state_dict': kernel_network[k_egf].state_dict(),
                            'optimizer_state_dict': Moptimizer[k_egf].state_dict(),
                        }, '{}/{}{}_{}.pt'.format(args.PATH, "KernelNetwork_egf", str(k).zfill(5), str(k_egf).zfill(5)))
                    np.save("{}/Data/learned_kernel.npy".format(args.PATH), learned_kernel_np)

                    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
                    ax[0].plot(np.log10(Eloss_list), label="Estep")
                    ax[0].plot(np.log10(Eloss_mse_list), "--", label="Estep MSE")
                    ax[0].plot(np.log10(Eloss_prior_list), ":", label="Estep Priors")
                    ax[0].plot(np.log10(Eloss_q_list), ":", label="q")
                    ax[0].legend()
                    ax[1].plot(np.log10(Mloss_list[k_egf]), label="Mstep")
                    if args.num_egf > 1:
                        ax[1].plot(np.log10(Mloss_multi_list[k_egf]), ":", label="Mstep Multi Loss")
                    ax[1].plot(np.log10(Mloss_mse_list[k_egf]), ":", label="Mstep MSE")
                    # ax[1].plot(np.log10(Mloss_kernorm_list[k_egf]), ":", label="Mstep Kernel Norm")
                    ax[1].plot(np.log10(Mloss_phiprior_list[k_egf]), ":", label="Mstep Priors")
                    ax[1].legend()
                    plt.savefig("{}/SeparatedLoss_{}.png".format(args.PATH,k_egf), dpi=300)
                    plt.close()

        ## Plot output
        learned_kernel = [kernel_network[i].module.generatekernel().detach() for i in range(args.num_egf)] \
            if len(args.device_ids) > 1 else [kernel_network[i].generatekernel().detach() for i in range(args.num_egf)]

        z_sample = torch.randn(args.btsize, npix).to(device=args.device)
        img, logdet = GForward(z_sample, stf_gen, npix, npiy, logscale_factor,
                               device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
        image = img.detach().cpu().numpy()
        y = [FForward(img, kernel_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
        inferred_trace = [y[i].detach().cpu().numpy() for i in range(args.num_egf)]
        learned_kernel_np = [learned_kernel[i].cpu().numpy()[0] for i in range(args.num_egf)]

        if args.output == True:
            if args.synthetics == True:
                plot_res(k, k_sub, image, learned_kernel_np, inferred_trace, stf0, gf, trc, args, true_gf=gf_true,
                         true_stf=stf_true)
            else:
                plot_res(k, k_sub, image, learned_kernel_np, inferred_trace, stf0, gf, trc, args)

    ############################################# GENERATE FIGURES ###########################################################

    learned_kernel = [kernel_network[i].module.generatekernel().detach() for i in range(args.num_egf)] \
            if len(args.device_ids) > 1 else [kernel_network[i].generatekernel().detach() for i in range(args.num_egf)]
    z_sample = torch.randn(args.btsize, npix).to(device=args.device)
    img, logdet = GForward(z_sample, stf_gen, npix, npiy, logscale_factor,
                           device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
    image = img.detach().cpu().numpy()
    y = [FForward(img, kernel_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
    inferred_trace = [y[i].detach().cpu().numpy() for i in range(args.num_egf)]
    learned_kernel_np = np.array([learned_kernel[i].cpu().numpy()[0] for i in range(args.num_egf)])

    np.save("{}/Data/reconSTF.npy".format(args.PATH), image)

    if st_trc is not None:
        st_trc_sd = st_trc.copy()
        st_trc_mn = st_trc.copy()
        for i in range(3):
            st_trc_mn[i].data = np.mean(inferred_trace, axis=(0,1))[i]
            st_trc_sd[i].data = np.std(inferred_trace, axis=(0,1))[i]
        st_trc_mn.write("{}/{}_out_mean.mseed".format(args.PATH, args.trc.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
        st_trc_sd.write("{}/{}_out_std.mseed".format(args.PATH, args.trc.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
    np.save("{}/Data/outTRC.npy".format(args.PATH), inferred_trace)

    if st_gf is not None:
        st_gf_out = st_gf.copy()
        lk = learned_kernel_np.reshape(args.num_egf*3, learned_kernel_np.shape[-1])
        for i in range(len(lk)):
            st_gf_out[i].data = lk[i, :]
        st_gf_out.write("{}/{}_out.mseed".format(args.PATH, args.egf.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
    np.save("{}/Data/outGF.npy".format(args.PATH), learned_kernel_np)

    # Plot results
    if args.synthetics == True and st_trc is None:
        plot_res(k, k_sub, image, learned_kernel_np, inferred_trace, stf0, gf, trc, args, true_gf=gf_true,
                     true_stf=stf_true)
        plot_trace(trc, inferred_trace, args)
    elif args.synthetics == True and st_gf is not None and st_trc is not None:
        plot_res(k, k_sub, image, learned_kernel_np, inferred_trace, stf0, gf, trc, args, true_gf=gf_true,
                 true_stf=stf_true)
        plot_st(st_trc, st_gf, inferred_trace, learned_kernel_np, image, args, init_trc)
    elif st_gf is not None and st_trc is not None:
        plot_st(st_trc, st_gf, inferred_trace, learned_kernel_np, image, args, init_trc)
    else:
        plot_res(k, k_sub, image, learned_kernel_np, inferred_trace, stf0, gf, trc, args)
        plot_trace(trc, inferred_trace, args)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='args')
    # Configurations
    parser.add_argument('--btsize', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--num_epochs', type=int, default=20, metavar='N',
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
                        help='number of layers for kernel (default: 7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--Elr', type=float, default=1e-3,
                        help='learning rate(default: 1e-4)')
    parser.add_argument('--Mlr', type=float, default=1e-3,
                        help='learning rate(default: 1e-4)')

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
                        help='trace file name, npy array or obspy stream')
    parser.add_argument('--M0', type=float, default=None,
                        help='Earthquake M0')
    parser.add_argument('--egf', type=str, default='',
                        help='EGF file name, npy array or obspy stream')
    parser.add_argument('--M0_egf', type=float, default=None,
                        help='EGF M0, list if multiple EGFs')
    parser.add_argument('--num_egf', type=int, default=1, metavar='N',
                        help='number of EGF (default: 1)')
    parser.add_argument('--egf_qual_weight', type=float, default=None,
                        help='if multiple EGFs, weight reflects quality of each EGF (default None = 1 for each). ')
    parser.add_argument('--stf0', type=str, default='',
                        help='init STF file name')
    parser.add_argument('--stf_size', type=int, default=100, metavar='N',
                        help='length of STF (default: 100)')

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
    parser.add_argument('--seqfrac', type=int, default=8,
                        help='seqfrac (default:2), should be < to stf length')

    # parameters
    parser.add_argument('--data_sigma', type=float, default=5e-5,
                        help='data sigma (default: 5e-5)')
    parser.add_argument('--px_init_weight', type=float, default=None,
                        help='weight on init STF on E step prior, default 0')
    parser.add_argument('--px_weight', type=float, nargs='+', default=None,
                        help='weight on E step priors, list (default None = function of data_sigma)')
    parser.add_argument('--logdet_weight', type=float, default=None,
                        help='β, controls entropy, E step prior (default None = function of data_sigma)')
    parser.add_argument('--kernel_norm_weight', type=float, default=None,
                        help='kernel norm weight + weight on TV, M step (default None = function of data_sigma)')
    parser.add_argument('--kernel_corrcoef_weight', type=float, default=None,
                        help='kernel correlation coef weight if multiple EGF, M step (default None = function of data_sigma)')
    parser.add_argument('--prior_phi_weight', type=float, default=None,
                        help='weight on init GF on M step (default None = function of data_sigma)')
    parser.add_argument('--egf_multi_weight', type=float, default=None,
                        help='if multiple EGFs, weight for multi-M-step priors. ')

    args = parser.parse_args()

    if os.uname().nodename == 'wouf':
        matplotlib.use('TkAgg')
        args.dir = '/home/thea/projet/2023_EGF/deconvEgf_res/borr_test/'
        args.trc = "/home/thea/projet/2023_EGF/borrego_springs/data/dg/BOR_m5_trc.mseed"
        args.egf = "/home/thea/projet/2023_EGF/borrego_springs/data/dg/BOR_m3_trc.mseed"
        # args.M0_egf = 1.27e12
        # args.M0 = 1.27e15
        # args.stf0 ="/home/thea/projet/EGF/cahuilla/semisynth/multi_semisy8_CSH_stf_true.npy"
        # args.gf_true = "/home/thea/projet/EGF/cahuilla/semisynth/data/multi_semisy8_CSH_gf_true.npy"
        # args.stf_true = "/home/thea/projet/EGF/cahuilla/semisynth/data/multi_semisy8_CSH_stf_true.npy"
        # args.dir = '/home/thea/projet/EGF/deconvEgf_res/multiM_semisy8_CSH_xx/'
        # args.trc = "/home/thea/projet/EGF/cahuilla/semisynth/data/multi_semisy8_CSH_trc_detrend.mseed"
        # args.egf = "/home/thea/projet/EGF/cahuilla/semisynth/data/multi_semisy8_CSH_m2_gf.mseed"
        # args.stf0 ="/home/thea/projet/EGF/cahuilla/semisynth/data/multi_semisy8_CSH_stf_true.npy"
        # args.gf_true = "/home/thea/projet/EGF/cahuilla/semisynth/multi_semisy8_CSH_gf_true.npy"
        # args.stf_true = "/home/thea/projet/EGF/cahuilla/semisynth/multi_semisy8_CSH_stf_true.npy"
        # args.dir = '/home/thea/projet/EGF/deconvEgf_res/synth_2a0_ampl/'
        # args.trc = "/home/thea/projet/EGF/synth_wf/data/2a0_m1_rec0_trc.npy"
        # args.egf = "/home/thea/projet/EGF/synth_wf/data/2a0_m0_rec0_gf.npy"
        # # args.stf0 ="/home/thea/projet/EGF/synth_wf/data/2a0_m1_rec0_stf_true.npy"
        # args.gf_true = "/home/thea/projet/EGF/synth_wf/data/2a0_m1_rec0_gf.npy"
        # args.stf_true = "/home/thea/projet/EGF/synth_wf/data/2a0_m1_rec0_stf_true.npy"
        args.output = True
        args.synthetics = False
        args.num_egf = 1
        args.btsize = 24
        args.num_subepochsE = 1
        args.num_subepochsM = 1
        args.num_epochs = 1
        args.seqfrac = 20
        args.stf_size = 250 #180
        # args.egf_qual_weight = [0.5, 0.5, 0.5]
        # args.px_init_weight = 5e4

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
        args.num_epochs = args.num_epochs + 11
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

    main_function(args)
