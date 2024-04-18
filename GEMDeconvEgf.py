#!/usr/bin/env python
# coding: utf-8

from deconvEgf_helpers import *

def main_function(args):
    
    start_time = time.time()

    ################################################ SET UP WEIGHTS ################################################

    args.phi_weight = 1e-1

    if args.stf_init_weight is None: # weight on init STF
        args.stf_init_weight = (1/args.data_sigma)/2e1

    if args.stf_weight is None: # weights for priors on Estep: list, [boundaries, TV, L1]
        args.stf_weight = [(1/args.data_sigma)/2e2, (1/args.data_sigma)/4e1, (1/args.data_sigma)/2e2]

    if args.logdet_weight is None: # weight on q_theta
        args.logdet_weight = (1/args.data_sigma)/1e2

    if args.prior_phi_weight is None: # weights for priors on Mstep: list, [L1, L2, TV]
        args.prior_phi_weight = [(1/args.data_sigma)/6e3, (1/args.data_sigma)/4e4, (1/args.data_sigma)/4e4]

    if args.egf_norm_weight is None:
        args.egf_norm_weight = (1/args.data_sigma)/1e6

    if args.num_egf > 1:
        if args.egf_multi_weight is None:
            args.egf_multi_weight = (1/args.data_sigma)/4e4
        if args.egf_qual_weight is None:
            args.egf_qual_weight = np.ones(args.num_egf).tolist()
    else:
        args.egf_multi_weight = 0.
        args.egf_qual_weight = [1]


    ################################################ SET UP DATA ################################################
    
    # Traces
    try:
        st_trc = obspy.read("{}".format(args.trc0))
        trc0 = np.concatenate([st_trc[k].data[:, None] for k in range(len(st_trc))], axis=1).T
    except TypeError:
        st_trc = None
        trc0 = np.load("{}".format(args.trc0))

    # EGF
    try:
        st_gf = obspy.read("{}".format(args.egf0))
        gf0 = np.concatenate([st_gf[k].data[:, None] for k in range(len(st_gf))], axis=1).T
        # Order in stream is: all traces E, all traces N, all traces Z
        gf0 = gf0.reshape(gf0.shape[0]//3, 3, gf0.shape[1], order='F')
        args.samp_rate = 1 / st_gf[0].stats['delta']
    except TypeError:
        st_gf = None
        gf0 = np.load("{}".format(args.egf0))
        gf0 = gf0.reshape(gf0.shape[0]//3, 3, gf0.shape[1])

    # STF
    if args.stf_dur is not None and st_gf is not None:
        len_stf = int(args.stf_dur / st_gf[0].stats['delta'])
        args.stf_size = len_stf
    elif args.stf_dur is not None and args.samp_rate is not None:
        len_stf = int(args.stf_dur * args.samp_rate)
        args.stf_size = len_stf
    else:
        len_stf = args.stf_size

    if len(args.stf0) > 0:
        try:
            st_stf0 = obspy.read("{}".format(args.stf0))
            stf0 = st_stf0[0].data
        except TypeError:
            st_stf0 = None
            stf0 = np.load("{}".format(args.stf0))
        if len_stf > len(stf0):
            stf_rs = np.zeros(len_stf)
            stf_rs[(len(stf_rs) - len(stf0)) // 2:-(len(stf_rs) - len(stf0)) // 2] = stf0
            stf0 = stf_rs
        elif len_stf < len(stf0):
            len_stf = len(stf0)
            print('STF size set to length of STF0')
    else:
        τc = len_stf // 30 ## TODO function of rate... M0 ?
        stf0 = 0.01 * np.ones(len_stf) + 0.99 * np.exp(-(np.arange(len_stf) - len_stf//2)**2 / (2 * (τc/2)**2))

    # Synthetics
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
        if len_stf > len(stf_true):
            stf_rs = np.zeros(len_stf)
            stf_rs[(len(stf_rs) - len(stf_true)) // 2:-(len(stf_rs) - len(stf_true)) // 2] = stf_true
            stf_true = stf_rs
        elif len_stf < len(stf_true):
            len_stf = len(stf_true)

    # Normalize
    gf0 = gf0 / np.amax(np.abs(gf0))
    gf0 = torch.Tensor(gf0).to(device=args.device)

    stf0 = stf0 / np.amax(stf0)
    stf0 = torch.Tensor(stf0).to(device=args.device)

    init_trc = trueForward(gf0, stf0.view(1, 1, -1), args.num_egf)
    trc0 /= np.amax(np.abs(trc0))
    trc0 = torch.Tensor(trc0).to(device=args.device)
    trc_ext = torch.Tensor(trc0).to(device=args.device) # will have a btsize format


    ################################################ MODEL SETUP ################################################
    
    n_flow = 32
    affine = True
    seqfrac = args.seqfrac

    # EGF initialization
    gf_network = [GFNetwork(gf0[i],
                            num_layers=args.num_layers,
                            num_egf=1,
                            device=args.device,
                            ).to(args.device) for i in range(args.num_egf)]

    # STF initialization
    if args.reverse:
        print("Generating Reverse RealNVP Network")
        permute = 'reverse'
    else:
        print("Generating Random RealNVP Network")
        permute = 'random'
    realnvp = realnvpfc_model.RealNVP(len_stf, n_flow, seqfrac=seqfrac, affine=affine, permute=permute).to(args.device)
    stf_gen = stf_generator(realnvp).to(args.device)

    # True forward model (with init GF), used for priors
    FTrue = lambda x: trueForward(torch.unsqueeze(gf0, dim=0), x, args.num_egf)

    print("Models Initialized")

    # Multiple GPUs
    if len(args.device_ids) > 1:
        stf_gen = nn.DataParallel(stf_gen, device_ids=args.device_ids)
        stf_gen.to(args.device)
        for i in range(args.num_egf):
            gf_network[i] = nn.DataParallel(gf_network[i], device_ids=args.device_ids)
            gf_network[i].to(args.device)

    # Priors on Mstep (EGF)
    if len(args.device_ids) > 1:
        gf_softl1 = lambda gf_network: torch.abs(1 - torch.sum(gf_network.module.generategf()))
    else:
        gf_softl1 = lambda gf_network: torch.abs(1 - torch.sum(gf_network.generategf()))

    f_phi_prior = lambda gf: priorPhi(gf, gf0)

    if args.num_egf == 1:
        prior_L2 = lambda gf, weight : weight * (0.5 * Loss_DTW_Mstep(gf, gf0) + Loss_L2(gf, gf0)) if weight > 0 else 0
        prior_TV = lambda gf, weight: weight * Loss_TV(gf)
    else:
        prior_L2 = lambda gf, weight, i : weight * (0.5 * Loss_DTW_Mstep(gf, gf0[i].unsqueeze(0)) + Loss_L2(gf, gf0[i].unsqueeze(0))) if weight > 0 else 0
        prior_TV = lambda gf, weight,i: weight * Loss_TV(gf)
    phi_priors = [f_phi_prior, prior_L2, prior_TV] # norms on init GF

    # Priors on Estep (STF)
    stf_softl1 = lambda stf, weight: torch.abs(1 - torch.sum(stf))
    prior_boundary = lambda stf, weight: weight * torch.sum(torch.abs(stf[:, :, 0]) * torch.abs(stf[:, :, -1]))
    prior_dtw = lambda stf, weight: weight * (Loss_L2(stf, stf0) + Loss_DTW(stf, stf0)) if weight > 0 else 0
    prior_TV_stf = lambda stf, weight: weight * Loss_TV(stf)

    flux = torch.abs(torch.sum(stf0))
    logscale_factor = stf_logscale(scale=flux/(0.8*stf0.shape[0]), device=args.device).to(args.device)
    logdet_weight = args.logdet_weight #flux/(len_stf*args.data_sigma)
    prior_x = prior_dtw
    prior_stf = [prior_boundary, prior_TV_stf, stf_softl1]  # prior on STF, can be a list

    # Optimizers (Adam)
    Eoptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(stf_gen.parameters())
                                         + list(logscale_factor.parameters())), lr=args.Elr)
    Moptimizer = [torch.optim.Adam(filter(lambda p: p.requires_grad, list(gf_network[i].parameters())),
                                   lr=args.Mlr) for i in range(args.num_egf) ]


    ################################################ TRAINING ################################################

    Eloss_list = []
    Eloss_prior_list = []
    Eloss_mse_list = []
    Eloss_q_list = []

    Mloss_list = {}
    Mloss_mse_list = {}
    Mloss_multi_list = {}
    Mloss_gfnorm_list = {}
    Mloss_phiprior_list = {}
    Mloss = {}

    for k_egf in range(args.num_egf):
        Mloss_list[k_egf] = []
        Mloss_mse_list[k_egf] = []
        Mloss_multi_list[k_egf] = []
        Mloss_gfnorm_list[k_egf] = []
        Mloss_phiprior_list[k_egf] = []

    ## Initialize
    if args.num_egf == 1:
        z_sample = torch.randn(2, len_stf).to(device=args.device)
        stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor,
                               device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
        stf_np = stf.detach().cpu().numpy()
        y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
        inferred_trace = [y0.detach().cpu().numpy() for y0 in y]
    else:
        z_sample = torch.randn(args.btsize, len_stf).to(device=args.device)

        stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor,
                               device=args.device, device_ids=args.device_ids if len(args.device_ids)>1 else None)
        stf_np = stf.detach().cpu().numpy()
        y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
        inferred_trace = [ y0.detach().cpu().numpy() for y0 in y ]

    learned_gf = [gf_network[i].module.generategf() for i in range(args.num_egf)] \
        if len(args.device_ids) > 1 else [gf_network[i].generategf() for i in range(args.num_egf)]

    learned_gf_np = [learned_gf[i].detach().cpu().numpy()[0] for i in range(args.num_egf)]
    
    if args.output:
        if args.synthetics:
            plot_res(0, 0, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args, true_gf=gf_true, true_stf=stf_true)
        else:
            plot_res(0, 0, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args)

    trc_ext = torch.cat(args.btsize*[trc_ext.unsqueeze(0)])

    # Save args
    with open("{}/args.json".format(args.PATH), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


    ############################################# RUN ###########################################################
    
    print("Starting iterations")

    for k in range(args.num_epochs):

        ############################ Estep - Update STF network ############################

        for k_sub in range(args.num_subepochsE):

            z_sample = torch.randn(args.btsize, len_stf).to(device=args.device)

            Eloss, qloss, priorloss, mseloss = EStep(z_sample, trc_ext, stf_gen, gf_network,
                                                     prior_x, prior_stf, logdet_weight,
                                                     len_stf, logscale_factor, args)

            Eloss_list.append(Eloss.detach().cpu().numpy())
            Eloss_prior_list.append(priorloss.detach().cpu().numpy())
            Eloss_q_list.append(qloss.detach().cpu().numpy())
            Eloss_mse_list.append(mseloss.detach().cpu().numpy())
            Eoptimizer.zero_grad()
            Eloss.backward()
            nn.utils.clip_grad_norm_(list(stf_gen.parameters()) + list(logscale_factor.parameters()), 1)
            Eoptimizer.step()

            if (args.EMFull and (k_sub % args.print_every == 0)) or (not args.EMFull and (k % args.print_every == 0)):
                
                print()
                print(f"Estep ----- epoch {k:}, subepoch {k_sub:}")
                print(f"Loss  ----- tot {Eloss_list[-1]:.2f}, prior {Eloss_prior_list[-1]:.2f}, q {Eloss_q_list[-1]:.2f}, mse {Eloss_mse_list[-1]:.2f}")

            if args.output and ((args.EMFull and (k_sub % args.save_every == 0)) or (not args.EMFull and (k % args.save_every == 0))):

                z_sample = torch.randn(args.btsize, len_stf).to(device=args.device)
                stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor,
                                    device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
                stf_np = stf.detach().cpu().numpy()

                learned_gf = [gf_network[i].module.generategf().detach() for i in range(args.num_egf)] \
                    if len(args.device_ids) > 1 else [gf_network[i].generategf().detach() for i in range(args.num_egf)]
                y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
                inferred_trace = [y[i].detach().cpu().numpy() for i in range(args.num_egf)]
                learned_gf_np = [learned_gf[i].cpu().numpy()[0] for i in range(args.num_egf)]

                # Save PyTorch model
                with torch.no_grad():
                    torch.save({
                        'epoch': k,
                        'model_state_dict': stf_gen.state_dict(),
                        'optimizer_state_dict': Eoptimizer.state_dict(),
                    }, '{}/stf_gen_{}_E{}.pt'.format(args.PATH, str(k).zfill(5), str(k_sub).zfill(5)))
                
                # Save STF
                np.save("{}/Data/stf.npy".format(args.PATH), learned_gf_np)

                # Plots
                for k_egf in range(args.num_egf):
                    plot_seploss(args,
                                 Eloss_list, Eloss_mse_list, Eloss_prior_list, Eloss_q_list,
                                 Mloss_list, Mloss_mse_list, Mloss_phiprior_list, Mloss_multi_list,
                                 k_egf)
                
                if args.synthetics:
                    plot_res(k, k_sub, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args, true_gf=gf_true, true_stf=stf_true, step='E')
                else:
                    plot_res(k, k_sub, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args, step='E')

        ############################ Mstep - Update GF network ############################

        for k_sub in range(args.num_subepochsM):

            z_sample = torch.randn(args.btsize, len_stf).to(device=args.device)
            x_sample = torch.randn(args.btsize, len_stf).to(device=args.device).reshape((-1, 1, len_stf))

            Mloss, mse, gfnorm, priorphi, multiloss = MStep(z_sample, x_sample, len_stf,
                                                            trc_ext,
                                                            stf_gen, gf_network,
                                                            FTrue, logscale_factor,
                                                            phi_priors, gf_softl1,
                                                            args)

            for k_egf in range(args.num_egf):

                Mloss_list[k_egf].append(Mloss[k_egf].detach().cpu().numpy())
                Mloss_mse_list[k_egf].append(mse[k_egf].detach().cpu().numpy())
                Mloss_multi_list[k_egf].append(multiloss.detach().cpu().numpy())
                Mloss_phiprior_list[k_egf].append(priorphi[k_egf].detach().cpu().numpy())
                Mloss_gfnorm_list[k_egf].append(gfnorm[k_egf].detach().cpu().numpy())
                Moptimizer[k_egf].zero_grad()
                Mloss[k_egf].backward(retain_graph=True)
                nn.utils.clip_grad_norm_(list(gf_network[k_egf].parameters()), 1)
                Moptimizer[k_egf].step()

                if (args.EMFull and (k_sub % args.print_every == 0)) or (not args.EMFull and (k % args.print_every == 0)):
                    
                    print()
                    print(f"Mstep ----- epoch {k:}, subepoch {k_sub:}, egf {k_egf:}")
                    print(f"Loss  ----- tot {Mloss_list[k_egf][-1]:.2f}, phi_prior {Mloss_phiprior_list[k_egf][-1]:.2f}, norm {Mloss_gfnorm_list[k_egf][-1]:.2f}, mse {Mloss_mse_list[k_egf][-1]:.2f}, multi {Mloss_multi_list[k_egf][-1]:.2f}")

                if args.output and ((args.EMFull and (k_sub % args.save_every == 0)) or (not args.EMFull and (k % args.save_every == 0))):

                    z_sample = torch.randn(args.btsize, len_stf).to(device=args.device)
                    stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor,
                                        device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
                    stf_np = stf.detach().cpu().numpy()

                    learned_gf = [gf_network[i].module.generategf().detach() for i in range(args.num_egf)] \
                        if len(args.device_ids) > 1 else [gf_network[i].generategf().detach() for i in range(args.num_egf)]
                    y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
                    inferred_trace = [y[i].detach().cpu().numpy() for i in range(args.num_egf)]
                    learned_gf_np = [learned_gf[i].cpu().numpy()[0] for i in range(args.num_egf)]

                    # Save PyTorch model
                    with torch.no_grad():
                        torch.save({
                            'epoch': k,
                            'model_state_dict': gf_network[k_egf].state_dict(),
                            'optimizer_state_dict': Moptimizer[k_egf].state_dict(),
                        }, '{}/egf_network_egf{}_{}_M{}.pt'.format(args.PATH, str(k_egf), str(k).zfill(5), str(k_sub).zfill(5)))
                    
                    # Save EGF
                    np.save("{}/Data/learned_gf.npy".format(args.PATH), learned_gf_np)

                    # Plots
                    plot_seploss(args,
                                 Eloss_list, Eloss_mse_list, Eloss_prior_list, Eloss_q_list,
                                 Mloss_list, Mloss_mse_list, Mloss_phiprior_list, Mloss_multi_list,
                                 k_egf)

                    if args.synthetics:
                        plot_res(k, k_sub, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args, true_gf=gf_true, true_stf=stf_true, step='M')
                    else:
                        plot_res(k, k_sub, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args, step='M')


    ############################################# GENERATE OUTPUT FIGURES ###########################################################

    learned_gf = [gf_network[i].module.generategf().detach() for i in range(args.num_egf)] \
            if len(args.device_ids) > 1 else [gf_network[i].generategf().detach() for i in range(args.num_egf)]
    z_sample = torch.randn(args.btsize, len_stf).to(device=args.device)
    stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor,
                           device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
    stf_np = stf.detach().cpu().numpy()
    y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
    inferred_trace = [y[i].detach().cpu().numpy() for i in range(args.num_egf)]
    learned_gf_np = np.array([learned_gf[i].cpu().numpy()[0] for i in range(args.num_egf)])
    
    print(stf_np.shape)
    
    # Scale stf area with M0
    if args.samp_rate is not None and args.M0 is not None:        
        area = np.trapz(y=stf_np, dx=.1, axis=-1)[..., np.newaxis]
        area = area.repeat(stf_np.shape[-1], axis=-1)
        stf_np /= area
        stf_np *= args.M0

    np.save("{}/Data/reconSTF.npy".format(args.PATH), stf_np)

    if st_trc is not None:
        st_trc_sd = st_trc.copy()
        st_trc_mn = st_trc.copy()
        for i in range(3):
            st_trc_mn[i].data = np.mean(inferred_trace, axis=(0,1))[i]
            st_trc_sd[i].data = np.std(inferred_trace, axis=(0,1))[i]
        st_trc_mn.write("{}/{}_out_mean.mseed".format(args.PATH, args.trc0.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
        st_trc_sd.write("{}/{}_out_std.mseed".format(args.PATH, args.trc0.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
    np.save("{}/Data/outTRC.npy".format(args.PATH), inferred_trace)

    if st_gf is not None:
        st_gf_out = st_gf.copy()
        lk = learned_gf_np.reshape(args.num_egf*3, learned_gf_np.shape[-1])
        for i in range(len(lk)):
            st_gf_out[i].data = lk[i, :]
        st_gf_out.write("{}/{}_out.mseed".format(args.PATH, args.egf0.rsplit("/", 1)[1].rsplit(".", 1)[0]) )
    np.save("{}/Data/outGF.npy".format(args.PATH), learned_gf_np)

    # Plot
    plot_seploss(args,
                 Eloss_list, Eloss_mse_list, Eloss_prior_list, Eloss_q_list,
                 Mloss_list, Mloss_mse_list, Mloss_phiprior_list, Mloss_multi_list,
                 k_egf)
    
    if args.synthetics and st_trc is None:
        plot_res(k, k_sub, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args, true_gf=gf_true, true_stf=stf_true)
        plot_trace(trc0, inferred_trace, args)

    elif args.synthetics and st_gf is not None and st_trc is not None:
        plot_res(k, k_sub, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args, true_gf=gf_true, true_stf=stf_true)
        plot_st(st_trc, st_gf, inferred_trace, learned_gf_np, stf_np, args, init_trc)

    elif st_gf is not None and st_trc is not None:
        plot_st(st_trc, st_gf, inferred_trace, learned_gf_np, stf_np, args, init_trc)

    else:
        plot_res(k, k_sub, stf_np, learned_gf_np, inferred_trace, stf0, gf0, trc0, args)
        plot_trace(trc0, inferred_trace, args)
    
    runtime = time.time() - start_time
    print()
    print("Runtime: {:0=2}h {:0=2}m {:02.0f}s".format(*[int(runtime//3600), int(runtime%3600//60), runtime%3600%60]))
    print()


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
    parser.add_argument('--EMFull', action='store_true', default=False,
                        help='True: E to convergence, M to convergence False: alternate E, M every epoch (default: False)')
    parser.add_argument('--num_layers', type=int, default=7, metavar='N',
                        help='number of layers for GF generator (default: 7)')
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
                        help='Output directory')
    parser.add_argument('--trc0', type=str, default='',
                        help='Path or name of trace file, npy array or obspy stream')
    parser.add_argument('--egf0', type=str, default='',
                        help='Path or name of EGF file, npy array or obspy stream')
    parser.add_argument('--stf0', type=str, default='',
                        help='init STF file name')
    parser.add_argument('--M0', type=float, default=None,
                        help='Main event seismic moment M0')
    parser.add_argument('--M0_egf', type=float, default=None,
                        help='EGF seismic moment(s) M0, list if multiple EGFs')
    parser.add_argument('--num_egf', type=int, default=1, metavar='N',
                        help='number of EGF (default: 1)')
    parser.add_argument('--samp_rate', type=float, default=None,
                        help='Sampling rate (Hz) of traces, gf and stf.')
    parser.add_argument('--stf_dur', type=float, default=None,
                        help='STF duration in seconds')
    parser.add_argument('--stf_size', type=int, default=100, metavar='N',
                        help='Length of STF (number of samples, default: 100)')

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

    # Weight parameters
    parser.add_argument('--data_sigma', type=float, default=5e-5,
                        help='data sigma (default: 5e-5)')
    parser.add_argument('--stf_init_weight', type=float, default=None,
                        help='weight on init STF on E step prior, default 0')
    parser.add_argument('--stf_weight', type=float, nargs='+', default=None,
                        help='weight on E step priors, list (default None = function of data_sigma)')
    parser.add_argument('--logdet_weight', type=float, default=None,
                        help='β, controls entropy, E step prior (default None = function of data_sigma)')
    parser.add_argument('--egf_norm_weight', type=float, default=None,
                        help='EGF norm weight, M step (default None = function of data_sigma)')
    parser.add_argument('--prior_phi_weight', type=float, default=None,
                        help='weight on init GF on M step (default None = function of data_sigma)')
    parser.add_argument('--egf_multi_weight', type=float, default=None,
                        help='if multiple EGFs, weight to closeness of EGFs to best EGF (the one that minimizes the fit to the data). ')
    parser.add_argument('--egf_qual_weight', type=float, default=None,
                        help='if multiple EGFs, weights the Mstep MSE loss of each EGFs (default None = 1 for each). ')

    args = parser.parse_args()
    
    print()
    print("############################################################")
    print("#                                                          #")
    print("#     DeepGEM: Generalized Expectation-Maximization        #")
    print("#             for Empirical Green's Functions              #")
    print("#                                                          #")
    print("############################################################")
    print()

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

    print("cuda available ", torch.cuda.is_available())
    print("---> num gpu", torch.cuda.device_count())
    print('---> using device:', args.device)
    if torch.cuda.is_available():
        print(" Using {} GPUS".format(len(args.device_ids)))

    if args.EMFull:
        args.num_epochs = args.num_epochs + 1
        args.num_subepochsE = args.num_subepochsE + 1
        args.num_subepochsM = args.num_subepochsM + 1
        print("Full EM w/ {} epochs and {} E subepochs {} M subepochs".format(args.num_epochs,
                                                                              args.num_subepochsE,
                                                                              args.num_subepochsM))
    else:
        args.num_epochs = args.num_epochs + 1
        args.num_subepochsE = 1
        args.num_subepochsM = 1
        print("Stochastic EM w/ {} epochs and {} E subepochs {} M subepochs".format(args.num_epochs,
                                                                                    args.num_subepochsE,
                                                                                    args.num_subepochsM))

    # Create target directories
    try:
        os.mkdir(args.PATH)
        print("Directory ", args.PATH, " Created ")
    except FileExistsError:
        print("Directory ", args.PATH, " already exists")

    try:
        os.mkdir(args.PATH + "/Data")
        print("Directory ", args.PATH + "/Data", " Created ")
    except FileExistsError:
        print("Directory ", args.PATH + "/Data", " already exists")

    main_function(args)
