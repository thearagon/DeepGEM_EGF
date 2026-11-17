#!/usr/bin/env python
# coding: utf-8

from deconvEgf_helpers import *


def main_function(args):
    ################################################ SET UP WEIGHTS ################################################

    # weight on q_theta
    args.logdet_weight = 1
    
    # weights for priors on Estep: list, [boundaries, TV]
    args.stf_weight = args.stf_weight or [(1 / args.data_sigma) *5e-1,
                                          (1 / args.data_sigma) * 1e-1]
    
    # weights for priors on Mstep: list, [L1, L2]
    args.prior_phi_weight = args.prior_phi_weight or [(1 / args.data_sigma) *1e2,
                                                      (1 / args.data_sigma) *1e2]
    
    # if multiple EGFs
    if args.num_egf > 1:
        args.egf_multi_weight = args.egf_multi_weight or (1 / args.data_sigma) / 4e4
        args.egf_qual_weight = args.egf_qual_weight or np.ones(args.num_egf).tolist()
    else:
        args.egf_multi_weight = 0.0
        args.egf_qual_weight = [1.0]

    ################################################ SET UP DATA ################################################

    def load_data(file_path, reshape_dims=None):
        if file_path.endswith(".npy"):
            data, array, sr = None, np.load(file_path), None
        else:
            data = obspy.read(file_path)  
            array = np.concatenate([trace.data[:, None] for trace in data], axis=1).T
            sr = 1 / data[0].stats['delta']
        if reshape_dims:
            array = array.reshape(reshape_dims, order='F')
        return data, array, sr

    # traces
    st_trc, trc0, _ = load_data(args.trc0)
    args.c_trc = float(np.amax(np.abs(trc0)))  # amplitude factor
    trc0 = torch.Tensor(trc0 / args.c_trc).to(args.device)
    args.num_comp = trc0.shape[0]

    # EGF
    st_gf, gf0, sr = load_data(args.egf0, reshape_dims=(args.num_egf, args.num_comp, -1))
    args.samp_rate = sr or args.samp_rate
    c_gf = np.amax(np.abs(gf0), axis=(1, 2), keepdims=True)
    args.c_gf = c_gf.ravel().tolist() # amplitude factor
    gf0 = torch.Tensor(gf0 / args.c_gf[0]).to(args.device)

    # Determine STF length
    len_stf = int(args.stf_dur * args.samp_rate) if args.stf_dur and args.samp_rate else args.stf_size
    args.stf_size = len_stf

    # STF
    if args.stf0:
        _, stf0, sr0 = load_data(args.stf0)
        stf0 = np.asarray(stf0, dtype=float).ravel()
        if sr0 and args.samp_rate and sr0 != args.samp_rate:          
            t0 = np.arange(len(stf0)) / sr0
            stf0 = np.interp(np.arange(0, t0[-1], 1 / args.samp_rate), t0, stf0)
        d = len_stf - len(stf0)                                      
        if d > 0:
            stf0 = np.pad(stf0, (d // 2, d - d // 2))
        elif d < 0:
            stf0 = stf0[(-d) // 2:(-d) // 2 + len_stf]
    
    else:
        τc = len_stf // 10
        stf0 = 0.01 + 0.99 * np.exp(-(np.arange(len_stf) - len_stf // 2) ** 2 / (2 * (τc / 2) ** 2))
    stf0 = torch.Tensor(stf0 / np.amax(stf0)).to(args.device)

    ## initial start for amplitude estimates
    with torch.no_grad():
        A, A_ls = [], []
        for i in range(args.num_egf):
            p = conv_same(gf0[i][:, None, :], stf0.view(1, 1, -1))[:, 0, :]        # (C, L)
            A_ls.append(float((p * trc0).sum() / (p ** 2).sum()))                 # timing-sensitive, can be < 0
            A.append(float(torch.sqrt((trc0 ** 2).sum() / (p ** 2).sum())))     # energy ratio, always > 0
    args.amp = A
    args.amp0 = A[0]    

    trc_ext = trc0.clone()
    gf0_detached = gf0.detach().cpu().numpy()
    trc0_detached = trc0.detach().cpu().numpy()

    # Synthetics
    if args.synthetics == True:
        st_gf_true, gf_true, _ = load_data(args.gf_true, reshape_dims=(args.num_comp, gf0.shape[-1]))
        gf_true = torch.Tensor(gf_true / np.amax(np.abs(gf_true))).to(args.device)
        stf_true = np.load("{}".format(args.stf_true))
        if len_stf > len(stf_true):
            stf_rs = np.zeros(len_stf)
            stf_rs[(len(stf_rs) - len(stf_true)) // 2:-(len(stf_rs) - len(stf_true)) // 2] = stf_true
            stf_true = stf_rs
        elif len_stf < len(stf_true):
            c = (len(stf_true) - len_stf) // 2
            stf_true = stf_true[c:c + len_stf]

    ################################################ MODEL SETUP ################################################

    n_flow = 32
    affine = True
    seqfrac = args.seqfrac
    args.sigma = [args.data_sigma] * args.num_comp


    # EGF initialization
    gf_network = [GFNetwork(gf0[i],
                            num_layers=args.num_layers,
                            num_egf=1,
                            num_comp=args.num_comp,
                            device=args.device,
                            log_amp=np.log(A[i] / A[0]),
                            learn_amp=(i != 0)
                            ).to(args.device) for i in range(args.num_egf)]

    # STF initialization
    permute = 'reverse' if args.reverse else 'random'
    print(f"Generating {'Reverse' if args.reverse else 'Random'} RealNVP Network")
    realnvp = realnvpfc_model.RealNVP(len_stf, n_flow, seqfrac=seqfrac, affine=affine, permute=permute).to(args.device)
    stf_gen = stf_generator(realnvp).to(args.device)
    

    # True forward model (with init GF), used for priors
    FTrue = lambda x, with_amp=True: trueForward(torch.unsqueeze(gf0, dim=0), x, args.num_egf, args.num_comp)

    print("Models Initialized")

    # Multiple GPUs
    if len(args.device_ids) > 1:
        stf_gen = nn.DataParallel(stf_gen, device_ids=args.device_ids)
        gf_network = [nn.DataParallel(network, device_ids=args.device_ids).to(args.device) for network in gf_network]

    # Priors on Mstep (EGF)
    g0 = lambda i: gf0[i].unsqueeze(0)  
    f_phi_prior = lambda gf, w, i: w * priorPhi(gf, g0(i))
    prior_L2 = lambda gf, w, i: w * Loss_L2(gf, g0(i)) if w > 0 else 0
    phi_priors = [f_phi_prior, prior_L2] # norms on init GF

    # Priors on Estep (STF)
    stf0_ext = (args.amp0 * stf0).view(1, 1, -1)
    prior_x = lambda stf: 0.5 * torch.sum((stf - stf0_ext) ** 2, dim=(1, 2)) / (args.amp0 * args.stf0_sigma) ** 2 

    prior_stf = [
            lambda stf, w: w * torch.mean(torch.abs(stf[:, :, 0]) + torch.abs(stf[:, :, -1])) / args.amp0,              # boundary
            lambda stf, w: w * torch.mean(torch.sum(torch.abs(stf[:, :, 1:] - stf[:, :, :-1]), dim=-1)) / args.amp0,   # TV
        ]


    flux = torch.abs(torch.sum(stf0))
    logscale_factor = stf_logscale(scale=torch.tensor(args.amp0), device=args.device).to(args.device)

    # Optimizers (Adam)
    Eoptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(stf_gen.parameters())
                                         + list(logscale_factor.parameters())), lr=args.Elr)
    Moptimizer = [torch.optim.Adam(filter(lambda p: p.requires_grad, list(gf_network[i].parameters())),
                                   lr=args.Mlr) for i in range(args.num_egf)]

    ################################################ TRAINING ################################################

    Eloss_list, Eloss_prior_list, Eloss_mse_list, Eloss_q_list = [], [], [], []
    Mloss_list = {k_egf: [] for k_egf in range(args.num_egf)}
    Mloss_mse_list = {k_egf: [] for k_egf in range(args.num_egf)}
    Mloss_multi_list = {k_egf: [] for k_egf in range(args.num_egf)}
    Mloss_phiprior_list = {k_egf: [] for k_egf in range(args.num_egf)}

    # Initialize
    z_sample_template = torch.randn(args.btsize, len_stf, device=args.device)
    x_sample_template = torch.randn(args.btsize, len_stf, device=args.device).reshape((-1, 1, len_stf))
    trc_ext_batched = trc_ext.unsqueeze(0).expand(args.btsize, -1, -1).contiguous()

    # Save args
    with open(f"{args.PATH}/args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    ############################################# RUN ###########################################################

    print("Starting iterations")
    for k in range(args.num_epochs):

        ############################ Estep - Update STF network ############################

        for k_sub in range(args.num_subepochsE):
            z_sample = torch.randn_like(z_sample_template)

            Eloss, qloss, priorloss, mseloss = EStep(z_sample, trc_ext_batched, stf_gen, gf_network,
                                                     prior_x, prior_stf, len_stf, logscale_factor, args)

            Eloss_list.append(Eloss.item())
            Eloss_prior_list.append(priorloss.item())
            Eloss_q_list.append(qloss.item())
            Eloss_mse_list.append(mseloss.item())

            Eoptimizer.zero_grad()
            Eloss.backward()
            nn.utils.clip_grad_norm_(
                list(stf_gen.parameters()) + list(logscale_factor.parameters()), max_norm=1.0
            )
            Eoptimizer.step()

            if (k % args.print_every == 0) and (k_sub % 100 == 0 and (k != 0)):
                print(f"\nEstep ----- Epoch {k}, Subepoch {k_sub}")
                print(f"Loss ----- Total: {Eloss_list[-1]:.2f}, Prior: {Eloss_prior_list[-1]:.2f}, "
                      f"Q: {Eloss_q_list[-1]:.2f}, MSE: {Eloss_mse_list[-1]:.2f}")

            if args.output and (k % args.save_every == 0 and (k_sub % 100 == 0 and (k != 0))):
                with torch.no_grad():
                    z_sample = torch.randn_like(z_sample_template)
                    stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor, device=args.device)
                    learned_gf = [gf_network[i].module.generategf().detach() if len(args.device_ids) > 1
                        else gf_network[i].generategf().detach()
                        for i in range(args.num_egf)]
                    y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
                    inferred_trace = [y_i.cpu().numpy() for y_i in y]
                    learned_gf_detached = [gf.cpu().numpy()[0] for gf in learned_gf]

                    torch.save({
                        'epoch': k,
                        'model_state_dict': stf_gen.state_dict(),
                        'optimizer_state_dict': Eoptimizer.state_dict(),
                    }, f'{args.PATH}/stf_gen_{str(k).zfill(5)}_E{str(k_sub).zfill(5)}.pt')
                    np.save(f"{args.PATH}/Data/stf.npy", stf.cpu().numpy())

                    for k_egf in range(args.num_egf):
                        plot_seploss(args, Eloss_list, Eloss_mse_list, Eloss_prior_list, Eloss_q_list,
                            Mloss_list, Mloss_mse_list, Mloss_phiprior_list, Mloss_multi_list, k_egf)

                    plot_res(k, k_sub, stf.cpu().numpy(), learned_gf_detached, inferred_trace,
                        gf0_detached, trc0_detached, args,
                        true_gf=gf_true if args.synthetics else None,
                        true_stf=stf_true if args.synthetics else None,
                        step='E')

        ############################ Mstep - Update GF network ############################
        for k_sub in range(args.num_subepochsM):
            z_sample = torch.randn_like(z_sample_template)
            x_sample = torch.randn_like(x_sample_template)

            Mloss, mse, priorphi, multiloss = MStep(z_sample, x_sample, len_stf,
                                                    trc_ext_batched,
                                                    stf_gen, gf_network,
                                                    FTrue, logscale_factor,
                                                    phi_priors, args)

            for k_egf in range(args.num_egf):
                Mloss_list[k_egf].append(Mloss[k_egf].item())
                Mloss_mse_list[k_egf].append(mse[k_egf].item())
                Mloss_multi_list[k_egf].append(multiloss[k_egf].item())
                Mloss_phiprior_list[k_egf].append(priorphi[k_egf].item())

                Moptimizer[k_egf].zero_grad()
                Mloss[k_egf].backward(retain_graph=True)
                nn.utils.clip_grad_norm_(gf_network[k_egf].parameters(), max_norm=1.0)
                Moptimizer[k_egf].step()

                if (k % args.print_every == 0) and (k != 0 and (k_sub % 100 == 0)):
                    print(f"\nMstep ----- Epoch {k}, Subepoch {k_sub}, EGF {k_egf}")
                    print(
                        f"Loss ----- Total: {Mloss_list[k_egf][-1]:.2f}, Phi_Prior: {Mloss_phiprior_list[k_egf][-1]:.2f}, "
                        f"MSE: {Mloss_mse_list[k_egf][-1]:.2f}, Multi: {Mloss_multi_list[k_egf][-1]:.2f}")

                if args.output and (k % args.save_every == 0 and (k != 0 and (k_sub % 100 == 0))):
                    with torch.no_grad():
                        z_sample = torch.randn_like(z_sample_template)
                        stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor, device=args.device,
                                               device_ids=args.device_ids if len(args.device_ids) > 1 else None)

                        learned_gf = [gf_network[i].module.generategf().detach() if len(args.device_ids) > 1
                            else gf_network[i].generategf().detach()
                            for i in range(args.num_egf)]
                        y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
                        inferred_trace = [y_i.cpu().numpy() for y_i in y]
                        learned_gf_detached = [gf.cpu().numpy()[0] for gf in learned_gf]

                        # Save PyTorch model
                        torch.save({
                            'epoch': k,
                            'model_state_dict': gf_network[k_egf].state_dict(),
                            'optimizer_state_dict': Moptimizer[k_egf].state_dict(),
                        }, f"{args.PATH}/egf_network_egf{k_egf}_{str(k).zfill(5)}_M{str(k_sub).zfill(5)}.pt")

                        # Save EGF
                        np.save(f"{args.PATH}/Data/learned_gf.npy", learned_gf_detached)

                        # Plots
                        plot_seploss(args,
                                     Eloss_list, Eloss_mse_list, Eloss_prior_list, Eloss_q_list,
                                     Mloss_list, Mloss_mse_list, Mloss_phiprior_list, Mloss_multi_list,
                                     k_egf)

                        plot_res(k, k_sub, stf.cpu().numpy(), learned_gf_detached, inferred_trace,
                                 gf0_detached, trc0_detached, args,
                                 true_gf=gf_true if args.synthetics else None,
                                 true_stf=stf_true if args.synthetics else None,
                                 step='M')

    ############################################# GENERATE OUTPUT FIGURES ###########################################################

    print("Done, printing results")
    learned_gf = torch.stack(
        [gf_network[i].module.generategf().detach() if len(args.device_ids) > 1 else gf_network[i].generategf().detach()
         for i in range(args.num_egf)], dim=0)
    z_sample = torch.randn_like(z_sample_template)
    stf, logdet = GForward(z_sample, stf_gen, len_stf, logscale_factor,
                       device=args.device, device_ids=(args.device_ids if len(args.device_ids) > 1 else None))
    stf_detached = stf.detach().cpu().numpy()
    with torch.no_grad():
        y = torch.stack([gf_network[i](stf) for i in range(args.num_egf)], dim=0)    # (E, B, C, L)
    inferred_trace = y.cpu().numpy()
    learned_gf_detached = learned_gf.detach().cpu().numpy()[:, 0]

    # check against M0
    stf_phys = stf_detached * args.c_trc / args.c_gf[0] # sum = M0 / M0_egf(ref), per posterior sample
    np.save(f"{args.PATH}/Data/reconSTF_ratio.npy", stf_phys)
    if args.M0_egf is not None:
        mrf = stf_phys * args.M0_egf * args.samp_rate  # moment-rate function, N·m/s
        M0_post = stf_phys.sum(axis=-1) * args.M0_egf                     # posterior samples of M0
        np.save(f"{args.PATH}/Data/MRF.npy", mrf)

    np.save(f"{args.PATH}/Data/reconSTF.npy", stf_detached)
    np.save(f"{args.PATH}/Data/outTRC.npy", inferred_trace)
    np.save(f"{args.PATH}/Data/outGF.npy", learned_gf_detached)

    if st_trc is not None:
        base = os.path.splitext(os.path.basename(args.trc0))[0]
        for e in range(args.num_egf):
            st_mn, st_sd = st_trc.copy(), st_trc.copy()
            for c in range(args.num_comp):
                st_mn[c].data = inferred_trace[e, :, c].mean(axis=0) * args.c_trc
                st_sd[c].data = inferred_trace[e, :, c].std(axis=0) * args.c_trc
            tag = f"_egf{e}" if args.num_egf > 1 else ""
            st_mn.write(f"{args.PATH}/{base}_out_mean{tag}.mseed")
            st_sd.write(f"{args.PATH}/{base}_out_std{tag}.mseed")
    
    if st_gf is not None:
        st_gf_out = st_gf.copy()
        lk = learned_gf_detached[:, 0].transpose(1, 0, 2).reshape(args.num_egf * args.num_comp, -1)
        for i in range(len(lk)):
            st_gf_out[i].data = lk[i] * args.c_gf[0]     
        base_gf = os.path.splitext(os.path.basename(args.egf0))[0]
        st_gf_out.write(f"{args.PATH}/{base_gf}_out.mseed")

    # Plot
    if st_gf is not None and st_trc is not None:
        plot_st(st_trc, st_gf, inferred_trace, learned_gf_detached, stf_detached, args)

    else:
        plot_res(k, k_sub, stf_detached, learned_gf_detached, inferred_trace,
                 gf0_detached, trc0_detached, args,
                 true_gf=gf_true if args.synthetics else None,
                 true_stf=stf_true if args.synthetics else None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='args')

    # Configurations
    parser.add_argument('-trc0', '--trc0', type=str, default='',
                        help='Path or name of trace file, npy array or obspy stream')
    parser.add_argument('-egf0', '--egf0', type=str, default='',
                        help='Path or name of EGF file, npy array or obspy stream')

    parser.add_argument('--btsize', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--num_epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_subepochsE', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_subepochsM', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
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
    parser.add_argument('--Mlr', type=float, default=5e-5,
                        help='learning rate(default: 1e-4)')

    parser.add_argument('--dv', type=str, default='cpu',
                        help='which GPU to use, or cpu by default')
    parser.add_argument('--multidv', type=int, nargs='+', default=None,
                        help="use multiple gpus (default: 1) use -1 for all")
    parser.add_argument('--output', action='store_true', default=False,
                        help='Plot figures, store output at each step')

    # User configurations
    parser.add_argument('-dir', '--dir', type=str, default="results",
                        help='Output directory')
    parser.add_argument('--stf0', type=str, default='',
                        help='init STF file name')
    parser.add_argument('--M0_egf', type=float, default=None,
                        help='Seismic moment of the reference EGF (N·m)')
    parser.add_argument('--num_egf', type=int, default=1, metavar='N',
                        help='number of EGF (default: 1)')
    parser.add_argument('--num_comp', type=int, default=3, metavar='N',
                        help='number of components (default: 3)')
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
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='permute parameter, if False, random, if True, reverse')
    parser.add_argument('--seqfrac', type=int, default=8,
                        help='seqfrac (default:2), should be < to stf length')

    # Weight parameters
    parser.add_argument('--data_sigma', type=float, default=5e-2,
                        help='data sigma (default: 5e-5)')
    parser.add_argument('--stf0_sigma', type=float, default=2e-1,
                        help='sigma for init STF on E step prior, default 2e-1')
    parser.add_argument('--stf_weight', type=float, nargs='+', default=None,
                        help='weight on E step priors, list (default None = function of data_sigma)')
    # parser.add_argument('--logdet_weight', type=float, default=1,
    #                     help='β, controls entropy, E step prior (default 1)')
    parser.add_argument('--prior_phi_weight', type=float, default=None, nargs='+',
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
            args.device_ids = [dv] + arr[0:dv] + arr[dv + 1:]
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
