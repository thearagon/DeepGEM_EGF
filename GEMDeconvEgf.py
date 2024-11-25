#!/usr/bin/env python
# coding: utf-8

from deconvEgf_helpers import *


def main_function(args):
    ################################################ SET UP WEIGHTS ################################################

    args.phi_weight = 1e-1
    # weight on init STF
    args.stf0_weight = args.stf0_weight or (1 / args.data_sigma) / 1e4
    # weights for priors on Estep: list, [boundaries, TV, L1]
    args.stf_weight = args.stf_weight or [(1 / args.data_sigma) / 2e3,
                                          (1 / args.data_sigma) / 4e2,
                                          (1 / args.data_sigma) / 2e3]
    # weight on q_theta
    args.logdet_weight = args.logdet_weight or (1 / args.data_sigma) / 1e2
    # weights for priors on Mstep: list, [L1, L2, TV]
    args.prior_phi_weight = args.prior_phi_weight or [(1 / args.data_sigma) / 6e3,
                                                      (1 / args.data_sigma) / 4e4, 0]
    # if multiple EGFs
    if args.num_egf > 1:
        args.egf_multi_weight = args.egf_multi_weight or (1 / args.data_sigma) / 4e4
        args.egf_qual_weight = args.egf_qual_weight or np.ones(args.num_egf).tolist()
    else:
        args.egf_multi_weight = 0.0
        args.egf_qual_weight = [1.0]

    ################################################ SET UP DATA ################################################

    def load_data(file_path, reshape_dims=None):
        try:
            data = obspy.read(file_path)
            array = np.concatenate([trace.data[:, None] for trace in data], axis=1).T
            if reshape_dims:
                array = array.reshape(reshape_dims, order='F')
            return data, array, 1 / data[0].stats['delta']
        except TypeError:
            array = np.load(file_path)
            if reshape_dims:
                array = array.reshape(reshape_dims, order='F')
            return None, array, None

    # traces
    st_trc, trc0, _ = load_data(args.trc0)
    trc0 = torch.Tensor(trc0 / np.amax(np.abs(trc0))).to(args.device)

    # EGF
    st_gf, gf0, args.samp_rate = load_data(args.egf0, reshape_dims=(args.num_egf, 3, -1))
    gf0 = torch.Tensor(gf0 / np.amax(np.abs(gf0))).to(args.device)

    # Determine STF length
    len_stf = int(args.stf_dur * args.samp_rate) if args.stf_dur and args.samp_rate else args.stf_size
    args.stf_size = len_stf

    # STF
    if args.stf0:
        _, stf0, _ = load_data(args.stf0)
        stf0 = np.resize(stf0, len_stf) if len(stf0) != len_stf else stf0
    else:
        τc = len_stf // 10
        stf0 = 0.01 + 0.99 * np.exp(-(np.arange(len_stf) - len_stf // 2) ** 2 / (2 * (τc / 2) ** 2))
    stf0 = torch.Tensor(stf0 / np.amax(stf0)).to(args.device)

    trc_ext = trc0.clone()
    gf0_detached = gf0.detach().cpu().numpy()
    trc0_detached = trc0.detach().cpu().numpy()

    # Synthetics
    if args.synthetics == True:
        st_gf_true, gf_true, _ = load_data(args.gf_true, reshape_dims=(3, gf0.shape[-1]))
        gf_true = torch.Tensor(gf_true / np.amax(np.abs(gf_true))).to(args.device)
        stf_true = np.load("{}".format(args.stf_true))
        stf_true = stf_true / np.amax(stf_true)
        if len_stf > len(stf_true):
            stf_rs = np.zeros(len_stf)
            stf_rs[(len(stf_rs) - len(stf_true)) // 2:-(len(stf_rs) - len(stf_true)) // 2] = stf_true
            stf_true = stf_rs
        elif len_stf < len(stf_true):
            len_stf = len(stf_true)

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
    permute = 'reverse' if args.reverse else 'random'
    print(f"Generating {'Reverse' if args.reverse else 'Random'} RealNVP Network")
    realnvp = realnvpfc_model.RealNVP(len_stf, n_flow, seqfrac=seqfrac, affine=affine, permute=permute).to(args.device)
    stf_gen = stf_generator(realnvp).to(args.device)

    # True forward model (with init GF), used for priors
    FTrue = lambda x: trueForward(torch.unsqueeze(gf0, dim=0), x, args.num_egf)

    print("Models Initialized")

    # Multiple GPUs
    if len(args.device_ids) > 1:
        stf_gen = nn.DataParallel(stf_gen, device_ids=args.device_ids)
        gf_network = [nn.DataParallel(network, device_ids=args.device_ids).to(args.device) for network in gf_network]

    # Priors on Mstep (EGF)
    f_phi_prior = lambda gf: priorPhi(gf, gf0)
    prior_L2 = lambda gf, weight, i: (
        weight * (0.5 * Loss_DTW_Mstep(gf, gf0 if args.num_egf == 1 else gf0[i].unsqueeze(0)) +
                  Loss_L2(gf, gf0 if args.num_egf == 1 else gf0[i].unsqueeze(0)))
        if weight > 0 else 0)
    prior_TV = lambda gf, weight, i: weight * Loss_TV(gf)
    phi_priors = [f_phi_prior, prior_L2, prior_TV]  # norms on init GF

    # Priors on Estep (STF)
    prior_stf = [
        lambda stf, weight: weight * torch.sum(torch.abs(stf[:, :, 0]) * torch.abs(stf[:, :, -1])), # Boundary
        lambda stf, weight: weight * Loss_TV(stf), # TV
        lambda stf, weight: torch.abs(1 - torch.sum(stf))] # soft L1

    # Gaussian prior for STF
    stf0_gauss = stf0 + torch.randn_like(stf0) * args.stf0_sigma
    stf0_gauss_ext = stf0_gauss.unsqueeze(0).expand(args.btsize, -1, -1).contiguous()
    prior_x = lambda stf: Loss_L2(stf, stf0_gauss_ext) / args.stf0_sigma ** 2

    flux = torch.abs(torch.sum(stf0))
    logscale_factor = stf_logscale(scale=flux / (0.8 * stf0.shape[0]), device=args.device).to(args.device)

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
                    np.save(f"{args.PATH}/Data/stf.npy", learned_gf_detached)

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
                Mloss_multi_list[k_egf].append(multiloss.item())
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
    y = torch.stack([
        FForward(stf, gf_network[i], args.data_sigma, args.device)
        for i in range(args.num_egf)
    ], dim=0)
    inferred_trace = y.detach().cpu().numpy()
    learned_gf_detached = learned_gf.detach().cpu().numpy()[:, 0]

    # Scale stf area with M0
    if args.samp_rate is not None and args.M0 is not None:
        area = np.trapz(y=stf_detached, dx=0.1, axis=-1)[..., np.newaxis]
        stf_detached /= np.repeat(area, stf_detached.shape[-1], axis=-1)
        stf_detached *= args.M0

    np.save(f"{args.PATH}/Data/reconSTF.npy", stf_detached)
    np.save(f"{args.PATH}/Data/outTRC.npy", inferred_trace)
    np.save(f"{args.PATH}/Data/outGF.npy", learned_gf_detached)

    if st_trc is not None:
        st_trc_mn, st_trc_sd = st_trc.copy(), st_trc.copy()
        inferred_trace_mean = np.mean(inferred_trace, axis=(0, 1))
        inferred_trace_std = np.std(inferred_trace, axis=(0, 1))
        for i in range(3):
            st_trc_mn[i].data = inferred_trace_mean[i]
            st_trc_sd[i].data = inferred_trace_std[i]
        st_trc_mn.write(f"{args.PATH}/{args.trc0.rsplit('/', 1)[1].rsplit('.', 1)[0]}_out_mean.mseed")
        st_trc_sd.write(f"{args.PATH}/{args.trc0.rsplit('/', 1)[1].rsplit('.', 1)[0]}_out_std.mseed")

    if st_gf is not None:
        st_gf_out = st_gf.copy()
        lk = learned_gf_detached.reshape(args.num_egf * 3, -1)
        for i in range(len(lk)):
            st_gf_out[i].data = lk[i, :]
        st_gf_out.write(f"{args.PATH}/{args.egf0.rsplit('/', 1)[1].rsplit('.', 1)[0]}_out.mseed")

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
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
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
    parser.add_argument('--Mlr', type=float, default=1e-5,
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
    parser.add_argument('--stf0_sigma', type=float, default=2e-1,
                        help='sigma for init STF on E step prior, default 2e-1')
    parser.add_argument('--stf0_weight', type=float, default=None,
                        help='weight for init STF on E step prior (default None = function of data_sigma)')
    parser.add_argument('--stf_weight', type=float, nargs='+', default=None,
                        help='weight on E step priors, list (default None = function of data_sigma)')
    parser.add_argument('--logdet_weight', type=float, default=None,
                        help='β, controls entropy, E step prior (default None = function of data_sigma)')
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
