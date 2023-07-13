#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import itertools
import scipy
from scipy import signal
import obspy

from pytorch_softdtw_cuda import soft_dtw_cuda_wojit as soft_dtw_cuda # from https://github.com/Maghoumi/pytorch-softdtw-cuda
from generative_model import realnvpfc_model



sns.set_style("white", {'axes.edgecolor': 'darkgray',
                        'axes.spines.right': False,
                        'axes.spines.top': False})
plt.style.use('myfig.mplstyle')
myblue = '#244c77ff'
mycyan = '#3f7f93ff'
myred = '#c3553aff'
myorange = '#f07101'
    
class KNetwork(torch.nn.Module):
    def __init__(self, ini, device, num_layers = 3, num_egf = 1):

        super(KNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_egf = num_egf
        self.device = device

        ## initialize kernel
        init = makeInit(ini, self.num_layers, self.device).view(self.num_layers, 1, 3*self.num_egf, ini.shape[-1])

        self.layers = torch.nn.Parameter(init, requires_grad = True)

    def load(self,filepath,device):
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
    def generatekernel(self):
        if self.num_layers >= 2:
            ker = self.layers[0]
            for i in range(1, self.num_layers):
                ker = F.conv1d(ker, self.layers[i].view(self.num_egf* 3, 1, self.layers[0].shape[-1]).flip(2),
                               padding='same', groups=3*self.num_egf)
        else:
            ker = self.layers[0]

        # out = ker / torch.max(torch.abs(ker))
        out = ker
        return out.reshape(out.shape[0], self.num_egf, 3, out.shape[-1])

    def forward(self, x):
        k = self.generatekernel()
        out = F.conv1d(k.reshape(3,1,k.shape[-1]),x, padding='same' )
        out = torch.transpose(out, 0, 1)
        out = out.reshape(x.shape[0], 3, out.shape[-1])
        return out

def trueForward(k, x, num_egf):
    out = F.conv1d(k.reshape(3*num_egf,1, k.shape[-1]), x, padding='same', groups=1)
    out = torch.transpose(out, 0, 1)
    out = out.reshape(x.shape[0], num_egf, 3, out.shape[-1])
    return out


def makeInit(init, num_layers, device, noise_amp=.5):
    """
    """
    l0 = torch.zeros(init.shape, device=device)
    l0[ :, init.shape[1]//2] = 1.

    out = torch.zeros(num_layers, init.shape[0], init.shape[1], device=device)
    for i in range(num_layers - 1):
        out[i] = l0 + (torch.randn(1, device=device)[0] * noise_amp / 100.) * torch.randn(l0.shape, device=device)
    out[-1] = init + (2 * noise_amp / 100.) * torch.randn(l0.shape, device=device)

    return out


######################################################################################################################

        
#                            EM
    

######################################################################################################################

def GForward(z_sample, img_generator, npix, npiy, logscale_factor, device=None, imginit=None, device_ids=None):
    if imginit is None:
        if device_ids is not None:
            img_samp, logdet = img_generator.module.reverse(z_sample)
        else:
            img_samp, logdet = img_generator.reverse(z_sample)
        img_samp = img_samp.reshape((-1, npiy, npix))
    else:
        ini = 0.05*torch.randn(imginit.shape) + 0.95*imginit
        img_samp = torch.repeat(ini, [len(z_sample)], axis=0)
        img_samp = img_samp.reshape((-1, npiy, npix))
        img_samp = torch.Tensor(img_samp).to(device=device)
        logdet = 0

    # apply scale factor
    logscale_factor_value = logscale_factor.forward()
    scale_factor = torch.exp(logscale_factor_value)
    # img = img_samp * scale_factor
    img = img_samp
    det_scale = logscale_factor_value * npix * npiy
    logdet = logdet + det_scale
    return img, logdet

def FForward(x, kernel_network, sigma, device):
    y = kernel_network(x)
    noise = torch.randn(y.shape)*sigma
    y += noise.to(device)
    return y


def EStep(z_sample, ytrue, img_generator, kernel_network, prior_x, prior_img, logdet_weight,
          npix, npiy, logscale_factor, args):

    device_ids = args.device_ids if len(args.device_ids) > 1 else None
    data_weight = 1 / args.data_sigma ** 2

    img, logdet = GForward(z_sample, img_generator, npix,npiy, logscale_factor, device=args.device, device_ids=device_ids)
    y = [FForward(img, kernel_network[i], args.data_sigma, args.device) for i in range(len(kernel_network))]

    ## log likelihood
    logqtheta = -logdet_weight*torch.mean(logdet)

    ## prior on trace
    meas_err = torch.stack([data_weight*nn.MSELoss()(y[i], ytrue) for i in range(len(kernel_network))])
    smoothmin_meas_err = - torch.logsumexp (-0.1 * meas_err, 0) / 0.1

    ## prior on STF
    priorx = torch.mean(prior_x(img, args.px_init_weight))

    if isinstance(prior_img, list):
        if isinstance(args.px_weight, list):
            priorimg = torch.mean( torch.Tensor( [torch.mean(prior_img[i](img, args.px_weight[i])) for i in range(len(prior_img))] ))
        else:
            priorimg = torch.mean( torch.Tensor( [torch.mean(prior_img[i](img, args.px_weight)) for i in range(len(prior_img))] ))
    else:
        priorimg = torch.mean(prior_img(img, args.px_weight))

    loss = logqtheta + priorx + smoothmin_meas_err + priorimg
    return loss, logqtheta, priorx+priorimg, smoothmin_meas_err


def MStep(z_sample, x_sample, npix, npiy, ytrue, img_generator, kernel_network, fwd_network,
          logscale_factor, prior_phi, ker_softl1, L1_prior,
          mEGF_kernel_list, mEGF_MSE_list, mEGF_y_list, args):

    device_ids = args.device_ids if len(args.device_ids) > 1 else None

    # inferred IMG
    # img, logdet = GForward(z_sample, img_generator, npix,npiy, logscale_factor, device=args.device, device_ids=device_ids)
    # # TRC from inferred IMG
    # y = FForward(img, kernel_network, args.data_sigma, args.device)
    # # TRC from random IMG
    # y_x = FForward(x_sample, kernel_network, args.data_sigma, args.device)
    # # TRC from random IMG but init GF
    # fwd = FForward(x_sample, fwd_network, args.data_sigma, args.device)

    kernel = [kernel_network[i].module.generatekernel().detach() for i in range(args.num_egf)] \
        if len(args.device_ids) > 1 else [kernel_network[i].generatekernel().detach() for i in range(args.num_egf)]

    img, logdet = GForward(z_sample, img_generator, npix, npiy, logscale_factor,
                           device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
    y = [FForward(img, kernel_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
    y_x = [FForward(x_sample, kernel_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
    fwd = FForward(x_sample, fwd_network, args.data_sigma, args.device)

    pphi = [args.phi_weight * nn.MSELoss()(y_x[i], fwd[:,i,:,:]) for i in range(args.num_egf)]

    # kernel = kernel_network.module.generatekernel() if device_ids is not None else kernel_network.generatekernel()

    ## Priors on init GF
    prior = [args.prior_phi_weight * prior_phi[0](kernel[i].squeeze(0)) for i in range(args.num_egf)]
    for i in range(args.num_egf):
        if args.num_egf == 1:
            prior[i] += prior_phi[1](args.prior_phi_weight, kernel[i].squeeze(0))[0]
        else:
            prior[i] += prior_phi[1](args.prior_phi_weight, kernel[i].squeeze(0), i)[0]

    ## Soft L1
    norm_k = [args.kernel_norm_weight * ker_softl1(kernel_network[i]) for i in range(args.num_egf)]

    meas_err = [(1e-1/args.data_sigma)* args.egf_qual_weight[i] * nn.MSELoss()(y[i], ytrue) for i in range(args.num_egf)]

    # Multi M-steps for multiple EGFs
    if args.num_egf > 1:

        # update with current kernel
        # mEGF_MSE_list[k_egf] = meas_err.detach()
        # mEGF_kernel_list[k_egf] = kernel.detach()
        # mEGF_y_list[k_egf] = y.detach()

        idx_best = torch.argmin(torch.stack(meas_err))

        # sdtw = soft_dtw_cuda.SoftDTW(use_cuda=False, gamma=1) if k_egf != idx_best else null

        # α = [torch.tanh( Loss_L2(yi, ytrue) ) / torch.sum( torch.Tensor([torch.tanh( Loss_L2(yk, ytrue) ) for yk in mEGF_y_list]) ) for yi in mEGF_y_list]
        # print(α)

        α = [Loss_L2(y[i], ytrue)  / torch.sum( torch.Tensor([ Loss_L2(y[k], ytrue)  for k in range(args.num_egf)]) ) for i in range(args.num_egf)]
        # print(α)
        # multi_loss = args.egf_multi_weight *α[k_egf][0]*Loss_L1(kernel.squeeze(0), mEGF_kernel_list[idx_best].squeeze(0))
        sdtw = soft_dtw_cuda.SoftDTW(use_cuda=False, gamma=1) if k_egf != idx_best else null
        multi_loss = args.egf_multi_weight * torch.sum( torch.Tensor([ α[i]*( Loss_L2(kernel[i].squeeze(0), kernel[idx_best].squeeze(0))
                                                                              + 0.35*torch.abs(sdtw(kernel[i].squeeze(0), kernel[idx_best].squeeze(0))[0]) ) for i in range(args.num_egf) ]) )
        # multi_loss =15*args.egf_multi_weight * torch.Tensor([ α[i]*Loss_L2(kernel[i].squeeze(0), kernel[idx_best].squeeze(0)) for i in range(args.num_egf) ])
        # multi_loss = torch.tensor(0.)

        # if k_egf == idx_best:
        #     # multi_loss = 1e-2 * args.egf_multi_weight * L1_prior(kernel.squeeze(0))
        #     multi_loss = 1e-2 * args.egf_multi_weight * L1_prior(kernel.squeeze(0), idx_best) + \
        #                  1e-2 * args.egf_multi_weight * L1_prior(kernel.squeeze(0), last_idx)
        #         #          + args.egf_multi_weight * 1e-3 * torch.sum(torch.Tensor(
        #         # [Loss_L2(kernel.squeeze(0), e.squeeze(0)) for i, e in enumerate(mEGF_kernel_list) if i != idx_best]))
        #     print('{} best: {}'.format(k_egf, multi_loss))
        # else:
        #     sdtw = soft_dtw_cuda.SoftDTW(use_cuda=False, gamma=1) if k_egf != idx_best else null
        #     multi_loss = 1e2*args.egf_multi_weight * (Loss_L2(kernel.squeeze(0), mEGF_kernel_list[idx_best].squeeze(0)) + \
        #                                               Loss_L2(kernel.squeeze(0), mEGF_kernel_list[last_idx].squeeze(0)) + \
        #                                             0.35*torch.abs(sdtw(kernel.squeeze(0), mEGF_kernel_list[idx_best].squeeze(0))[0] ))
        #                  # + args.egf_multi_weight*1e-2* torch.sum( torch.Tensor([Loss_L2(kernel.squeeze(0),e.squeeze(0)) for i, e in enumerate(mEGF_kernel_list) if i != idx_best]) )
    else:
        multi_loss = torch.tensor(0.)

    loss = {}
    for i in range(args.num_egf):
        loss[i] = torch.Tensor(meas_err[i] + norm_k[i] + prior[i] + pphi[i] + multi_loss)

    return loss, meas_err, norm_k, prior, multi_loss


######################################################################################################################

        
#                            DPI
    

######################################################################################################################



class img_logscale(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, device, scale=1):
        super().__init__()
        log_scale = torch.Tensor(torch.log(scale)*torch.ones(1, device=device))
        self.log_scale = nn.Parameter(log_scale)

    def forward(self):
        return self.log_scale

class stf_generator(nn.Module):
    '''Softplus and norm for realnvp for STF'''
    def __init__(self, realnvp, softplus=True):
        super().__init__()
        self.realnvp = realnvp
        self.softplus = softplus

    def forward(self, input):
        return self.realnvp.forward(input)

    def reverse(self,input):
        img, logdet = self.realnvp.reverse(input)
        if self.softplus:
            out = torch.nn.Sigmoid()(img)
            det_sigmoid = torch.sum(-img - 2 * torch.nn.Softplus()(-img), -1)
            logdet = logdet + det_sigmoid
        else:
            out = img
        return out, logdet


######################################################################################################################


#                            Losses


######################################################################################################################

def dtw_classic(x, y, dist='square'):
    """Classic Dynamic Time Warping (DTW) distance between two time series.

    References
    ----------
    .. [1] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
           for spoken word recognition". IEEE Transactions on Acoustics,
           Speech, and Signal Processing, 26(1), 43-49 (1978).

    Modified from:
    Author: Johann Faouzi <johann.faouzi@gmail.com>
    License: BSD-3-Clause
    Pyts, A Python Package for Time Series Classification
    """
    def _square(x, y):
        return torch.square(x - y)

    def _absolute(x, y):
        return torch.abs(x - y)

    def _accumulated_cost_matrix(cost_matrix):
        n_timestamps_1, n_timestamps_2 = cost_matrix.shape
        acc_cost_mat = torch.empty((n_timestamps_1, n_timestamps_2))
        acc_cost_mat[0] = cost_matrix[0].cumsum(dim=0)
        acc_cost_mat[:, 0] = cost_matrix[:, 0].cumsum(dim=0)
        for j in range(1, n_timestamps_2):
            for i in range(1, n_timestamps_1):
                acc_cost_mat[i, j] = cost_matrix[i, j] + min(
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1]
                )
        return acc_cost_mat

    if dist == 'square':
        dist_ = _square
    elif dist == 'absolute':
        dist_ = _absolute

    if x.dim() > 1:
        x_mean = torch.mean(x, axis=(0,1))
        n_timestamps_1, n_timestamps_2 = x.shape[-1], y.shape[-1]
        cost_mat = torch.empty((n_timestamps_1, n_timestamps_2))
        for j in range(n_timestamps_2):
            for i in range(n_timestamps_1):
                cost_mat[i, j] = dist_(x_mean[i], y[j])
    else:
        n_timestamps_1, n_timestamps_2 = x.shape[-1], y.shape[-1]
        cost_mat = torch.empty((n_timestamps_1, n_timestamps_2))
        for j in range(n_timestamps_2):
            for i in range(n_timestamps_1):
                cost_mat[i, j] = dist_(x[i], y[j])

    acc_cost_mat = _accumulated_cost_matrix(cost_mat)

    dtw_dist = acc_cost_mat[-1, -1]
    if dist == 'square':
        dtw_dist = torch.sqrt(dtw_dist)

    return dtw_dist

def priorPhi(k, k0):
    ker = k - k0
    out = ker
    return torch.mean(torch.abs(out))

def Loss_L2(z, z0):
    return torch.sqrt(torch.sum( (z - z0)**2 ))

def Loss_L1(z, z0):
    return torch.sum( torch.abs(z - z0) )

def Loss_TV(z):
    return torch.mean( torch.abs(z[:,:, 1::] - z[:,:, 0:-1]))

def Loss_DTW(z, z0):
    # Dynamic Time Warping loss with initial STF
    # not using fastDTW because does not allow different sizes for z and z0
    return dtw_classic(z, z0)

def Loss_DTW_Mstep(z, z0):
    # uses fast DTW, similar to L2 if aligned
    sdtw = soft_dtw_cuda.SoftDTW(use_cuda= False, gamma=0.1)
    return sdtw(z, z0)

def Loss_multicorr(z, args):

    n = len(z)
    comb = torch.combinations(torch.arange(0, n), 2)

    sdtw = soft_dtw_cuda.SoftDTW(use_cuda= False, gamma=0.1)
    coef = sdtw(z[comb[:,0], :, :], z[comb[:,1], :, :])

    # nbr_combi = np.math.factorial(n) / 2 / np.math.factorial(n-2)
    # coef = torch.zeros(( int(nbr_combi) ,3))

    # for i,co in enumerate(itertools.combinations(range(n), 2)):
        # for k in range(3):
            # Pearson coeff
            # coef[i,k] = 1 - torch.corrcoef(z[co, k, :])[0,1]

            # TV
            # coef[i,k] = torch.mean(torch.abs(z[co[0], k, :] - z[co[1], k, :]) )

            # DTW
            # coef[i,k] = dtw_classic(z[co[0], k, :], z[co[1], k, :])
            # coef[i,k] = Loss_DTW(z[co[0], k, :],
            #                      z[co[1], k, :],
            #                      args)
    return torch.mean(coef) / 10**(torch.floor(torch.log10(torch.mean(coef))))

def null(x,y):
    return [0.]

######################################################################################################################

        
#                            PLOT
    

######################################################################################################################

def plot_res(k, k_sub, image, learned_k, learned_trc, stf0, gf, trc, args, true_stf=None, true_gf=None):
    mean_img = np.mean(image, axis=0)
    stdev_img = np.std(image, axis=0)
    gf_np = gf.detach().cpu().numpy()
    stf0 = stf0.detach().cpu().numpy()
    trc = trc.detach().cpu().numpy()
    mean_trc = [np.mean(learned_trc[i], axis=0) for i in range(len(learned_trc))]
    stdev_trc = [np.std(learned_trc[i], axis=0) for i in range(len(learned_trc))]

    for e in range(args.num_egf):
        learned_kernel = learned_k[e][0]
        gf = gf_np[e]
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot2grid((12,4), (0, 0), colspan=3, rowspan=2)
        ax2 = plt.subplot2grid((12,4), (2, 0), colspan=3, rowspan=2)
        ax3 = plt.subplot2grid((12,4), (4, 0), colspan=3, rowspan=2)
        ax4 = plt.subplot2grid((12,4), (3, 3), colspan=1, rowspan=4)
        ax5 = plt.subplot2grid((12,4), (6, 0), colspan=3, rowspan=2)
        ax6 = plt.subplot2grid((12,4), (8, 0), colspan=3, rowspan=2)
        ax7 = plt.subplot2grid((12,4), (10, 0), colspan=3, rowspan=2)
        x = np.arange(0, gf.shape[1])

        if true_gf is not None:
            true_gf = signal.resample(true_gf, gf.shape[-1],axis=1)
            ax1.plot(x, true_gf[0], lw=0.5, color=myred, label='Target')
        ax1.plot(x, gf[0], lw=0.5, color=myblue, label='Prior')
        ax1.plot(x, learned_kernel[0], lw=0.5, color=myorange, zorder=2, label='Inferred')
        ax1.fill_between(x, learned_kernel[0], learned_kernel[0],
                         facecolor=myorange, alpha=0.35, zorder=0, label='2σ')
        ax1.text(0.03, 0.9, 'E',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax1.transAxes)

        ax1.legend(loc=(1.05, 0), frameon=False)
        if true_gf is not None:
            ax2.plot(x, true_gf[1], lw=0.5, color=myred)
        ax2.plot(x, gf[1], lw=0.5, color=myblue)
        ax2.plot(x, learned_kernel[1], lw=0.5, color=myorange, zorder=2)
        ax2.text(0.03, 0.9, 'N',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax2.transAxes)
        if true_gf is not None:
            ax3.plot(x , true_gf[2], lw=0.5, color=myred)
        ax3.plot(x, gf[2], lw=0.5, color=myblue)
        ax3.plot(x, learned_kernel[2], lw=0.5, color=myorange, zorder=2)
        ax3.text(0.03, 0.9, 'Z',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax3.transAxes)
        ax1.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)

        # trace
        learned_trace = mean_trc[e]

        ax5.plot(trc[0], lw=0.5, color=myred)
        ax5.plot(learned_trace[0], lw=0.5, color=myorange, zorder=2)
        ax5.text(0.03, 0.9, 'E',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax5.transAxes)
        ax6.plot(trc[1], lw=0.5, color=myred)
        ax6.plot(learned_trace[1], lw=0.5, color=myorange, zorder=2)
        ax6.text(0.03, 0.9, 'N',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax6.transAxes)
        ax7.plot(trc[2], lw=0.5, color=myred)
        ax7.plot(learned_trace[2], lw=0.5, color=myorange, zorder=2)
        ax7.text(0.03, 0.9, 'Z',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax7.transAxes)
        ax5.get_xaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)

        # STF
        xinf = np.linspace(0, mean_img.shape[1], mean_img.shape[1])
        if true_stf is not None:
            if len(true_stf) < mean_img[0].shape[0]:
                true_stf_rs = np.zeros(mean_img[0].shape)
                true_stf_rs[:len(true_stf)] = true_stf
                ax4.plot(xinf, true_stf_rs, lw=0.8, color=myred)
            else:
                ax4.plot(xinf, true_stf[:mean_img[0].shape[0]], lw=0.8, color=myred)
        ax4.plot(xinf, mean_img[0], lw=1, color=myorange)
        ax4.fill_between(xinf, mean_img[0] - 2*stdev_img[0], mean_img[0] + 2*stdev_img[0],
                         facecolor=myorange, alpha=0.35, zorder=0, label='2σ')


        fig.savefig("{}/out_egf{}_{}_{}.png".format(args.PATH, str(e), str(k).zfill(5), str(k_sub).zfill(5)), format='png', dpi=300,
                bbox_inches="tight")
        plt.close()
    return


def plot_st(st_trc, st_gf, inferred_trace, learned_kernel, image, args):
    mean_img = np.mean(image, axis=0)
    stdev_img = np.std(image, axis=0)
    mean_trc = np.mean(inferred_trace, axis=0)
    stdev_trc = np.std(inferred_trace, axis=0)
    gf = np.concatenate([st_gf[k].data[:, None] for k in range(len(st_gf))], axis=1).T
    gf = gf.reshape(gf.shape[0] // 3, 3, gf.shape[1], order='F')
    trc = np.concatenate([st_trc[k].data[:, None] for k in range(len(st_trc))], axis=1).T
    gf = gf/np.amax(np.abs(gf))
    trc = trc/np.amax(np.abs(trc))

    if args.num_egf == 1:
        rap = [np.amax(st_trc[i].data) / np.amax(st_gf[i].data) for i in range(3)]

        fig = plt.figure(figsize=(6, 1.2))
        plt.subplots_adjust(wspace=0.15)
        ax = plt.subplot(1, 3, 1)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)
        chan = ['E', 'N', 'Z']
        for i in range(3):
            tmax = np.amax(st_trc[0].times())
            ax.fill_between(st_trc[0].times() - (2 - i) * tmax // 5, mean_trc[0,i] - stdev_trc[0,i] + (2 - i) * 0.6,
                            mean_trc[0,i] + stdev_trc[0,i] + (2 - i) * 0.6,
                            facecolor=myorange, alpha=0.25, zorder=0, label='Standard deviation', clip_on=False)
            l1 = ax.plot(st_trc[0].times() - (2 - i) * tmax // 5, trc[i] + (2 - i) * 0.6, color='k', lw=0.8,
                         clip_on=False)
            l2 = ax.plot(st_trc[0].times() - (2 - i) * tmax // 5, mean_trc[0,i] + (2 - i) * 0.6, lw=0.7, color=myorange,
                         clip_on=False)
            ax.text(np.amin(st_trc[0].times() - (2 - i) * tmax // 5) - 5, np.mean(trc[i] + (2 - i) * 0.6), chan[i],
                    horizontalalignment='right',
                    verticalalignment='top', weight='bold')
            ax.text(np.amin(st_trc[0].times() - (2 - i) * tmax // 5), (2 - i) * 0.6 + 0.25, 'x ' + str(int(rap[i])),
                    horizontalalignment='left', verticalalignment='top', fontsize='small')
        plt.xlim(np.amin(st_trc[0].times() - (2) * tmax // 5) + tmax // 5, tmax - tmax // 7)
        plt.xlabel('Time (s)', labelpad=2, loc='left')
        ticklab = ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        ax.xaxis.set_label_coords(-2 * tmax // 5, 0, transform=trans)

        ax = plt.subplot(1, 3, 2)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)
        for i in range(3):
            tmax = np.amax(st_gf[0].times())
            ax.plot(st_gf[0].times() - (2 - i) * tmax // 5, gf[0,i] + (2 - i) * 0.6, color='k', lw=0.8, clip_on=False)
            ax.plot(st_gf[0].times() - (2 - i) * tmax // 5, learned_kernel[0,0,i] + (2 - i) * 0.6, lw=0.7, color=myorange,
                    clip_on=False)
        plt.xlim(np.amin(st_gf[0].times() - (2) * tmax // 5) + tmax // 5, tmax - tmax // 7)
        ax.text(1, .8, 'data', horizontalalignment='right', verticalalignment='top', color='k', transform=ax.transAxes)
        ax.text(1, 0.9, 'predictions', horizontalalignment='right', verticalalignment='top', color=myorange,
                transform=ax.transAxes)

        ax = plt.subplot(1, 3, 3)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)
        ax.fill_between(np.arange(len(mean_img[0])) / st_gf[0].stats.sampling_rate, mean_img[0] - stdev_img[0],
                        mean_img[0] + stdev_img[0],
                        facecolor=myorange, alpha=0.25, zorder=0, label='Standard deviation')
        ax.plot(np.arange(len(mean_img[0])) / st_gf[0].stats.sampling_rate, mean_img[0], lw=1, color=myorange)

    else:
        fig = plt.figure(figsize=(4,(args.num_egf+1)*1.2))
        plt.subplots_adjust(wspace=0.15)

        ax = plt.subplot(args.num_egf+1, 2, 1)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=True)
        ax.fill_between(np.arange(len(mean_img[0]))/st_gf[0].stats.sampling_rate, mean_img[0] - stdev_img[0], mean_img[0] + stdev_img[0],
                         facecolor=myorange, alpha=0.25, zorder=0, label='Standard deviation')
        ax.plot(np.arange(len(mean_img[0]))/st_gf[0].stats.sampling_rate, mean_img[0], lw=1, color=myorange)
        plt.xlabel('Time (s)', labelpad=2, loc='left')
        tmax=np.amax(np.arange(len(mean_img[0]))/st_gf[0].stats.sampling_rate)
        ticklab = ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        ax.xaxis.set_label_coords(-.35*tmax, 0, transform=trans)
        ax.text(1.1, .75, 'data', horizontalalignment='left', verticalalignment='top', color='k', transform=ax.transAxes)
        ax.text(1.1, 0.9, 'predictions', horizontalalignment='left', verticalalignment='top', color=myorange,
                transform=ax.transAxes)

        for k in range(args.num_egf):
            rap = [np.amax(st_trc[i].data) / np.amax(st_gf[k+args.num_egf*i].data) for i in range(3)]
            ax = plt.subplot(args.num_egf+1, 2, k*2+3)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=True)
            chan = ['E', 'N', 'Z']
            for i in range(3):
                tmax = np.amax(st_trc[0].times())
                ax.fill_between(st_trc[0].times()-(2-i)*tmax//5, mean_trc[k,i] - stdev_trc[k,i]+(2-i)*0.6, mean_trc[k,i] + stdev_trc[k,i]+(2-i)*0.6,
                                 facecolor=myorange, alpha=0.25, zorder=0, label='Standard deviation',clip_on=False)
                l1 = ax.plot(st_trc[0].times()-(2-i)*tmax//5, trc[i]+(2-i)*0.6, color='k', lw=0.8,clip_on=False)
                l2 = ax.plot(st_trc[0].times()-(2-i)*tmax//5, mean_trc[k,i]+(2-i)*0.6, lw=0.7, color=myorange,clip_on=False)
                ax.text(np.amin(st_trc[0].times()-(2-i)*tmax//5)-tmax//7, np.mean(trc[i]+(2-i)*0.6), chan[i],
                         horizontalalignment='right',
                         verticalalignment='top',weight='bold')
                ax.text(np.amin(st_trc[0].times() - (2 - i) * tmax // 5) , (2 - i) * 0.6+0.3, 'x '+str(int(rap[i])),
                        horizontalalignment='left', verticalalignment='top', fontsize='small')
            plt.xlim(np.amin(st_trc[0].times()-(2)*tmax//5)+10, tmax-5)
            plt.xlabel('Time (s)', labelpad=2, loc='left')
            ticklab = ax.xaxis.get_ticklabels()[0]
            trans = ticklab.get_transform()
            ax.xaxis.set_label_coords(-2*tmax//5, 0, transform=trans)

            ax = plt.subplot(args.num_egf+1, 2, k*2+4)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=True)
            for i in range(3):
                tmax = np.amax(st_gf[0].times())
                ax.plot(st_gf[0].times()-(2-i)*tmax//5, gf[k,i]+(2-i)*0.6, color='k', lw=0.8,clip_on=False)
                ax.plot(st_gf[0].times()-(2-i)*tmax//5, learned_kernel[k,0,i]+(2-i)*0.6, lw=0.7, color=myorange,clip_on=False)
            plt.xlim(np.amin(st_gf[0].times()-(2)*tmax//5)+10, tmax-5)

    fig.savefig("{}/out_{}.pdf".format(args.PATH, 'res'),
                bbox_inches="tight")
    plt.close()
    return


def plot_trace(trc, inferred_trace, args):

    mean_blur_img = np.mean(inferred_trace, axis=(0,1))
    stdev_blur_img = np.std(inferred_trace, axis=(0,1))
    std_max = stdev_blur_img.max()
    truetrc = trc.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((6, 4), (2, 0), colspan=3, rowspan=2)
    ax3 = plt.subplot2grid((6, 4), (4, 0), colspan=3, rowspan=2)
    ax1.plot(truetrc[0], lw=0.5, color=myblue)
    ax1.plot(mean_blur_img[0], lw=0.5, color=myorange, zorder=2)
    ax1.fill_between(np.arange(len(mean_blur_img[0])), mean_blur_img[0] - stdev_blur_img[0],
                     mean_blur_img[0] + stdev_blur_img[0],
                     facecolor=myorange, alpha=0.25, zorder=0)
    ax1.text(0.03, 0.9, 'E',
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes)
    ax2.plot(truetrc[1], lw=0.5, color=myblue)
    ax2.plot(mean_blur_img[1], lw=0.5, color=myorange, zorder=2)
    ax2.fill_between(np.arange(len(mean_blur_img[1])), mean_blur_img[1] - stdev_blur_img[1],
                     mean_blur_img[1] + stdev_blur_img[1],
                     facecolor=myorange, alpha=0.25, zorder=0)
    ax2.text(0.03, 0.9, 'N',
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax2.transAxes)
    ax3.plot(truetrc[2], lw=0.5, color=myblue)
    ax3.plot(mean_blur_img[2], lw=0.5, color=myorange, zorder=2)
    ax3.fill_between(np.arange(len(mean_blur_img[2])), mean_blur_img[2] - stdev_blur_img[2],
                     mean_blur_img[2] + stdev_blur_img[2],
                     facecolor=myorange, alpha=0.25, zorder=0)
    ax3.text(0.03, 0.9, 'Z',
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax3.transAxes)
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("{}/outTRC.pdf".format(args.PATH), format='pdf',
                bbox_inches="tight")
    plt.close()
    return
