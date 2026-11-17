#!/usr/bin/env python
# coding: utf-8

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_warn_always(False)
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
GRAY = '#555555'
mpl.rcParams.update({'text.color': GRAY, 'axes.labelcolor': GRAY, 'axes.edgecolor': GRAY,
                     'xtick.color': GRAY, 'ytick.color': GRAY, 'font.size': 7})
import random
import json
import itertools
from scipy import signal
import obspy
from pytorch_softdtw_cuda import soft_dtw_cuda_wojit as soft_dtw_cuda # from https://github.com/Maghoumi/pytorch-softdtw-cuda
from generative_model import realnvpfc_model
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

def conv_same(gf, stf):
    """True convolution with np.convolve 'same' alignment. gf: (C,1,L), stf: (B,1,K) -> (C,B,L)."""
    K = stf.shape[-1]
    return F.conv1d(F.pad(gf, (K // 2, (K - 1) // 2)), stf.flip(-1))

def gauss_nll(y, d, sigma):
    """Gaussian negative log-likelihood per sample (constants dropped).
    y, d : (B, C, L) predicted and observed traces, in normalized units
    sigma: float or list/tensor of length C (noise std per component, same units)
    returns (B,)"""
    sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
    if sigma.dim() == 1:
        sigma = sigma[:, None]                      # broadcast over time
    return 0.5 * torch.sum(((y - d) / sigma) ** 2, dim=(-2, -1))

class GFNetwork(torch.nn.Module):

    def __init__(self, ini, device, num_layers=3, num_egf=1, num_comp=3, log_amp=0.0, learn_amp=False):

        super(GFNetwork, self).__init__()
        self.num_layers = num_layers
        self.num_egf = num_egf
        self.device = device
        self.num_comp = num_comp
        self.log_amp = nn.Parameter(torch.tensor(float(log_amp), device=device), requires_grad=learn_amp)

        ## initialize GF
        init = makeInit(ini, self.num_layers, self.device).view(self.num_layers, 1, self.num_comp*self.num_egf, ini.shape[-1])

        self.layers = torch.nn.Parameter(init, requires_grad=True)
        self.register_buffer("gf_ref", ini.detach().clone().reshape(1, -1, ini.shape[-1]))   # prior EGF (1, C, L)

    def load(self, filepath, device):
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
    def generategf(self):
        if self.num_layers >= 2:
            gf = self.layers[0]
            for i in range(1, self.num_layers):
                gf = F.conv1d(gf, self.layers[i].view(self.num_egf* self.num_comp, 1, self.layers[0].shape[-1]).flip(2),
                               padding='same', groups=self.num_comp*self.num_egf)
        else:
            gf = self.layers[0]

        a = torch.sum(gf * self.gf_ref) / torch.sum(self.gf_ref ** 2)     # amount of prior EGF in gf
        out = gf / torch.clamp(a, min=1e-3)
        return out.reshape(out.shape[0], self.num_egf, self.num_comp, out.shape[-1])

    def forward(self, x, with_amp=True):
        k = self.generategf()
        out = conv_same(k.reshape(k.shape[2], 1, -1), x).transpose(0, 1) 
        return out * torch.exp(self.log_amp) if with_amp else out

def trueForward(k, x, num_egf, num_comp):
    out = conv_same(k.reshape(num_comp * num_egf, 1, k.shape[-1]), x).transpose(0, 1)
    return out.reshape(x.shape[0], num_egf, num_comp, out.shape[-1]) 


def makeInit(init, num_layers, device, noise_amp=.1):
    l0 = torch.zeros(init.shape, device=device)
    l0[:, init.shape[1]//2] = 1.

    out = torch.zeros(num_layers, init.shape[0], init.shape[1], device=device)
    for i in range(num_layers - 1):
        out[i] = l0 + (torch.randn(1, device=device)[0] * noise_amp / 100.) * torch.randn(l0.shape, device=device)
    out[-1] = init + (2 * noise_amp / 100.) * torch.randn(l0.shape, device=device)
    return out / torch.amax(torch.abs(out))


######################################################################################################################
#       
#                            EM
#
######################################################################################################################


def GForward(z_sample, stf_generator, len_stf, logscale_factor, device=None, stfinit=None, device_ids=None):
    if stfinit is None:
        if device_ids is not None:
            stf_samp, logdet = stf_generator.module.reverse(z_sample)
        else:
            stf_samp, logdet = stf_generator.reverse(z_sample)
        stf_samp = stf_samp.reshape((-1, 1, len_stf))
    else:
        ini = 0.05 * torch.randn_like(stfinit) + 0.95 * stfinit
        stf_samp = ini.repeat(len(z_sample), 1, 1).reshape((-1, 1, len_stf))
        stf_samp = stf_samp.to(device)
        logdet = torch.tensor(0.0, device=device)

    # apply scale factor
    logscale_factor_value = logscale_factor.forward()
    stf = stf_samp * torch.exp(logscale_factor_value)
    logdet = logdet + logscale_factor_value * len_stf
    
    return stf, logdet

def FForward(x, gf_network, sigma, device, with_amp=True):
    y = gf_network(x, with_amp=with_amp)
    noise = torch.randn(y.shape)*sigma
    return y + noise.to(device)


def EStep(z_sample, ytrue, stf_generator, gf_network, prior_x, prior_stf,
          len_stf, logscale_factor, args):
    device_ids = args.device_ids if len(args.device_ids) > 1 else None
    stf, logdet = GForward(z_sample, stf_generator, len_stf, logscale_factor,
                           device=args.device, device_ids=device_ids)

    # Noise-free predictions: the likelihood already accounts for the noise
    y = [gf_network[i](stf) for i in range(len(gf_network))]

    # Gaussian NLL per sample and per EGF. egf_qual_weight w_i acts as sigma_i = sigma / sqrt(w_i)
    nll = torch.stack([args.egf_qual_weight[i] * gauss_nll(y[i], ytrue, args.sigma)
                       for i in range(len(gf_network))])                      # (E, B)
    # Several EGFs: equal-weight mixture, -log sum_i p(d | x, EGF_i)  (refined in point b)
    nll = -torch.logsumexp(-nll, dim=0) if len(gf_network) > 1 else nll[0]   # (B,)
    data_term = torch.mean(nll)

    logqtheta = -args.logdet_weight * torch.mean(logdet)
    priorx = torch.mean(prior_x(stf))
    w = args.stf_weight if isinstance(args.stf_weight, list) else [args.stf_weight] * len(prior_stf)
    priorstf = torch.stack([p(stf, wi) for p, wi in zip(prior_stf, w)]).mean()

    loss = data_term + logqtheta + priorx + priorstf
    return loss, logqtheta, priorx + priorstf, data_term

def MStep(z_sample, x_sample, len_stf, ytrue, stf_generator, gf_network, fwd_network,
          logscale_factor, prior_phi, args):

    # sample STF
    stf, logdet = GForward(z_sample, stf_generator, len_stf, logscale_factor,
                           device=args.device, device_ids=args.device_ids if len(args.device_ids) > 1 else None)
    # trace output from sampled STF
    y = [FForward(stf, gf_network[i], args.data_sigma, args.device) for i in range(args.num_egf)]
    # trace from random STF with current EGFs
    y_x = [FForward(x_sample, gf_network[i], args.data_sigma, args.device, with_amp=False) for i in range(args.num_egf)]
    # trace from random STF but prior EGFs
    fwd = FForward(x_sample, fwd_network, args.data_sigma, args.device)

    pphi = [F.mse_loss(y_x[i], fwd[:,i,:,:]) for i in range(args.num_egf)]

    net = lambda i: gf_network[i].module if len(args.device_ids) > 1 else gf_network[i]
    gf = [net(i).generategf() for i in range(args.num_egf)]          # no .detach()

    ## Priors on init GF
    prior = [sum(prior_phi[k](gf[i].squeeze(0), args.prior_phi_weight[k], i) for k in range(len(prior_phi)))
            for i in range(args.num_egf)]

    ## MSE loss
    meas_err = [args.egf_qual_weight[i] * torch.mean(gauss_nll(y[i], ytrue, args.sigma))
                for i in range(args.num_egf)]

    ## Multi M-steps for multiple EGFs
    zero = torch.zeros((), device=args.device)
    if args.num_egf > 1:
        mse_raw = torch.stack([F.mse_loss(y[i], ytrue) for i in range(args.num_egf)])
        alpha = (mse_raw / mse_raw.sum()).detach()  # Goodness of fit for EGF i
        idx_best = int(torch.argmin(torch.stack(meas_err)))
        gf_best = gf[idx_best].squeeze(0).detach()                    
        multi_loss = [zero if i == idx_best else
                    args.egf_multi_weight * alpha[i] * (Loss_L2(gf[i].squeeze(0), gf_best)
                        + 0.35 * Loss_DTW_Mstep(gf[i].squeeze(0), gf_best, gamma=1.0))
                    for i in range(args.num_egf)] # Closeness to best EGF (idx_best)
    else:
        multi_loss = [zero]

    loss = {i: meas_err[i] + prior[i] + pphi[i] + multi_loss[i] for i in range(args.num_egf)}
    return loss, meas_err, prior, multi_loss



######################################################################################################################
#       
#                            DPI
#
######################################################################################################################


class stf_logscale(nn.Module):
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

    def reverse(self, input):
        stf, logdet = self.realnvp.reverse(input)
        if self.softplus:
            out = F.softplus(stf)
            logdet = logdet + torch.sum(F.logsigmoid(stf), -1)
            # out = torch.nn.Sigmoid()(stf)
            # det_sigmoid = torch.sum(-stf - 2 * torch.nn.Softplus()(-stf), -1)
            # logdet = logdet + det_sigmoid
        else:
            out = stf
        return out, logdet


######################################################################################################################
#
#                            LOSSES
#
######################################################################################################################


def dtw_classic(x, y, dist='absolute'):
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
    return torch.mean(torch.abs(k - k0))

def Loss_TSV(z, z0):
    return torch.mean((z - z0)**2)

def Loss_L2(z, z0):
    return torch.sqrt(torch.sum((z - z0)**2))

def Loss_L1(z, z0):
    return torch.sum(torch.abs(z - z0))

def Loss_TV(z):
    return torch.abs(z[:, :, 1::] - z[:, :, 0:-1]).sum()

def Loss_DTW(z, z0):
    # Dynamic Time Warping loss with initial STF
    # not using fastDTW because does not allow different sizes for z and z0
    return dtw_classic(z, z0)

def Loss_DTW_Mstep(z, z0, gamma=0.1):
    # uses fast DTW, similar to L2 if aligned
    sdtw = soft_dtw_cuda.SoftDTW(use_cuda=False, gamma=gamma)
    a, b = z.transpose(-1, -2), z0.transpose(-1, -2)   # (batch, time, comp)
    return (sdtw(a, b) - 0.5 * (sdtw(a, a) + sdtw(b, b)))[0]

def null(x, y):
    return [0.]


######################################################################################################################
#       
#                            PLOT
#   
######################################################################################################################


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
myblue = '#244c77ff'
mycyan = '#3f7f93ff'
myred = '#c3553aff'
myorange = '#f07101'


def plot_seploss(args, Eloss_list, Eloss_mse_list, Eloss_prior_list, Eloss_q_list, Mloss_list, Mloss_mse_list, Mloss_phiprior_list, Mloss_multi_list, idx_egf):    
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))
    
    ax[0].plot(np.log10(Eloss_list), label="Estep")
    ax[0].plot(np.log10(Eloss_mse_list), "--", label="Estep MSE")
    ax[0].plot(np.log10(Eloss_prior_list), "--", label="Estep Priors")
    ax[0].plot(np.log10(Eloss_q_list), ":", label="q")
    
    ax[1].plot(np.log10(Mloss_list[idx_egf]), label="Mstep")
    ax[1].plot(np.log10(Mloss_mse_list[idx_egf]), "--", label="Mstep MSE")
    ax[1].plot(np.log10(Mloss_phiprior_list[idx_egf]), "--", label="Mstep Priors")
    
    if args.num_egf > 1:
        ax[1].plot(np.log10(Mloss_multi_list[idx_egf]), ":", label="Mstep Multi Loss")
    for k in range(2):
        ax[k].legend()
        ax[k].set_xlabel('sub.epochs #')
        ax[k].set_title(['Estep losses', 'Mstep losses'][k])

    fig.savefig("{}/SeparatedLoss_egf{}.png".format(args.PATH, idx_egf), dpi=300, bbox_inches='tight')
    plt.close()


def plot_res(k, k_sub, inferred_stf, learned_gf, learned_trc, gf0_np, trc0, args, true_stf=None, true_gf=None, step=''):
    mean_stf = np.mean(inferred_stf, axis=0)
    stdev_stf = np.std(inferred_stf, axis=0)
    mean_trc = [np.mean(learned_trc[i], axis=0) for i in range(len(learned_trc))]
    stdev_trc = [np.std(learned_trc[i], axis=0) for i in range(len(learned_trc))]

    for e in range(args.num_egf):
        inferred_gf = learned_gf[e][0]
        gf0 = gf0_np[e]
        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot2grid((12,4), (0, 0), colspan=3, rowspan=2)
        ax4 = plt.subplot2grid((12,4), (3, 3), colspan=1, rowspan=4)
        ax5 = plt.subplot2grid((12,4), (6, 0), colspan=3, rowspan=2)
        if args.num_comp > 1:
            ax2 = plt.subplot2grid((12,4), (2, 0), colspan=3, rowspan=2)
            ax6 = plt.subplot2grid((12,4), (8, 0), colspan=3, rowspan=2)
        if args.num_comp > 2:
            ax3 = plt.subplot2grid((12,4), (4, 0), colspan=3, rowspan=2)        
            ax7 = plt.subplot2grid((12,4), (10, 0), colspan=3, rowspan=2)
        
        x = np.arange(0, gf0.shape[1])

        ax1.set_title('EGF')
        ax5.set_title('Traces')
        ax4.set_title('STF')

        if true_gf is not None:
            true_gf = signal.resample(true_gf, gf0.shape[-1],axis=1)
            ax1.plot(x, true_gf[0], lw=0.5, color=myred, label='Target')
        ax1.plot(x, gf0[0], lw=0.5, color=myblue, label='Prior')
        ax1.plot(x, inferred_gf[0], lw=0.5, color=myorange, zorder=2, label='Inferred')
        ax1.text(0.03, 0.9, 'E',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax1.transAxes)

        ax1.legend(loc=(1.05, 0), frameon=False)
        if args.num_comp > 1:
            if true_gf is not None:
                ax2.plot(x, true_gf[1], lw=0.5, color=myred)
            ax2.plot(x, gf0[1], lw=0.5, color=myblue)
            ax2.plot(x, inferred_gf[1], lw=0.5, color=myorange, zorder=2)
            ax2.text(0.03, 0.9, 'N',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax2.transAxes)
            ax1.get_xaxis().set_visible(False)
        if args.num_comp > 2:
            if true_gf is not None:
                ax3.plot(x , true_gf[2], lw=0.5, color=myred)
            ax3.plot(x, gf0[2], lw=0.5, color=myblue)
            ax3.plot(x, inferred_gf[2], lw=0.5, color=myorange, zorder=2)
            ax3.text(0.03, 0.9, 'Z',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax3.transAxes)
            ax2.get_xaxis().set_visible(False)
        
        

        # Traces
        learned_trace = mean_trc[e]

        ax5.plot(trc0[0], lw=0.5, color=myred)
        ax5.plot(learned_trace[0], lw=0.5, color=myorange, zorder=2)
        ax5.text(0.03, 0.9, 'E',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax5.transAxes)
        if args.num_comp > 1:
            ax6.plot(trc0[1], lw=0.5, color=myred)
            ax6.plot(learned_trace[1], lw=0.5, color=myorange, zorder=2)
            ax6.text(0.03, 0.9, 'N',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax6.transAxes)
            ax5.get_xaxis().set_visible(False)
        if args.num_comp > 2:
            ax7.plot(trc0[2], lw=0.5, color=myred)
            ax7.plot(learned_trace[2], lw=0.5, color=myorange, zorder=2)
            ax7.text(0.03, 0.9, 'Z',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax7.transAxes)
            ax6.get_xaxis().set_visible(False)
            
        # STF, in moment-ratio units (sum = M0 / M0_egf)
        scale = getattr(args, "c_trc", 1.0) / getattr(args, "c_gf", [1.0])[0]
        stf_mean, stf_std = mean_stf[0] * scale, stdev_stf[0] * scale
        K = stf_mean.shape[0]
        tinf = np.arange(K) / args.samp_rate if args.samp_rate else np.arange(K)
        if true_stf is not None:
            ts = np.asarray(true_stf, dtype=float).ravel()
            if len(ts) != K:                                   # centre it as the forward model does
                full = np.zeros(K)
                o = (K - 1) // 2 - (len(ts) - 1) // 2
                idx = np.arange(len(ts)) + o
                ok = (idx >= 0) & (idx < K)
                full[idx[ok]] = ts[ok]
                ts = full
            ax4.plot(tinf, ts, lw=0.8, color=myred, label=f'True (Σ={ts.sum():.3g})')
        ax4.plot(tinf, stf_mean, lw=1, color=myorange, label=f'Mean (Σ={stf_mean.sum():.3g})')
        ax4.fill_between(tinf, stf_mean - 2 * stf_std, stf_mean + 2 * stf_std,
                         facecolor=myorange, alpha=0.35, zorder=0, label='2σ')
        ax4.legend(loc='upper right', fontsize=5, frameon=False)
        ax4.set_xlabel('time (s)' if args.samp_rate else 'samples', fontsize=6)

        fig.savefig("{}/out_egf{}_{}_{}{}.png".format(args.PATH, str(e), str(k).zfill(5), step, str(k_sub).zfill(5)), dpi=300, bbox_inches="tight")
        plt.close()


def plot_st(st_trc, st_gf, inferred_trace, inferred_gf, inferred_stf, args):
    mean_trc = np.mean(inferred_trace, axis=1)   
    stdev_trc = np.std(inferred_trace, axis=1)
    gf0 = np.concatenate([st_gf[k].data[:, None] for k in range(len(st_gf))], axis=1).T
    gf0 = gf0.reshape(gf0.shape[0] // args.num_comp, args.num_comp, gf0.shape[1], order='F')
    trc0 = np.concatenate([st_trc[k].data[:, None] for k in range(len(st_trc))], axis=1).T

    gf0 /= np.amax(np.abs(gf0), axis=(1, 2), keepdims=True) 
    trc0 /= np.amax(np.abs(trc0))

    scale = getattr(args, "c_trc", 1.0) / getattr(args, "c_gf", [1.0])[0]
    mean_stf, stdev_stf = np.mean(inferred_stf, axis=0) * scale, np.std(inferred_stf, axis=0) * scale
    A = inferred_stf.reshape(inferred_stf.shape[0], -1).sum(-1) * scale

    if args.num_egf == 1:
        rap = [np.amax(st_trc[i].data) / np.amax(st_gf[i].data) for i in range(args.num_comp)]

        fig = plt.figure(figsize=(6, 1.2))
        plt.subplots_adjust(wspace=0.15)
        ax = plt.subplot(1, 3, 1)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tick_params(
            axis='x',
            which='both',
            bottom=True,
            top=False,
            labelbottom=True)
        chan = ['E', 'N', 'Z']
        for i in range(args.num_comp):
            tmax = np.amax(st_trc[0].times())
            ax.fill_between(st_trc[0].times() - (2 - i) * tmax // 5, mean_trc[0,i] - stdev_trc[0,i] + (2 - i) * 0.6,
                            mean_trc[0,i] + stdev_trc[0,i] + (2 - i) * 0.6,
                            facecolor=myorange, alpha=0.25, zorder=0, label='Standard deviation', clip_on=False)
            l1 = ax.plot(st_trc[0].times() - (2 - i) * tmax // 5, trc0[i] + (2 - i) * 0.6, color='k', lw=0.7,
                         clip_on=False)
            l2 = ax.plot(st_trc[0].times() - (2 - i) * tmax // 5, mean_trc[0,i] + (2 - i) * 0.6, lw=0.6, color=myorange,
                         clip_on=False)
            if args.num_comp != 1:
                ax.text(np.amin(st_trc[0].times() - (2 - i) * tmax // 5) - 5, np.mean(trc0[i] + (2 - i) * 0.6), chan[i],
                        horizontalalignment='right',
                        verticalalignment='top', weight='bold')
            ax.text(np.amin(st_trc[0].times() - (2 - i) * tmax // 5), (2 - i) * 0.6 + 0.25, 'x ' + str(int(rap[i])),
                    horizontalalignment='left', verticalalignment='top', fontsize='small')
        plt.xlim(np.amin(st_trc[0].times() - (2) * tmax // 5) + tmax // 5, tmax - tmax // 7)
        plt.xlabel('time (s)', labelpad=2, loc='left')
        ticklab = ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        ax.xaxis.set_label_coords(-2 * tmax // 5, 0, transform=trans)

        ax = plt.subplot(1, 3, 2)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tick_params(
            axis='x',
            which='both',
            bottom=True,
            top=False,
            labelbottom=True)
        for i in range(args.num_comp):
            tmax = np.amax(st_gf[0].times())
            ax.plot(st_gf[0].times() - (2 - i) * tmax // 5, gf0[0,i] + (2 - i) * 0.6, color='k', lw=0.7, clip_on=False)
            ax.plot(st_gf[0].times() - (2 - i) * tmax // 5, inferred_gf[0,0,i] + (2 - i) * 0.6, lw=0.6, color=myorange,
                    clip_on=False)
        plt.xlim(np.amin(st_gf[0].times() - (2) * tmax // 5) + tmax // 5, tmax - tmax // 7)
        ax.text(1, .8, 'data', horizontalalignment='right', verticalalignment='top', color='k', transform=ax.transAxes)
        ax.text(1, 0.9, 'predictions', horizontalalignment='right', verticalalignment='top', color=myorange,
                transform=ax.transAxes)

        ax = plt.subplot(1, 3, 3)
        sr = st_gf[0].stats.sampling_rate
        if args.M0_egf is not None:
            conv, unit = sr * args.M0_egf, 'N·m/s'
            amp = rf"$M_0$ = {A.mean() * args.M0_egf:.2e} N·m"
        else:
            conv, unit = sr, r's$^{-1}$'
            amp = rf"$M_0/M_0^{{EGF}}$ = {A.mean():.3g}"
        t = np.arange(len(mean_stf[0])) / sr
        lo, mid, hi = (mean_stf[0] - stdev_stf[0]) * conv, mean_stf[0] * conv, (mean_stf[0] + stdev_stf[0]) * conv
        p10 = int(np.floor(np.log10(np.nanmax(hi)))) if np.nanmax(hi) > 0 else 0   # put the exponent in the label
        lo, mid, hi = lo / 10 ** p10, mid / 10 ** p10, hi / 10 ** p10
        ylab = (r'$\dot{M}$' if args.M0_egf is not None else r'$\dot{M}/M_0^{EGF}$') + rf' ($10^{{{p10}}}$ {unit})'

        ax.fill_between(t, lo, hi, facecolor=myorange, alpha=0.25, lw=0, zorder=0)
        ax.plot(t, mid, lw=1, color=myorange)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(bottom=0)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.get_yaxis().set_visible(True)
        ax.tick_params(axis='both', which='both', top=False)
        ax.set_ylabel(ylab)
        ax.text(0.04, 0.95, amp, ha='left', va='top', fontsize=6, transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.35', fc='lightgray', ec='none', alpha=0.8))

    else:
        subfig = 1
        fig = plt.figure(figsize=(6, (args.num_egf + 1) * 1.2))
        plt.subplots_adjust(wspace=0.2, hspace=0.35)

        ax = plt.subplot(args.num_egf+1, 2, 1)
        ax.text(-0.2, 0.9, '({})'.format(chr(ord('`') + subfig)),
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, weight='bold')
        subfig += 1

        sr = st_gf[0].stats.sampling_rate
        if args.M0_egf is not None: 
            conv, ylab = sr * args.M0_egf, 'Moment rate (N·m/s)'
            amp = f"M0 = {A.mean() * args.M0_egf:.2e} N·m"
        else:                                                                       # relative to the EGF
            conv, ylab = sr, r'$\dot{M}\,/\,M_0^{EGF}$ (s$^{-1}$)'
            amp = f"M0/M0_egf = {A.mean():.3g}"
        t = np.arange(len(mean_stf[0])) / sr
        ax.spines['right'].set_visible(True)
        ax.get_yaxis().set_visible(True)
        plt.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, labelsize='small')
        ax.fill_between(t, (mean_stf[0] - stdev_stf[0]) * conv, (mean_stf[0] + stdev_stf[0]) * conv,
                        facecolor=myorange, alpha=0.25, zorder=0, label='Standard deviation')
        ax.plot(t, mean_stf[0] * conv, lw=0.8, color=myorange)
        ax.set_ylabel(ylab, fontsize='small')
        ax.yaxis.get_offset_text().set_fontsize('x-small')
        ax.text(1, 1.02, amp, ha='right', va='bottom', fontsize='x-small', transform=ax.transAxes)
        
        plt.xlabel('time (s)', labelpad=2, loc='left')
        ax.text(0, .4, 'Data', horizontalalignment='left', verticalalignment='top', color='k', transform=ax.transAxes)
        ax.text(0,0.2, 'Predictions', horizontalalignment='left', verticalalignment='top', color=myorange,
                transform=ax.transAxes)

        chan = ['E', 'N', 'Z']

        ## mean egf
        ax = plt.subplot(args.num_egf + 1, 2, 2)
        ax.text(-0.2, 0.9, '({})'.format(chr(ord('`') + subfig)),
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, weight='bold')
        subfig += 1
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tick_params(bottom=True, top=False, labelbottom=False)

        mean_egf = np.mean(inferred_gf, axis=0)[0]
        std_egf = np.std(inferred_gf, axis=0)[0]
        tmax = np.amax(st_gf[0].times())
        for i in range(args.num_comp):
            ax.fill_between(st_gf[0].times() - (2 - i) * tmax // 5, mean_egf[i] - std_egf[i] + (2 - i) * 0.6,
                            mean_egf[i] + std_egf[i] + (2 - i) * 0.6,
                            facecolor='#632f48', alpha=0.25, zorder=0, label='Standard deviation')
            ax.plot(st_gf[0].times() - (2 - i) * tmax // 5, mean_egf[i] + (2 - i) * 0.6, lw=0.5, color='#632f48',
                    clip_on=False)
            ax.text(np.amin(st_gf[0].times() - (2 - i) * tmax // 5) - tmax / 20, np.mean(mean_egf[i] + (2 - i) * 0.6),
                    chan[i],
                    horizontalalignment='right',
                    verticalalignment='top')
        ax.text(1.1, 0.9, 'Mean EGF', horizontalalignment='right', verticalalignment='top', color='#632f48',
                transform=ax.transAxes)
        plt.xlim(np.amin(st_gf[0].times() - (2) * tmax // 5) + tmax // 5, tmax - tmax // 7)

        for k in range(args.num_egf):
            rap = [np.amax(st_trc[i].data) / np.amax(st_gf[k+args.num_egf*i].data) for i in range(args.num_comp)]
            ax = plt.subplot(args.num_egf+1, 2, k*2+3)
            ax.text(-0.2, 0.9, '({})'.format(chr(ord('`') + subfig)),
                    horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, weight='bold')
            subfig += 1
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.tick_params(bottom=True, top=False, labelbottom=False)

            tmax = np.amax(st_trc[0].times())
            for i in range(args.num_comp):
                ax.fill_between(st_trc[0].times()-(2-i)*tmax//5, mean_trc[k,i] - stdev_trc[k,i]+(2-i)*0.6, mean_trc[k,i] + stdev_trc[k,i]+(2-i)*0.6,
                                facecolor=myorange, alpha=0.25, zorder=0, label='Standard deviation',clip_on=False)
                l1 = ax.plot(st_trc[0].times()-(2-i)*tmax//5, trc0[i]+(2-i)*0.6, color='k', lw=0.7,clip_on=False)
                l2 = ax.plot(st_trc[0].times()-(2-i)*tmax//5, mean_trc[k,i]+(2-i)*0.6, lw=0.6, color=myorange,clip_on=False)
                ax.text(np.amin(st_trc[0].times() - (2 - i) * tmax // 5) -tmax/20, np.mean(trc0[i] + (2 - i) * 0.6), chan[i],
                        horizontalalignment='right',
                        verticalalignment='top')
                # ax.text(np.amin(st_trc[0].times() - (2 - i) * tmax // 5) , (2 - i) * 0.6+0.3, 'x '+str(int(rap[i])),
                #         horizontalalignment='left', verticalalignment='top', fontsize='small')
            plt.xlim(np.amin(st_trc[0].times() - (2) * tmax // 5) + tmax // 5, tmax - tmax // 7)
            if k == args.num_egf - 1:
                plt.xlabel('time (s)', labelpad=2, loc='left')
                plt.tick_params(bottom=True, top=False, labelbottom=True)
                ticklab = ax.xaxis.get_ticklabels()[0]
                trans = ticklab.get_transform()
                ax.xaxis.set_label_coords(-2 * tmax // 5, 0, transform=trans)

            ax = plt.subplot(args.num_egf+1, 2, k*2+4)
            ax.text(-0.2, 0.9, '({})'.format(chr(ord('`') + subfig)),
                    horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, weight='bold')
            subfig += 1
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.tick_params(bottom=True, top=False, labelbottom=False)
            for i in range(args.num_comp):
                tmax = np.amax(st_gf[0].times())
                ax.plot(st_gf[0].times()-(2-i)*tmax//5, gf0[k,i]+(2-i)*0.6, color='k', lw=0.7,clip_on=False)
                ax.plot(st_gf[0].times()-(2-i)*tmax//5, inferred_gf[k,0,i]+(2-i)*0.6, lw=0.6, color=myorange,clip_on=False)
            plt.xlim(np.amin(st_gf[0].times() - (2) * tmax // 5) + tmax // 5, tmax - tmax // 7)

            if k == args.num_egf - 1:
                plt.tick_params(bottom=True, top=False, labelbottom=True)

    figname = "{}/out_{}.pdf".format(args.PATH, 'res')
    fig.savefig(figname, bbox_inches="tight")
    plt.close()

    return figname
