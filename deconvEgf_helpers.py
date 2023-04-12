#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from torch.nn import MSELoss

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
# import math
import itertools
import scipy
from scipy import signal
# import argparse

from generative_model import realnvpfc_model
import obspy


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
        init = makeInit(ini, self.num_layers).reshape((self.num_layers, 1, 3*self.num_egf, ini.shape[-1]))
        init = torch.Tensor(init).to(device=self.device)

        self.layers = torch.nn.Parameter(init, requires_grad = True)

    def load(self,filepath,device):
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=True)
        
    def generatekernel(self):
        ker = self.layers[0]
        if self.num_layers >= 2:
            for i in range(1, self.num_layers):
                ker = F.conv1d(ker, self.layers[i].view(self.num_egf* 3, 1, self.layers[0].shape[-1]).flip(2),
                               padding='same', groups=3*self.num_egf)
        out = ker / torch.max(torch.abs(ker)).to(device=self.device)
        return out.view(out.shape[0], self.num_egf, 3, out.shape[-1])

    def forward(self, x):
        k = self.generatekernel()
        out = F.conv1d(k.view(3*self.num_egf,1,k.shape[-1]),x.flip(2), padding='same' )
        out = torch.transpose(out, 0, 1)
        out = out.view(out.shape[0], self.num_egf, 3, out.shape[-1]).to(device=self.device)
        return out

def trueForward(k, x, num_egf):
    out = F.conv1d(k.view(3*num_egf,1, k.shape[-1]), x.flip(2), padding='same', groups=1)
    out = torch.transpose(out, 0, 1)
    out = out.view(out.shape[0], num_egf, 3, out.shape[-1])
    return out


def makeInit(init, num_layers, noise_amp=.5):
    """
    """
    l0 = np.zeros(init.shape)
    l0[:, :, init.shape[-1]//2] = 1.

    out = np.zeros((num_layers, init.shape[0], init.shape[1], init.shape[-1]))
    for i in range(num_layers - 1):
        out[i] = l0 + (np.random.rand() * noise_amp / 100.) * np.random.rand(*l0.shape)
    out[-1] = init + (2 * noise_amp / 100.) * np.random.rand(*l0.shape)

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
        ini = 0.05*torch.randn(imginit.shape) + 0.9*imginit
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


def EStep(z_sample, device, ytrue, img_generator, kernel_network, prior_x, prior_img, logdet_weight,
          prior_x_weight, img_prior_weight, sigma, npix, npiy,logscale_factor, data_weight, device_ids=None, num_egf=1):

    img, logdet = GForward(z_sample, img_generator, npix,npiy, logscale_factor, device=device, device_ids=device_ids)
    y = FForward(img, kernel_network, sigma, device)

    ## log likelihood
    logqtheta = -logdet_weight*torch.mean(logdet)
    ## prior on trace
    meas_err = data_weight*nn.MSELoss()(y, ytrue)

    ## prior on STF
    priorx = torch.mean(prior_x(img, prior_x_weight))
    if isinstance(prior_img, list):
        if isinstance(img_prior_weight, list):
            priorimg = torch.mean( torch.Tensor( [torch.mean(prior_img[i](img, img_prior_weight[i])) for i in range(len(prior_img))] ))
        else:
            priorimg = torch.mean( torch.Tensor( [torch.mean(prior_img[i](img, img_prior_weight)) for i in range(len(prior_img))] ))
    else:
        priorimg = torch.mean(prior_img(img, img_prior_weight))
    loss = logqtheta + priorx + meas_err + priorimg
    return loss, logqtheta, priorx+priorimg, meas_err


def MStep(z_sample, x_sample, npix, npiy, device, ytrue, img_generator, kernel_network, phi_weight,
          fwd_network, sigma, logscale_factor, prior_phi, prior_phi_weight, ker_softl1, kernel_norm_weight, k_weight, prior_k, device_ids=None, num_egf=1):

    # inferred IMG
    img, logdet = GForward(z_sample, img_generator, npix,npiy, logscale_factor, device=device, device_ids=device_ids)
    # TRC from inferred IMG
    y = FForward(img, kernel_network, sigma, device)
    # TRC from random IMG
    y_x = FForward(x_sample, kernel_network, sigma, device)
    # TRC from random IMG but init GF
    fwd = FForward(x_sample, fwd_network, sigma, device)
    pphi = phi_weight*nn.MSELoss()(y_x, fwd)

    if device_ids is not None:
        kernel = kernel_network.module.generatekernel()
    else:
        kernel = kernel_network.generatekernel()

    ## Priors on init GF
    prior = 3*prior_phi_weight * prior_phi[0](kernel.squeeze(0))
    prior += prior_phi[1](kernel.squeeze(0), prior_phi_weight )

    ## Soft L1
    norm_k = kernel_norm_weight * ker_softl1(kernel_network)

    #+ correlation if multiple EGFs
    if k_weight > 0:
        norm_k += torch.mean(prior_k(kernel.squeeze(0), k_weight ))

    # loss = nn.MSELoss(reduction='none')(y, ytrue)
    # meas_err = (phi_weight/sigma)*torch.mean(torch.Tensor([torch.mean(loss[:,i,:])/torch.max(loss[:,i,:]) for i in range(loss.shape[1]) ]))
    meas_err = (1e-1/sigma)*nn.MSELoss()(y, ytrue)
    loss =  pphi + meas_err + norm_k + prior

    return loss, pphi, meas_err, norm_k, prior


######################################################################################################################

        
#                            DPI
    

######################################################################################################################



class img_logscale(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, scale=1):
        super().__init__()
        log_scale = torch.Tensor(torch.log(scale)*torch.ones(1))
        self.log_scale = nn.Parameter(log_scale)

    def forward(self):
        return self.log_scale
    
    
def priorPhi(k, k0):
    ker = k - k0
    # out = ker[0]/torch.max(torch.abs(ker[0])) + ker[1]/torch.max(torch.abs(ker[1])) + ker[2]/torch.max(torch.abs(ker[2]))
    out = ker
    return torch.mean(torch.abs(out))

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
        return (x - y) ** 2

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

def Loss_TSV(z, z0):
    return torch.mean( (z - z0)**2 )

def Loss_TV_3c(z):
    loss =  torch.abs(z[:, 1::] - z[:, 0:-1])
    TV = torch.mean( torch.Tensor( [torch.mean( loss[i,:])/torch.max(loss[i,:]) for i in range(loss.shape[0]) ]))
    return TV

def Loss_TV(z):
    # total variation loss
    # return torch.mean( torch.abs(z[:, 1::] - z[:, 0:-1]), (-1))
    return torch.mean( torch.abs(z[:,:, 1::] - z[:,:, 0:-1]))

def Loss_DTW(z, z0):
    # Dynamic Time Warping loss with initial STF
    # Time shift insensitive
    return dtw_classic(z, z0)

def Loss_multicorr(z):
    # calculates Pearson coeff/TV for every channel of every couple of EGF

    n = len(z)
    nbr_combi = np.math.factorial(n) / 2 / np.math.factorial(n-2)
    coef = torch.zeros(( int(nbr_combi) ,3))

    for i,co in enumerate(itertools.combinations(range(n), 2)):
        for k in range(3):
            # Pearson coeff
            # coef[i,k] = torch.corrcoef(z[co, k, :])[0,1]

            # TV
            # coef[i,k] = torch.mean(torch.abs(z[co[0], k, :] - z[co[1], k, :]) )

            # DTW
            coef[i,k] = dtw_classic(z[co[0], k, :], z[co[1], k, :])

    # return 1 - torch.mean(torch.abs( coef ))
    return torch.mean(coef) / 10**(torch.floor(torch.log10(torch.mean(coef))))


######################################################################################################################

        
#                            PLOT
    

######################################################################################################################

def plot_res(k, k_sub, image, learned_k, stf0, gf, args, true_stf=None, true_gf=None):
    mean_img = np.mean(image, axis=0)
    stdev_img = np.std(image, axis=0)
    gf_np = gf.detach().cpu().numpy()
    stf0 = stf0.detach().cpu().numpy()

    for e in range(args.num_egf):
        learned_kernel = learned_k[e]
        gf = gf_np[e]
        fig = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=3, rowspan=2)
        ax2 = plt.subplot2grid((6, 4), (2, 0), colspan=3, rowspan=2)
        ax3 = plt.subplot2grid((6, 4), (4, 0), colspan=3, rowspan=2)
        ax4 = plt.subplot2grid((6, 4), (2, 3), colspan=1, rowspan=4)
        x = np.arange(0, gf.shape[1])

        if true_gf is not None:
            true_gf = signal.resample(true_gf, gf.shape[-1],axis=1)
            ax1.plot(x, true_gf[0], lw=0.5, color=myred, label='Target')
        ax1.plot(x, gf[0], lw=0.5, color=myblue, label='Prior')
        ax1.plot(x, learned_kernel[0], lw=0.5, color=myorange, zorder=2, label='Inferred')
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

        # x0 = np.linspace(0, len(stf0), len(stf0))
        xinf = np.linspace(0, mean_img.shape[1], mean_img.shape[1])
        if args.px_init_weight > 0:
            if mean_img.shape[1] > len(stf0):
                stf_rs = np.zeros(mean_img[0].shape)
                stf_rs[:len(stf0)] = stf0
                ax4.plot(xinf, stf_rs, lw=0.8, color=myblue)
            else:
                ax4.plot(xinf, stf0, lw=0.8, color=myblue)
        if true_stf is not None:
            # xtrue = np.linspace(0, len(stf0), len(true_stf))
            true_stf_rs = np.zeros(mean_img[0].shape)
            true_stf_rs[:len(true_stf)] = true_stf
            ax4.plot(xinf, true_stf_rs, lw=0.8, color=myred)
        ax4.plot(xinf, mean_img[0], lw=1, color=myorange)
        ax4.fill_between(xinf, mean_img[0] - stdev_img[0], mean_img[0] + stdev_img[0],
                         facecolor=myorange, alpha=0.35, zorder=0, label='Standard deviation')

        fig.savefig("{}/out_egf{}_{}_{}.png".format(args.PATH, str(e), str(k).zfill(5), str(k_sub).zfill(5)), format='png', dpi=300,
                bbox_inches="tight")
        plt.close()
    return


def plot_st(st_trc, st_gf, image_blur, learned_kernel, image, args):
    mean_img = np.mean(image, axis=0)
    stdev_img = np.std(image, axis=0)
    mean_trc = np.mean(image_blur, axis=0)
    stdev_trc = np.std(image_blur, axis=0)
    gf = np.concatenate([st_gf[k].data[:, None] for k in range(len(st_gf))], axis=1).T
    gf = gf.reshape(gf.shape[0] // 3, 3, gf.shape[1])
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
            ax.plot(st_gf[0].times() - (2 - i) * tmax // 5, learned_kernel[0,i] + (2 - i) * 0.6, lw=0.7, color=myorange,
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
                ax.plot(st_gf[0].times()-(2-i)*tmax//5, learned_kernel[k,i]+(2-i)*0.6, lw=0.7, color=myorange,clip_on=False)
            plt.xlim(np.amin(st_gf[0].times()-(2)*tmax//5)+10, tmax-5)

    fig.savefig("{}/out_{}.pdf".format(args.PATH, 'res'),
                bbox_inches="tight")
    plt.close()
    return


def plot_trace(trc, image_blur, args):

    mean_blur_img = np.mean(image_blur, axis=(0,1))
    stdev_blur_img = np.std(image_blur, axis=(0,1))
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

def plot_trace_diff(trc, image_blur, args):

    mean_blur_img = np.mean(image_blur, axis=0)
    stdev_blur_img = np.std(image_blur, axis=0)
    std_max = stdev_blur_img.max()
    truetrc = trc.detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot2grid((6, 4), (0, 0), colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((6, 4), (2, 0), colspan=3, rowspan=2)
    ax3 = plt.subplot2grid((6, 4), (4, 0), colspan=3, rowspan=2)
    ax1.plot(truetrc[0], lw=0.5, color=myblue)
    for e in range(args.num_egf):
        ax1.plot(mean_blur_img[e,0], lw=0.5, color=myorange, zorder=50)
        ax1.fill_between(np.arange(len(mean_blur_img[e,0])), mean_blur_img[e,0] - stdev_blur_img[e,0],
                         mean_blur_img[e,0] + stdev_blur_img[e,0],
                         facecolor=myorange, alpha=0.25, zorder=0)
    ax1.text(0.03, 0.9, 'E',
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes)
    ax2.plot(truetrc[1], lw=0.5, color=myblue)
    for e in range(args.num_egf):
        ax2.plot(mean_blur_img[e,1], lw=0.5, color=myorange, zorder=50)
        ax2.fill_between(np.arange(len(mean_blur_img[e,1])), mean_blur_img[e,1] - stdev_blur_img[e,1],
                         mean_blur_img[e,1] + stdev_blur_img[e,1],
                         facecolor=myorange, alpha=0.25, zorder=0)
    ax2.text(0.03, 0.9, 'N',
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax2.transAxes)
    ax3.plot(truetrc[2], lw=0.5, color=myblue)
    for e in range(args.num_egf):
        ax3.plot(mean_blur_img[e,2], lw=0.5, color=myorange, zorder=50)
        ax3.fill_between(np.arange(len(mean_blur_img[e,2])), mean_blur_img[e,2] - stdev_blur_img[e,2],
                         mean_blur_img[e,2] + stdev_blur_img[e,2],
                         facecolor=myorange, alpha=0.25, zorder=0)
    ax3.text(0.03, 0.9, 'Z',
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax3.transAxes)
    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("{}/outTRC_multi.pdf".format(args.PATH), format='pdf',
                bbox_inches="tight")
    plt.close()
    return
