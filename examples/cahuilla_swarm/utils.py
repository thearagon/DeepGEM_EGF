# Import Python Libraries
import numpy as np
import sys
import os
import copy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import pandas as pd
import subprocess
from scipy import signal
from scipy import stats
import obspy
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta
import random
from datetime import datetime
import requests
import json
# from gradio_client import Client as gradioclient


def dist2lat(kms):
    "Given a distance north, return the change in latitude."
    earth_radius = 6271.0
    radians_to_degrees = 180.0 / math.pi
    return (kms/earth_radius)*radians_to_degrees

def dist2lon(latitude, kms):
    "Given a latitude and a distance west, return the change in longitude."
    earth_radius = 6271.0
    degrees_to_radians = math.pi / 180.0
    radians_to_degrees = 180.0 / math.pi
    r = earth_radius*math.cos(latitude*degrees_to_radians)
    return (kms/r)*radians_to_degrees

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a =np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def signalPower(x):
    return np.mean(x**2)

def SNR(signal, noise):
    powS = signalPower(signal)
    powN = signalPower(noise)
    return powS/powN

def writeDeepGEMlauncher(datadir,outdir,stations,num_egf,epochs=10,epochsE=40,epochsM=20):
    
    with open('EGF_ex.sh', 'w') as fi:
        fi.write(f'''
#!/bin/bash

datadir={datadir}
gemdir=$1

# Declare arrays
declare -a sta=({' '.join(stations)})
declare -a numegf=({' '.join(map(str, num_egf))})

arraylength=${{#sta[@]}}

for (( i=0; i<${{arraylength}}; i++ ));
do
    outdir={outdir}/${{sta[$i]}}/
    mkdir -p $outdir
    
    python3 $gemdir/GEMDeconvEgf.py -dir $outdir  \
        -trc0 $datadir/${{sta[$i]}}_trc_P.mseed  \
        -egf0 $datadir/${{sta[$i]}}_multi_gf_P.mseed  \
        --num_egf ${{numegf[$i]}}  \
        --num_epochs {epochs}  \
        --num_subepochsE {epochsE}  \
        --num_subepochsM {epochsM}  \
        --EMFull  \
        --output  \
        --data_sigma 1e-6  \
        --Elr 6e-3  \
        --Mlr 1e-4  \
        --stf_dur 2  \
        --egf_multi_weight .5  \
        --stf0_sigma 1e2
done''')
        
    return 'EGF_ex.sh'
                 
