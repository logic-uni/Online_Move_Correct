"""
# coding: utf-8
@author: Yuhao Zhang
last updated : 09/20/2023
data from: Xinrong Tan
data collected: 05/10/2023
"""

import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.pyplot import *
import pynacollada as pyna
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
from matplotlib.colors import hsv_to_rgb
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import log
import warnings
import scipy.io as sio
np.set_printoptions(threshold=np.inf)

# Experiment info
t=3041.407
sample_rate=30000 #spikeGLX neuropixel sample rate
marker_path = 'E:/marker/20230510/rod_marker.csv'
identities = np.load('E:/xinrong/20230510/cage1-2-R-1_g0/cage1-2-R-1_g0_imec0/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load('E:/xinrong/20230510/cage1-2-R-1_g0/cage1-2-R-1_g0_imec0/spike_times.npy')  #
channel = np.load('E:/xinrong/20230510/cage1-2-R-1_g0/cage1-2-R-1_g0_imec0/channel_positions.npy')

def marker():

    with open(marker_path) as file_name:
        array = np.loadtxt(file_name, delimiter=",")

    array=array.astype('int64')
    marker_raw=np.array([])
    marker=np.array([])

    #Binarization
    for i in range(len(array)):
        if array[i]>2:
            array[i]=1
        else:
            array[i]=0
    #Rising edge detection
    for m in range(len(array)):
        if array[m]-array[m-1]==1:
            marker = np.append(marker,m/10593)  #10593 is sample rate of marker of spikeGLX

    #push rod time +-1s
    marker_1s=np.empty([len(marker),2]) 
    for a in range(len(marker)):
        marker_1s[a,0]=marker[a]-0.5
        marker_1s[a,1]=marker[a]+0.5

    #remove start 1000s
    clean=np.array([])
    for e in range(len(marker)):
        if marker[e]>1000:
            clean=np.append(clean,marker[e])

    return marker

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

# spike counts
def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1))
    domain += offsets[:, None]
    return callback(domain)

def build_spike_histogram(time_domain,
                          spike_times,
                          dtype=None,
                          binarize=False):

    time_domain = np.array(time_domain)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]

    data = np.array(spike_times)

    start_positions = np.searchsorted(data, starts.flat)
    end_positions = np.searchsorted(data, ends.flat, side="right")
    counts = (end_positions - start_positions)

    tiled_data[:, :].flat = counts > 0 if binarize else counts

    return tiled_data

def spike_counts(
    spike_times,
    bin_edges,
    movement_start_time,
    binarize=False,
    dtype=None,
    large_bin_size_threshold=0.001,
    time_domain_callback=None
):

    #build time domain
    bin_edges = np.array(bin_edges)
    domain = build_time_window_domain(
        bin_edges,
        movement_start_time,
        callback=time_domain_callback)

    out_of_order = np.where(np.diff(domain, axis=1) < 0)
    if len(out_of_order[0]) > 0:
        out_of_order_time_bins = \
            [(row, col) for row, col in zip(out_of_order)]
        raise ValueError("The time domain specified contains out-of-order "
                            f"bin edges at indices: {out_of_order_time_bins}")

    ends = domain[:, -1]
    starts = domain[:, 0]
    time_diffs = starts[1:] - ends[:-1]
    overlapping = np.where(time_diffs < 0)[0]

    if len(overlapping) > 0:
        # Ignoring intervals that overlaps multiple time bins because
        # trying to figure that out would take O(n)
        overlapping = [(s, s + 1) for s in overlapping]
        warnings.warn("You've specified some overlapping time intervals "
                        f"between neighboring rows: {overlapping}, "
                        "with a maximum overlap of"
                        f" {np.abs(np.min(time_diffs))} seconds.")
        
    #build_spike_histogram
    tiled_data = build_spike_histogram(
        domain,
        spike_times,
        dtype=dtype,
        binarize=binarize
    )
    return tiled_data

def firingrate_time(id,marker):

    # bin
    bin_width = 0.14
    duration = 2  #一个trial的时间，或你关注的时间段的长度
    pre_time = 0
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)  
    # histograms
    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    #print(histograms)
    return histograms


def population_spikecounts(marker):
    neuron_id=np.array([1170,1171,1172,1173,1174,1176,1177,1178,1180,1181,1185,1188,1196,1197,1198,1199,1200,1211])

    #get a 2D matrix with neurons, trials(trials contain times), trials and times are in the same dimension
    for j in range(len(neuron_id)): #第j个neuron

        #每个neuron的tials水平append
        for i in range(len(marker)):
            if i == 0:
                one_neruon = firingrate_time(neuron_id[j],marker)[0]
            else:
                trail = firingrate_time(neuron_id[j],marker)[i]
                one_neruon = np.append(one_neruon, trail)

        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    
    neurons_topca=neurons.T

    print(neurons_topca.shape)
    return neurons_topca

def manifold():
    bin_size=0.1
    count=population_spikecounts(marker()[20:130])
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)

    #reduce dimension
    rd = Isomap(n_components = 2, n_neighbors = 50).fit_transform(rate.values)
    #rd = TSNE(n_components=2,random_state=21,perplexity=10).fit_transform(rate.values)
    print(rd[:,0])
    plt.plot(rd[:,0], rd[:,1])
    #plt.show()
    #X_pca = PCA(n_components=3).fit_transform(rate.values)

    #plot dynamic graph
    '''
    x_track=np.zeros((1,2))
    x_track[0,0]=rd[0,0]
    x_track[0,1]=rd[0,1]
    plt.grid(True)
    plt.ion()  # interactive mode on!!!! 很重要,有了他就不需要plt.show()了

    for i in range(0,len(rd)):
        plt.plot(x_track[:,0], x_track[:,1])
        x_track_s=[rd[i,0],rd[i,1]]
        x_track = np.vstack((x_track, x_track_s))
        plt.pause(0.2)
    '''
    xdata=rd[:,0]
    ydata=rd[:,1]
    u=np.diff(xdata, n=1)
    v=np.diff(ydata, n=1)
    u = np.insert(u, 0, 0)
    v = np.insert(v, 0, 0)

    #plot vector field  u=dPC1, v=dPC2, xdata=PC1. ydata=PC2
    #plt.quiver(u,v,xdata,ydata,width = 0.003,scale = 100)     #X,Y,U,V 确定位置和对应的导数
    #plot vector field  u=dPC1, v=dPC2, xdata=PC1. ydata=PC2
    #plt.quiver(xdata,ydata,u,v,width = 0.005,scale = 100)

    plt.show()

manifold()
