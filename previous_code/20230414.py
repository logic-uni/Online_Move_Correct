'''Author: Yuhao Zhang, Date: 20230503'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import log
import warnings
import scipy.io as sio
np.set_printoptions(threshold=np.inf)

t=2668.5
sample_rate=30000
marker_path = '/marker/20230414/rod_marker.csv'
identities = np.load('/data/20230414/cage2-2-R-before-20mins-bank0_g0/cage2-2-R-before-20mins-bank0_g0_imec0/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load('/data/20230414/cage2-2-R-before-20mins-bank0_g0/cage2-2-R-before-20mins-bank0_g0_imec0/spike_times.npy')  #
channel = np.load('/data/20230414/cage2-2-R-before-20mins-bank0_g0/cage2-2-R-before-20mins-bank0_g0_imec0/channel_positions.npy')
n_spikes = identities[ identities == 11 ].size #统计该id的neuron的发放次数，对应phy中的n_spikes一列
average_firingrates = n_spikes/t  #对应phy中的fr一列
#print(channel)
#print(identities)
#print(average_firingrates)



#get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

# +-2 nuerons around selected id
def neurons_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty((6,len(y)))
    for m in range(-2,2):
        x = np.where(identities == id+m)
        y=x[0]
        for i in range(0,len(y)):
            z=y[i]
            spike_times[m,i]=times[z]/sample_rate
    return spike_times

#plot single neuron spike train
def raster_plot_singleneuron(spike_times):
    y=np.empty(len(spike_times))      
    plt.plot(spike_times,y, '|', color='gray') 
    plt.title('neuron 15') 
    plt.xlabel("time") 
    plt.xlim(0,10)
    plt.show()

#plot neurons around id spike train
def raster_plot_neurons(spike_times,id): 
    y = np.zeros((5, len(spike_times[0])))
    for i in range(0,5):
        y[i,:]=id+i
        plt.plot(spike_times[i] , y[i], '|', color='gray') 
    plt.title('spike train') 
    plt.xlabel("time")
    plt.ylabel("unit id")  
    plt.xlim(500,560)
    plt.ylim(id-1,id+5)
    plt.show()

#spike counts
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

#PSTH
def PSTH(id,m_start_time):
    # bin
    bin_width = 0.005
    duration = 0.05
    pre_time = -2*duration
    post_time = 2*duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)   

    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=m_start_time,
        )

    print(histograms)

    plt.figure()
    plt.plot(histograms.mean(1))

    plt.axvspan(0, duration, color='gray', alpha=0.1)
    plt.ylabel('Firing rate (spikes/second)')
    plt.xlabel('Time (s)')
    # plt.xlim(pre_time,post_time)
    if plt:
        plt.title('PSTH, Push rod')
        plt.xlabel('')
        plt.ylabel('')
    plt.show()

def PSTH2(id,m_start_time):
    # bin
    bin_width = 0.001
    duration = 0.05
    pre_time = -2*duration
    post_time = 2*duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)   

    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=m_start_time,
        )

    print(histograms)
    firing_rate=histograms.mean(1)/bin_width
    print(firing_rate)
    sio.savemat('/firing_rate/20230414/firing_rate_765.mat', {'firing_rate_765':firing_rate})
    
    plt.figure()
    plt.plot(firing_rate)

    plt.axvspan(0, duration, color='gray', alpha=0.1)
    plt.ylabel('Firing rate (spikes/second)')
    plt.xlabel('Time (s)')
    # plt.xlim(pre_time,post_time)
    if plt:
        plt.title('PSTH, Push rod')
        plt.xlabel('')
        plt.ylabel('')
    plt.show()
    
def FiringPattern_AmountofInfo(id,m_start_time):

    # bin
    bin_width = 0.001
    duration = 0.05
    pre_time = -2*duration
    post_time = 2*duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)   

    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=m_start_time,
        )
    data=histograms

    # statistics fixed t
    result_dic={}
    for j in range(0,len(data)):
        trial=data[j]  # get a trial-*
        for t in range(0,int(len(trial)/8-1),2):   # 隔一个word取一个
            array = np.array(trial[t*8:(t*8)+7])   # slice into 8 bits
            str1 = ''.join(str(i) for i in array)  # array to str
            if str1 not in result_dic:   # use dic to statistic, key=str, value=number of times
                result_dic[str1]=1
            else:
                result_dic[str1]+=1
    total=sum(result_dic.values())
    p={k: v / total for k, v in result_dic.items()}
    print(p)

    # information entropy
    h=0
    for i in p:
        h = h - p[i]*log(p[i],2)
    print('Shannon Entrophy=%f'%h)

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
            marker_raw = np.append(marker_raw,m/10600)
    #All start push times
    for j in range(1,len(marker_raw)):
        if marker_raw[j]-marker_raw[j-1]>4:
            marker=np.append(marker,marker_raw[j])
    #push rod time +-1s
    marker_3s=np.empty([len(marker),2], dtype = int) 
    for a in range(len(marker)):
        marker_3s[a,0]=int(marker[a]-1)
        marker_3s[a,1]=int(marker[a]+1)

    return marker


def test_20230414():

    with open('/marker/20230414/rod_marker.csv') as file_name:
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
            marker_raw = np.append(marker_raw,m/10593)
    #All start push times
    for j in range(1,len(marker_raw)):
        if marker_raw[j]-marker_raw[j-1]>4:
            marker=np.append(marker,marker_raw[j])
    #push rod time +-1s
    marker_1s=np.empty([len(marker),2], dtype = int) 
    for a in range(len(marker)):
        marker_1s[a,0]=int(marker[a]-0.5)
        marker_1s[a,1]=int(marker[a]+0.5)

    # bin
    bin_width = 0.005
    duration = 2   #一个trial的时间，或你关注的时间段的长度，这里是开始推杆的+-0.1s
    pre_time = -2*duration
    post_time = 2*duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)   

    histograms=spike_counts(
        singleneuron_spiketrain(765),
        bin_edges=bins,
        movement_start_time=marker,
        )

    print(histograms)

    firing_rate=histograms.mean(1)/bin_width
    '''
    if k == 0:
        temp=firing_rate
    else:
        temp=temp+firing_rate

    av_firing_rate=temp/len(marker_1s)
'''
    print(firing_rate)
    sio.savemat('/firing_rate/20230414/firing_rate_765.mat', {'firing_rate_765':firing_rate})
        
    plt.plot(firing_rate)

    plt.axvspan(0, duration, color='gray', alpha=0.1)
    plt.ylabel('Firing rate (spikes/second)')
    plt.xlabel('Time (s)')
    plt.xlim(-0.5,11)
    if plt:
        plt.title('PSTH, Push rod')
        plt.xlabel('')
        plt.ylabel('')
        
    plt.show()

def PETH_heatmap_2(data,id):  #已调试

    data_minus_mean=data-np.mean(data)

    # plot
    fig, ax = plt.subplots(figsize=(12, 12))
    div = make_axes_locatable(ax)
    cbar_axis = div.append_axes("right", 0.2, pad=0.05)
    img = ax.imshow(
        data_minus_mean, 
        extent=(-0.5,5,0,len(data)),  #前两个值是x轴的时间范围，后两个值是y轴的值
        interpolation='none',
        aspect='auto',
        vmin=0, 
        vmax=10 #热图深浅范围
    )
    plt.colorbar(img, cax=cbar_axis)

    cbar_axis.set_ylabel('Spike counts', fontsize=20)
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel('Trials', fontsize=20)
    reltime = np.arange(-0.5, 5.01, 0.2)
    ax.set_xticks(np.arange(-0.5, 5.01, 0.2))
    ax.set_xticklabels([f'{mp:1.3f}' for mp in reltime[::1]], rotation=45)
    ax.set_xlabel('Time(s) Move: 0s', fontsize=20)
    ax.set_title('Purkinje cell spike counts Neuron id %d'%id, fontsize=20)
    plt.show()

def firingrate_time(id,marker):

    # bin
    bin_width = 0.14
    duration = 5   #一个trial的时间，或你关注的时间段的长度
    pre_time = -0.5
    post_time = duration
    bins = np.arange(pre_time, post_time+bin_width, bin_width)  
    # histograms
    histograms=spike_counts(
        singleneuron_spiketrain(id),
        bin_edges=bins,
        movement_start_time=marker,
        )
    print(histograms)

    return histograms

PETH_heatmap_2(firingrate_time(955,marker()),955)
#PSTH2(273,marker())

#print(singleneuron_spiketrain(1196))
#FiringPattern_AmountofInfo(1196)
#raster_plot_neurons(neurons_spiketrain(1196),1196)
#raster_plot_singleneuron(singleneuron_spiketrain(1196))