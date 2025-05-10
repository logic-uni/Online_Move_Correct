"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 10/28/2024
data from: Xinrong Tan
data collected: 05/10/2023
"""
import math
import torch
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib.pyplot import *
from ast import literal_eval
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
from elephant import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numba
np.set_printoptions(threshold=np.inf)

### path
mice = '20230511'
main_path = r'E:\xinrong\mice_1\20230511\cage1-2-R-2_g0\cage1-2-R-2_g0_imec0'
save_path = r'C:\Users\zyh20\Desktop\Perturbation_analysis\manifold_perturbation\20230511'
fig_save_path = r'C:\Users\zyh20\Desktop\PC2\each_trial_pertu_start-10ms_to_pertu_end'

### marker
events = pd.read_csv(main_path+'/event_series.csv',index_col=0)
print(events)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
identities = np.load(main_path+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(main_path+'/spike_times.npy')  #
ch = pd.read_csv(main_path+'/neuropixels_site_area.csv')#防止弹出警告
cluster_info = pd.read_csv(main_path+'/cluster_info.tsv', sep='\t')#防止弹出警告
print(ch)
print(cluster_info)

# get single neuron spike train
def singleneuron_spiketimes(id):
    x = np.where(identities == id)
    y=x[0]
    #y = np.where(np.isin(identities, id))[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def popu_fr_onetrial(neuron_ids,marker_start,marker_end,fr_bin):   #开始推杆，到推杆结束的一个trial的population spike counts
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        one_neruon = fr.to_array().astype(int)[0]
        #print(one_neruon)
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    return neurons

def minkowski_distance(x, y, p):
    return np.sum(np.abs(x - y)**p)**(1/p)

def highD_dis(data):
    # 1. 获取第一个时刻的高维坐标点 (作为原点)
    origin = data[:, 0]

    # 2. 计算每个时刻的高维向量 (与第一个时刻的坐标点相减)
    vectors = data - origin[:, np.newaxis]

    # 3. 计算每个向量的模长 (欧几里得范数)
    #norms = np.linalg.norm(vectors, axis=0)
    #plt.plot(norms[1:], marker='o')
    # 存储每个时刻与第一个时刻的闵可夫斯基距离
    p = 2
    distances = [minkowski_distance(data[:, i], origin, p) for i in range(data.shape[1])]
    '''
    plt.plot(distances[1:])
    plt.axvline(19, color='red', linestyle='--')
    plt.axvline(49, color='blue', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Minkowski distance')
    plt.title('Change in High-Dimensional Vector Norm Over Time')
    plt.grid(True)
    plt.show()
    '''
    return distances[1:]

def plot_dis_mean(dis):
    dis_mean = np.mean(dis, axis=0)
    time = np.arange(-40, 205, 5)
    print(time)
    plt.plot(time,dis_mean)
    plt.axvline(0, color='red', linestyle='--')
    plt.axvline(150, color='blue', linestyle='--')
    plt.xlabel('Time(ms)')
    plt.ylabel('Minkowski distance')
    plt.title('High-Dimensional Vector Norm')
    plt.grid(True)
    plt.show()

def main(region):
    fr_bin = 5
    plt.figure()
    result = np.empty((0, 60))
    dis_trial = np.empty((0, 49))
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    neuron_id = np.array(neurons['cluster_id']).astype(int)
    print(neuron_id)
    for i in np.arange(0,len(events['push_on'])-1):
        trail_start = events['push_on'].iloc[i]
        trial_end = events['push_off'].iloc[i]
        pert_start = events['pert_on'].iloc[i]
        pert_end = events['pert_off'].iloc[i]
        reward_on = events['reward_on'].iloc[i]
        if pert_start != 0 and reward_on != 0 and pert_end > pert_start:
            #data = popu_fr_onetrial(neuron_id,pert_start,pert_end+0.001,fr_bin)
            #data = popu_fr_onetrial(neuron_id,pert_start-0.01,pert_end+0.001,fr_bin)
            #data = popu_fr_onetrial(neuron_id,pert_start-0.154,pert_end,fr_bin)
            #data = popu_fr_onetrial(neuron_id,pert_start-0.154,pert_start,fr_bin)
            data = popu_fr_onetrial(neuron_id,trail_start,trial_end,fr_bin)
            print(data.shape)
            trial_time = data.shape[1]
            pert_s = int((pert_start-trail_start)*trial_time/(trial_end-trail_start))
            pert_e = int((pert_end-trail_start)*trial_time/(trial_end-trail_start))
            interval = pert_e-pert_s
            if interval == 30:  ##30对应150ms       
                distance = highD_dis(data[:,(pert_s-10):(pert_e+10)])
                dis_trial = np.vstack((dis_trial, distance))

    plot_dis_mean(dis_trial)


region = 'Interposed nucleus'
main(region)
#Simple lobule
#Lobules IV-V
#Interposed nucleus