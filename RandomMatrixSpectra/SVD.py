"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 05/10/2025
data from: Xinrong Tan
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from ast import literal_eval
from elephant.conversion import BinnedSpikeTrain
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
mice = 'mice_1'  # 10-13 mice_1, 14-17 mice_2, 18-22 mice_3
session_name = '20230510'
region = 'Interposed nucleus'   #Simple lobule  Lobule III  Lobules IV-V  Interposed nucleus  
avoid_spikemore1 = True # 避免1ms的bin里有多个spike,对于1ms内多个spike的，强行置1

# ------- NO NEED CHANGE -------
### path
main_path = f'/data1/zhangyuhao/xinrong_data/NP1/{mice}/{session_name}/Sorted'
sorted_path = main_path + '/xinrong_sorted'
save_path = f'/home/zhangyuhao/Desktop/Result/Online_Move_Correct/Spectrum_analysis/{session_name}'
events = pd.read_csv(main_path+'/event_series.csv',index_col=0)
print(events)
### electrophysiology
sample_rate = 30000 #spikeGLX neuropixel sample rate
identities = np.load(sorted_path+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(sorted_path+'/spike_times.npy')  #
ch = pd.read_csv(sorted_path + '/neuropixels_site_area.csv')
cluster_info = pd.read_csv(sorted_path+'/cluster_info.tsv', sep='\t')
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

def popu_fr_onetrial(neuron_ids,marker_start,marker_end,fr_bin):  ## marker_start,marker_end单位是s
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        if avoid_spikemore1 == False:
            one_neruon = fr.to_array().astype(int)[0]
        else:
            fr_binar = fr.binarize()  # 对于可能在1ms内出现两个spike的情况，强制置为该bin下即1ms只能有一个spike
            one_neruon = fr_binar.to_array().astype(int)[0]
        
        #print(one_neruon)
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    return neurons

def Plot_svd_spec(S, num, time_interval):
    plt.figure(figsize=(8, 5))
    plt.plot(S, marker='o', linestyle='-', color='b')
    plt.title('Singular Values from SVD')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.show()
    plt.savefig(save_path+f"/{region}_trial_{num}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

# ------- Main Function -------
def main():
    fr_bin = 1
    plt.figure(figsize=(10, 10))

    # select region neurons
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    if region not in ch['area_name'].values:
        print("region not exist")
        return
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    popu_id = np.array(neurons['cluster_id']).astype(int)
    num_neurons = len(popu_id)

    # get each trial truncated data
    num_trials = 0
    trunc_data_trials = []
    for trial in np.arange(0,len(events['push_on'])-1):  # compute successful num_trials
        pert_start, pert_end, reward_on = events['pert_on'].iloc[trial], events['pert_off'].iloc[trial], events['reward_on'].iloc[trial] # load marker unit s
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:   # pert_start != 0 means there's perturbation, reward_on != 0 means this is a success pushing trial, perturbation length is 0.14s~0.16s randomly
            trunc2 = popu_fr_onetrial(popu_id,pert_start-0.1,pert_start+0.2+0.0006,fr_bin)  #单位是s  perturbation为零点，截取-100ms到200ms
            trunc_data_trials.append(trunc2)
            num_trials += 1

    trunc_data_trials_npy = np.array(trunc_data_trials)
    # compute SVD
    token_size = 20  # 20（20ms）
    num_splits = int(trunc_data_trials_npy[0].shape[1] / token_size)   # 300列分割为15个块，每个20列
    m, n = num_neurons, token_size   # 每个分割块的形状  长度为token_size，宽度为neurons number
    U_all = np.zeros((num_trials, num_splits, m, n))  # full_matrices=False时U的形状
    S_all = np.zeros((num_trials, num_splits, n))     # 奇异值向量
    Vh_all = np.zeros((num_trials, num_splits, n, n)) # V的共轭转置

    for i in range(num_trials):
        arr = trunc_data_trials_npy[i]
        for j in range(num_splits):
            # 分割块
            start_col = j * token_size
            block = arr[:, start_col:start_col + token_size]
            # 计算SVD，full_matrices=False减少存储空间
            U, S, Vh = np.linalg.svd(block, full_matrices=False)
            # 存储结果
            U_all[i, j] = U
            S_all[i, j] = S
            Vh_all[i, j] = Vh

    np.save(save_path + f"/{region}_U.npy", U_all)
    np.save(save_path + f"/{region}_S.npy", S_all)
    np.save(save_path + f"/{region}_Vh.npy", Vh_all)

main()