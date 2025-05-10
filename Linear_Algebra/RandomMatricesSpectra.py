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
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import cm
np.set_printoptions(threshold=np.inf)

# ------- NEED CHANGE -------
mice = 'mice_1'  # 10-13 mice_1, 14-17 mice_2, 18-22 mice_3
session_name = '20230512'
region = 'Lobules IV-V'   #Simple lobule  Lobule III  Lobules IV-V  Interposed nucleus  
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

# ------- spike counts -------
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

# ------- SVD analysis -------
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
    
def SVD_comput(truncdata):
    U, S, Vt = np.linalg.svd(truncdata)  # 加上full_matrices=False 仅保留非零奇异值对应的向量
    return U, S, Vt

def SVD_analysis(data, num, time_interval=20):
    n_samples, n_features = data.shape
    print(n_features)
    num_segments = n_features // time_interval  # 计算分段数量
    results = []
    for i in range(num_segments):
        # 截取当前时间段的列数据
        start = i * time_interval
        end = (i+1) * time_interval
        truncdata = data[:, start:end]
        # 计算当前分块的SVD
        U, S, Vt = SVD_comput(truncdata)
        # 将结果打包为元组并保存
        results.append( (U, S, Vt) )

    return results

# ------- Eigen analysis -------
def eigen_comput(data):
    print(f'trunc_squa_mat: {data.shape}')
    eigenvalues, eigenvectors = np.linalg.eig(data)
    return eigenvalues,eigenvectors

def Plot_eigenvalue_spec(evalue1,evalue2,evalue3,evalue4,num,time_interval):
    plt.xlabel('Real')
    plt.xlim(-3.5,3.5)
    plt.ylim(-3.5,3.5)
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.gca().set_aspect('equal')
    # 画半径为1的圆
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 1 * np.cos(theta)
    y = 1 * np.sin(theta)
    plt.plot(x, y,c='black')
    text = plt.text(0.05, 0.95, f'Trial {num}\n\n{region}\n\nEach Truncated Time Length: {time_interval} ms', 
                    ha='left', va='top', transform=plt.gca() .transAxes, fontsize=10)
    def eigen_value_plot(eigenvalues,mode,marker,state):
        plt.scatter(eigenvalues.real, eigenvalues.imag, c=mode, marker=marker,s=100,label=state)   # 绘制特征值在复平面上的位置
    eigen_value_plot(evalue1,'g','o','before pertur 1')
    eigen_value_plot(evalue2,'b','^','before pertur 2')
    eigen_value_plot(evalue3,'r','*','after pertur 1')
    eigen_value_plot(evalue4,'yellow','x','after pertur 2')
    plt.legend()
    plt.title(f'Spectrum Analysis')
    plt.savefig(save_path+f"/{region}_trial_{num}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()

def filter_eva0(value,vector):
    if value.dtype != 'complex':
        indices = np.flatnonzero(value)
        filt = vector[indices]
    else:
        filt = vector
    return filt

def eigen_analysis(data, num, time_interval):
    # -- compute each stage eigen vector and vector --
    evalue1, evector1 = eigen_comput(data[:,0:time_interval])
    evalue2, evector2 = eigen_comput(data[:,time_interval:time_interval*2])
    evalue3, evector3 = eigen_comput(data[:,time_interval*2:time_interval*3]) 
    evalue4, evector4 = eigen_comput(data[:,time_interval*3:time_interval*4])
    #  -- concatenate --
    evectors_trial = np.array([evector1, evector2, evector3, evector4], dtype=np.complex64)
    evalues_trial = np.array([evalue1, evalue2, evalue3, evalue4], dtype=np.complex64)
    # -- plot --
    #Plot_eigenvalue_spec(evalue1,evalue2,evalue3,evalue4,num,time_interval) 
    # -- filter those eigenvalue=0 --
    #filt_evc1, filt_evc2, filt_evc3, filt_evc4 = filter_eva0(evalue1, evector1), filter_eva0(evalue2, evector2), filter_eva0(evalue3, evector3), filter_eva0(evalue4, evector4)
    return evectors_trial, evalues_trial

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
    # Parameters for truncated square matrix
    time_interval = len(popu_id)  #单位ms
    two_seg_bin_num = time_interval*2*fr_bin/1000  # pertur前后各两个方阵 #转换为s

    num = 1
    # initialize saving list
    #Evalues_list, Evectors_list = [], []
    svd_results = []
    for i in np.arange(0,len(events['push_on'])-1):  # enumarate trials
        pert_start, pert_end, reward_on = events['pert_on'].iloc[i], events['pert_off'].iloc[i], events['reward_on'].iloc[i] # load marker unit s
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:   # pert_start != 0 means there's perturbation, reward_on != 0 means this is a success pushing trial, perturbation length is 0.14s~0.16s randomly
            # eigen analysis
            '''
            trunc1 = popu_fr_onetrial(popu_id,pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006,fr_bin)  # 滑动截断得到方阵 pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006单位是s
            evectors_trial, evalues_trial= eigen_analysis(trunc1, num, time_interval)
            Evectors_list.append(evectors_trial)
            Evalues_list.append(evalues_trial)
            '''
            # SVD analysis
            trunc2 = popu_fr_onetrial(popu_id,pert_start-0.1,pert_start+0.2+0.0006,fr_bin)  #单位是s  perturbation为零点，截取-100ms到200ms
            svd_reslut_trial = SVD_analysis(trunc2, num)
            svd_results.append(svd_reslut_trial)

            num += 1

    #evalues, evectors = np.array(Evalues_list), np.array(Evectors_list)  
    #np.save(save_path + f"/{region}_evalues.npy", evalues)
    #np.save(save_path + f"/{region}_evectors.npy", evectors)
    np.save(save_path + f"/{region}_svdresults.npy", svd_results)

main()