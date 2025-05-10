"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 05/08/2025
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
session_name = '20230510'
region = 'Lobules IV-V'   #Simple lobule  Lobule III  Lobules IV-V  Interposed nucleus  
eigv_condi = 'eigv_big1' # eigv_big1, eigv_0to1, eigv_big0
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
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("行为总时长")
print(events['trial_start'].iloc[-1])
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

def SVD_analysis(data):
    U, S, Vt = np.linalg.svd(data)
    #Plot_svd_spec(S, num, time_interval)
    return U, S, Vt

# ------- Eigen analysis -------
def eigen_comput(data):
    print(f'trunc_squa_mat: {data.shape}')
    eigenvalues, eigenvectors = np.linalg.eig(data)
    magnitudes = np.abs(eigenvalues)
    return eigenvalues,eigenvectors,magnitudes

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
    evalue1, evector1, mag1 = eigen_comput(data[:,0:time_interval])
    evalue2, evector2, mag2 = eigen_comput(data[:,time_interval:time_interval*2])
    evalue3, evector3, mag3 = eigen_comput(data[:,time_interval*2:time_interval*3]) 
    evalue4, evector4, mag4 = eigen_comput(data[:,time_interval*3:time_interval*4])
    #  -- concatenate --
    evectors_trial = np.array([evector1, evector2, evector3, evector4], dtype=np.complex64)
    evalues_trial = np.array([evalue1, evalue2, evalue3, evalue4], dtype=np.complex64)
    mags_trial = np.array([mag1, mag2, mag3, mag4], dtype=np.float32)
    # -- plot --
    #Plot_eigenvalue_spec(evalue1,evalue2,evalue3,evalue4,num,time_interval) 
    # -- filter those eigenvalue=0 --
    filt_evc1, filt_evc2, filt_evc3, filt_evc4 = filter_eva0(evalue1, evector1), filter_eva0(evalue2, evector2), filter_eva0(evalue3, evector3), filter_eva0(evalue4, evector4)
    return

# ------- CovarMat analysis -------
def cov_analysis(data):
    ## covarience matrix是对称阵，所以只有实数值的特征值
    ATA = np.dot(data.T, data)
    # 对 A^T A 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(ATA)
    # 绘制特征值的直方图
    plt.figure(figsize=(8, 6))
    plt.hist(eigenvalues, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Eigenvalues of $A^T A$')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    # 存储特征值最大的特征向量
    max_index = np.argmax(eigenvalues)  # 找到最大特征值的索引
    max_eigenvalue = eigenvalues[max_index]  # 最大特征值
    max_eigenvector = eigenvectors[:, max_index]  # 对应的特征向量
    return max_eigenvector

def save(values, vectors, type, region):
    evalues = np.array(values)   
    evectors = np.array(vectors)        
    np.save(save_path + f"/{region}_{type}values.npy", evalues)
    np.save(save_path + f"/{region}_{type}vectors.npy", evectors)

# ------- Main Function -------
def main():
    fr_bin = 1
    plt.figure(figsize=(10, 10))
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    if region not in ch['area_name'].values:
        print("region not exist")
        return
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    popu_id = np.array(neurons['cluster_id']).astype(int)
    neuron_num = len(popu_id)
    num = 1
    # initialize saving list
    Evalues_list, Evectors_list = [], []
    Singvalues_list, Singvectors_list = [], []
    CovEvalues_list, CovEvectors_list = [], []
    for i in np.arange(0,len(events['push_on'])-1):  # enumarate trials
        pert_start, pert_end, reward_on = events['pert_on'].iloc[i], events['pert_off'].iloc[i], events['reward_on'].iloc[i] # load marker unit s
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:   # pert_start != 0 means there's perturbation, reward_on != 0 means this is a success pushing trial, perturbation length is 0.14s~0.16s randomly
            # eigen analysis
            time_interval = neuron_num  #单位ms
            two_seg_bin_num = time_interval*2*fr_bin/1000  # pertur前后各两个方阵 #转换为s
            trunc1 = popu_fr_onetrial(popu_id,pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006,fr_bin)  # 滑动截断得到方阵 pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006单位是s
            evectors_trial, evalues_trial= eigen_analysis(trunc1, num, time_interval)
            Evectors_list.append(evectors_trial)
            Evalues_list.append(evalues_trial)
            # SVD analysis
            trunc2 = popu_fr_onetrial(popu_id,pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006,fr_bin)
            Singvectors_trial, Singvalues_trial = SVD_analysis(trunc2, num)
            Singvectors_list.append(Singvectors_trial)
            Singvalues_list.append(Singvalues_trial)
            # Cov analysis
            trunc3 = popu_fr_onetrial(popu_id,pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006,fr_bin)
            CovEvectors_trial, CovEvalues_trial = cov_analysis(trunc3, num)
            CovEvectors_list.append(CovEvectors_trial)
            CovEvalues_list.append(CovEvalues_trial)

            num += 1

    save(Evalues_list, Evectors_list, 'eigen', region)
    save(Singvalues_list, Singvectors_list, 'sing', region)
    save(CovEvalues_list, CovEvectors_list, 'cov', region)

main()