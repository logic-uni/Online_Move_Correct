"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 11/12/2024
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
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("行为总时长")
print(events['trial_start'].iloc[-1])
print(cluster_info)

# get single neuron spike train
def singleneuron_spiketrain(id):
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
        spike_times = singleneuron_spiketrain(neuron_ids[j])
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

def reduce_dimension(count,bin_size,region_name,n_components): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
    #smooth data
    count = pd.DataFrame(count)
    rate = np.sqrt(count/bin_size)
    #对数据做均值  默认: window=50  min_periods=1  感觉改这些值影响不大，改firing的bin size影响较大
    rate = rate.rolling(window=50,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2) 
    #reduce dimension
    
    pca = PCA(n_components)
    X_pca = pca.fit_transform(rate.values)   #对应的是Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_   #每个主成分所解释的方差比例
    explained_variance_sum = np.cumsum(explained_variance_ratio)  #计算累积解释方差比例
    
    #X_isomap = Isomap(n_components = 3, n_neighbors = 21).fit_transform(rate.values)  #对应的是Residual variance
    #X_tsne = TSNE(n_components=3,random_state=21,perplexity=20).fit_transform(rate.values)  #t-SNE没有Explained variance，t-SNE 旨在保留局部结构而不是全局方差
    return X_pca

def SVD_matrix_spectrum(data):
    # 1. 进行奇异值分解
    U, S, Vt = np.linalg.svd(data)

    # 2. 绘制奇异值的分布
    plt.figure(figsize=(8, 5))
    plt.plot(S, marker='o', linestyle='-', color='b')
    plt.title('Singular Values from SVD')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True)
    plt.show()

    # 3. 计算最大奇异值对应的特征向量
    max_singular_value_index = np.argmax(S)
    max_singular_value_vector = U[:, max_singular_value_index]

    # 输出最大奇异值及其对应的特征向量
    print(f'Max Singular Value: {S[max_singular_value_index]}')
    print(f'Corresponding Feature Vector (Left Singular Vector): {max_singular_value_vector}')

def redu_eign(data):
    print(data.shape)
    data2pca=data.T
    redu_dim_data=reduce_dimension(data2pca,0.1,region,30)
    ## 这里的矩阵是人为确定过保证Neuron和time两个轴的维度相同，对这样截断的矩阵做谱分析，因此存在虚数
    print(redu_dim_data.shape)
    # 对非对称矩阵进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(redu_dim_data)
    return eigenvalues

def sliding_spectrum(data,mode,marker):  #mode='b' or 'r'
    print(data.shape)
    eigenvalues, eigenvectors = np.linalg.eig(data)
    # 绘制特征值在复平面上的位置
    plt.scatter(eigenvalues.real, eigenvalues.imag, c=mode, marker=marker,s=100)
    plt.pause(1)

def covmatr_spectrum(data):
    ## covarience matrix是对称阵，所以只有实数值的特征值
    ATA = np.dot(data.T, data)
    # 对 A^T A 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(ATA)

    # 绘制特征值的直方图
    plt.figure(figsize=(8, 6))
    plt.hist(eigenvalues, bins=30, color='blue', edgecolor='black', alpha=0.7)

    # 设置标题和标签
    plt.title('Histogram of Eigenvalues of $A^T A$')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')

    # 显示图像
    plt.grid(True)
    plt.show()

    # 存储特征值最大的特征向量
    max_index = np.argmax(eigenvalues)  # 找到最大特征值的索引
    max_eigenvalue = eigenvalues[max_index]  # 最大特征值
    max_eigenvector = eigenvectors[:, max_index]  # 对应的特征向量
    return max_eigenvector

def eigen_vector_included_angle(A,B):
    # 将矩阵展平为一维数组
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # 计算点积和模长
    dot_product = np.dot(A_flat, B_flat)
    norm_A = np.linalg.norm(A_flat)
    norm_B = np.linalg.norm(B_flat)

    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B)
    
    print(similarity)

def main(region):
    fr_bin = 1
    plt.figure()
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.xlabel('Real')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.ylabel('Imaginary')
    plt.grid(True)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 1 * np.cos(theta)
    y = 1 * np.sin(theta)
    plt.plot(x, y)

    result = np.empty((0, 60))
    dis_trial = np.empty((0, 49))
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    neuron_id = np.array(neurons['cluster_id']).astype(int)
    neuron_num = len(neuron_id)
    two_seg_bin_num = neuron_num*2*fr_bin/1000
    print(two_seg_bin_num)
    plt.title(f'Dynamic Spectrum of Random Matrix_time_interval={neuron_num}ms')
    num = 1
    for i in np.arange(0,len(events['push_on'])-1):
        trail_start = events['push_on'].iloc[i]
        trial_end = events['push_off'].iloc[i]
        pert_start = events['pert_on'].iloc[i]
        pert_end = events['pert_off'].iloc[i]
        reward_on = events['reward_on'].iloc[i]
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:
            data = popu_fr_onetrial(neuron_id,pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006,fr_bin)  # pert 前0.218s 后0.218s
            print(data.shape)
            text = plt.text(0.95, 0.95, f'trail {num}', ha='right', va='top', transform=plt.gca().transAxes,fontsize=25)
            sliding_spectrum(data[:,0:neuron_num],'g','o') 
            sliding_spectrum(data[:,neuron_num:neuron_num*2],'b','+') 
            sliding_spectrum(data[:,neuron_num*2:neuron_num*3],'r','*') 
            sliding_spectrum(data[:,neuron_num*3:neuron_num*4],'cyan','x')
            plt.pause(2)
            text.remove()
            num = num +1

region = 'Interposed nucleus'
main(region)
#Simple lobule
#Lobules IV-V
#Interposed nucleus

#SVD_matrix_spectrum(data[:,(pert_s):(pert_s+40)])
#max_eigen1 = covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
#max_eigen2 = covmatr_spectrum(data[:,(pert_s):(pert_s+40)])
#eigen_vector_included_angle(max_eigen1,max_eigen2)
#covmatr_spectrum(data[:,(pert_s-40):(pert_s)])
#covmatr_spectrum(data[:,(pert_s-40):(pert_e+150)])

#data = popu_fr_onetrial(neuron_id,pert_start,pert_end+0.001,fr_bin)
#data = popu_fr_onetrial(neuron_id,pert_start-0.01,pert_end+0.001,fr_bin)
#data = popu_fr_onetrial(neuron_id,pert_start-0.154,pert_end,fr_bin)
#data = popu_fr_onetrial(neuron_id,pert_start-0.154,pert_start,fr_bin)