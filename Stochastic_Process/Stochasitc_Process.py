"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 11/15/2024
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
from scipy.stats import gaussian_kde
from scipy.stats import expon
import numba
np.set_printoptions(threshold=np.inf)

### path
mice = '20230511'
main_path = r'E:\xinrong\mice_1\20230511\cage1-2-R-2_g0\cage1-2-R-2_g0_imec0'
fig_save_path = r'C:\Users\zyh20\Desktop\Perturbation_analysis\Stochastic Process'

### marker
events = pd.read_csv(main_path+'/event_series.csv',index_col=0)
print(events)

### electrophysiology
sample_rate=30000 #spikeGLX neuropixel sample rate
identities = np.load(main_path+'/spike_clusters.npy') #存储neuron的编号id,对应phy中的第一列id
times = np.load(main_path+'/spike_times.npy')  
ch = pd.read_csv(main_path+'/neuropixels_site_area.csv')#防止弹出警告
cluster_info = pd.read_csv(main_path+'/cluster_info.tsv', sep='\t')#防止弹出警告
print(ch)
print("电生理总时长")
print((times[-1]/sample_rate)[0])
print("行为总时长")
print(events['trial_start'].iloc[-1])
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

def neuron_fr_trials(neuron_id,fr_bin,mode):   #开始推杆，到推杆结束的一个trial的population spike counts
    neruon_trials = np.array([])
    a=0
    for j in range(len(events['pert_on'])): #第j个trial
        # marker series
        reset_on = events['reset_on'].iloc[j]
        reset_off = events['reset_off'].iloc[j]
        push_on = events['push_on'].iloc[j]
        pert_on = events['pert_on'].iloc[j]
        pert_off = events['pert_off'].iloc[j]
        push_off = events['push_off'].iloc[j]
        reward_on = events['reward_on'].iloc[j]
        
        # select success trials
        if pert_on != 0 and reward_on != 0 and pert_off > pert_on:
            # total
            spike_times = singleneuron_spiketimes(neuron_id)
            # trancation
            if mode == 'reset':
                ## during reset
                t1 = reset_on 
                t2 = reset_on+2.8  # unit sec
            elif mode == 'pertur':
                ## during perturbation
                t1 = pert_on-0.5  # unit sec
                t2 = pert_off+0.5  # unit sec
            spike_times_trail = spike_times[(spike_times > t1) & (spike_times < t2)]   
            spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start = t1, t_stop = t2)
            # binned
            fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
            neruon_trial = fr.to_array().astype(int)[0]
            if a == 0:
                neruon_trials = neruon_trial
            else:
                neruon_trials = np.vstack((neruon_trials, neruon_trial))

            a=a+1
        fi_rate_trials = neruon_trials*1000/fr_bin # neruon_trials是15ms内计数值 除以time bin 15ms 得到spike/s的单位

    return fi_rate_trials  #数组长度为总时长/bin长度，2800/15 = 186个

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

def Stoch_P_trials(matrices,num_columns,num_matrices):
    print(matrices.shape)

    # 初始化存储结果的数组
    results = []

    # 统计每一列的各个值出现的概率
    for col in range(num_columns):
        col_probabilities = []
        for mat_idx in range(num_matrices):
            values, counts = np.unique(matrices[mat_idx, :, col], return_counts=True)
            probabilities = counts / counts.sum()  # 计算概率
            col_probabilities.append((values, probabilities))
        results.append(col_probabilities)

    # 绘制三维图
    for col in range(num_columns):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 准备数据
        matrices_indices = np.arange(num_matrices)  # 矩阵索引
        x_values = []
        y_values = []
        z_values = []
        
        for mat_idx in range(num_matrices):
            values, probabilities = results[col][mat_idx]
            x_values.extend([mat_idx] * len(values))
            y_values.extend(values)
            z_values.extend(probabilities)
        
        ax.scatter(x_values, y_values, z_values, marker='o')

        ax.set_title(f'Value Probability Distribution at Time {col}')
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Firing rate(spike/s)')
        ax.set_zlabel('Probability')
        plt.show()

def Stoch_P_one_neuron_fr_trial(data,id,region,fr_bin,mode):
    print(data.shape)
    data_length = data.shape[1]
    trial_num = data.shape[0]
    # 设置3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 每一列的直方图
    for col in range(data_length):
        # 计算直方图
        hist, bins = np.histogram(data[:, col], bins=10, range=(0, 500),density=False)  # 最多10个bar统计，i.o.w. 每个时刻点的fr最多有10种状态，fr范围[0,500]spike/s
        # 计算条形的中心
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # 绘制每个切片
        hist_normalized = hist / trial_num  #归一化为 [0, 1] 之间的概率 
        colors = ['blue', 'green', 'red', 'cyan']
        ax.bar(bin_centers, hist_normalized, zs=col*fr_bin, zdir='y', width=20, alpha=0.8,color=colors)

    # 设置轴标签
    ax.set_xlabel('Firing rate (spike/s)')
    ax.set_ylabel('Time (ms)')
    ax.set_zlabel('Probability')
    ax.set_title(f'Neuron{id}_{region}_Each time firing rate distribution_(Stochastic Process View)')
    ax.set_yticks(np.arange(0,data_length*16,500))
    # 画marker线
    x = np.linspace(0, 500)  #500 is firing rate range
    z = np.zeros_like(x)  # z=0
    if mode == 'reset':
        ax.plot(x, [0]*len(x), z, color='cyan', lw=2, label="reset on")
        ax.plot(x, [2000]*len(x), z, color='blue', lw=2, label="reset off")
        ax.plot(x, [2500]*len(x), z, color='red', lw=2, label="push on")
    elif mode == 'pertur':
        ax.plot(x, [500]*len(x), z, color='red', lw=2, label="pert on")
        ax.plot(x, [650]*len(x), z, color='blue', lw=2, label="pert off")
    plt.legend()
    plt.savefig(fig_save_path+f"/during_pertur/Simple lobule/Neuron{id}_{region}_Each time firing rate distribution_(Stochastic Process View).jpg",dpi=600)

def Stoch_P_one_neuron_distribuplot(data,num_cols):
    num_cols = 10   # 时刻数

    # 初始化存储概率密度的数组
    x = np.linspace(0, 500, 1000)  # 定义x轴范围
    density_values = np.zeros((num_cols, len(x)))

    # 计算每一列的概率密度
    for col in range(num_cols):
        kde = gaussian_kde(data[:, col])  # 计算核密度估计
        density_values[col] = kde(x)  # 评估密度函数

    # 准备绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 创建每个切片
    y = np.arange(num_cols)  # 每一列的索引

    # 绘制切片
    for col in range(num_cols):
        ax.plot(x, y[col] * np.ones_like(x), density_values[col], alpha=0.7)

    # 设置坐标轴标签
    ax.set_xlabel('Value')
    ax.set_ylabel('Columns (Time Moments)')
    ax.set_zlabel('Probability Density')
    ax.set_title('3D Density Slices for Each Column')

    # 设置视角
    ax.view_init(elev=20, azim=30)

    plt.show()

def main(region):
    all_matrices=[]
    fr_bin = 15 # unit ms  #时间窗口应较小，才可以捕捉高发放率编码的时刻的状态
    # get neuron ids
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    neuron_ids = np.array(neurons['cluster_id']).astype(int)
    print(neuron_ids)
    '''
    # get neuron_fr trials matrix
    for neuron_id in np.arange(0,len(neuron_ids)):
        data = neuron_fr_trials(neuron_id,fr_bin)
        print(data.shape)
        all_matrices.append(data)

    all_matrices = np.stack(all_matrices, axis=0)
    print(all_matrices.shape)
    Stoch_P_trials(all_matrices,20,len(neuron_ids))
    '''
    # get each neuron_fr trials matrix
    mode='pertur'  # reset or pertur
    for neuron_id in neuron_ids:
        data = neuron_fr_trials(neuron_id,fr_bin,mode)
        print(neuron_id)
        Stoch_P_one_neuron_fr_trial(data,neuron_id,region,fr_bin,mode)
    
region = 'Simple lobule'
main(region)

#Simple lobule
#Lobules IV-V
#Interposed nucleus