"""
# coding: utf-8
@author: Yuhao Zhang
last updated : 10/28/2024
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

def plot_average_std(data):
    # 计算每个时间点的平均值和标准差
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)

    # 创建时间轴，假设时间间隔为1单位
    time = np.arange(data.shape[1])

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制平均值曲线（实线）
    plt.plot(time, mean_values, label='Mean', color='blue')

    # 使用 fill_between 绘制标准差的范围（浅色阴影）
    plt.fill_between(time, mean_values - std_values, mean_values + std_values, 
                    color='blue', alpha=0.3, label='Standard Deviation Range')

    # 图例和标签
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Mean and Standard Deviation Over Time')
    plt.legend()

def ol_index(oscillating_data):
    window_size = 5
    result = np.empty((0, 30))
    t = np.arange(0, len(oscillating_data))
    # 计算滑动窗口内的标准差来衡量波动性
    rolling_std = pd.Series(oscillating_data).rolling(window=window_size).std()
    # 计算滑动窗口内的极差
    rolling_range = pd.Series(oscillating_data).rolling(window=window_size).apply(lambda x: np.max(x) - np.min(x))
    '''
    # 标准差收敛图
    #plt.plot(t, rolling_std, color='green')
    plt.plot(t, rolling_range, color='red')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    '''
    return rolling_range

def plot_flutuate(data):
    print(data.shape[0])
    print(data.shape[1])
    trials = data.shape[0]  # 实验次数
    time_points = data.shape[1]  # 每次实验的时间点数量
    time = np.linspace(0, 300, time_points)  # 时间
    # 创建网格
    X, Y = np.meshgrid(time, np.arange(trials))

    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 选择 plot_surface 或 plot_wireframe
    ax.plot_surface(X, Y, data, cmap='viridis')

    # 设置标签
    ax.set_xlabel('Time')
    ax.set_ylabel('Trial')
    ax.set_zlabel('Value')

def ploton_one_line(trajectory):
    # 首点和尾点
    start_point = trajectory[0]
    end_point = trajectory[-1]

    # 1. 平移，使首点为原点
    translated_trajectory = trajectory - start_point

    # 2. 计算旋转角度
    dx, dy = end_point - start_point
    theta = np.arctan2(dy, dx)  # atan2 计算角度

    # 3. 生成旋转矩阵
    rotation_matrix = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta), np.cos(-theta)]
    ])

    # 4. 旋转轨迹
    rotated_trajectory = translated_trajectory @ rotation_matrix.T

    # 5. 可视化结果
    #plt.plot(trajectory[:, 0], trajectory[:, 1], 'o-', label='Original trajectory')
    plt.plot(rotated_trajectory[:, 0], rotated_trajectory[:, 1])
    #plt.axhline(0, color='gray', linestyle='--')
    #plt.axvline(0, color='gray', linestyle='--')
    #plt.legend()
    plt.title('Trajectory Transformation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

def thrD_manifold(redu_dim_data,fr_bin,pert_start,pert_end,per_s,per_e):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    # 仅perturbation期间
    ax.set_title(f"manifold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.plot3D(redu_dim_data[:,0],redu_dim_data[:,1],redu_dim_data[:,2],'red')

    ax.plot3D(redu_dim_data[0:per_s+1,0],redu_dim_data[0:per_s+1,1],redu_dim_data[0:per_s+1,2],'yellow')
    ax.plot3D(redu_dim_data[per_s:per_e,0],redu_dim_data[per_s:per_e,1],redu_dim_data[per_s:per_e,2],'red')
    ax.plot3D(redu_dim_data[per_e-1:,0],redu_dim_data[per_e-1:,1],redu_dim_data[per_e-1:,2],'blue')
    plt.show()
    #干扰前0.2到干扰后0.2
    befor_per = int(0.2*1000/fr_bin)
    after_per = int((pert_end - pert_start + 0.2) * 1000/fr_bin)
    ax.plot3D(redu_dim_data[0:befor_per,0],redu_dim_data[0:befor_per,1],redu_dim_data[0:befor_per,2],'yellow')
    ax.plot3D(redu_dim_data[befor_per:after_per,0],redu_dim_data[befor_per:after_per,1],redu_dim_data[befor_per:after_per,2],'red')
    ax.plot3D(redu_dim_data[after_per:,0],redu_dim_data[after_per:,1],redu_dim_data[after_per:,2],'blue')
    plt.show()

def twoD_manifold(redu_dim_data,per_s,per_e):
    plt.figure()
    plt.plot(redu_dim_data[0,0],redu_dim_data[0,1],'o')
    plt.plot(redu_dim_data[0:per_s+1,0],redu_dim_data[0:per_s+1,1],'green')
    plt.plot(redu_dim_data[per_s,0],redu_dim_data[per_s,1],'o')
    plt.plot(redu_dim_data[per_s:per_e,0],redu_dim_data[per_s:per_e,1],'red')
    plt.plot(redu_dim_data[per_e-1,0],redu_dim_data[per_e-1,1],'o')
    plt.plot(redu_dim_data[per_e-1:,0],redu_dim_data[per_e-1:,1],'blue')
    plt.plot(redu_dim_data[-1,0],redu_dim_data[-1,1],'o')
    # 仅perturbation期间
    plt.plot(redu_dim_data[0,0],redu_dim_data[0,1],'o')
    plt.plot(redu_dim_data[:,0],redu_dim_data[:,1])

def oneD_manifold(redu_dim_data):
    #mean = np.mean(redu_dim_data[:,0])
    #plt.plot(redu_dim_data[:,0])
    #temp = ol_index(redu_dim_data[:,1])
    #result = np.vstack((result, temp))  
    #result = np.vstack((result, redu_dim_data[:,1]))

    #plt.plot(redu_dim_data[:,1])
    #plt.savefig(fig_save_path+f"/trail{i}_PC2_manifold.png",dpi=600,bbox_inches = 'tight')
    #plt.clf()
    ## 放到一条线上画出
    arr = redu_dim_data[:,0]
    indices = np.arange(1, len(arr) + 1)
    result = np.column_stack((indices, arr))
    ploton_one_line(result)

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
            # 开始到结束
            per_s = int((pert_start - trail_start) * 1000/fr_bin)
            per_e = int((pert_end - trail_start) * 1000/fr_bin)
        
            '''
            data2pca=data.T
            redu_dim_data=reduce_dimension(data2pca,0.1,region)
            print(redu_dim_data.shape)
            
            ### 3D manifold
            thrD_manifold(redu_dim_data,fr_bin,pert_start,pert_end,per_s,per_e)
            
            ### 2D manifold
            twoD_manifold(redu_dim_data,per_s,per_e)
            
            # plot 1D manifold
            oneD_manifold(redu_dim_data)
            
    #plot_average_std(result)
    #plot_flutuate(result)
    '''
    plt.show()

    
region = 'Interposed nucleus'
main(region)

#Simple lobule
#Lobules IV-V
#Interposed nucleus