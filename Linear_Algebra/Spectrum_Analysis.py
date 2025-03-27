"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/16/2025
data from: Xinrong Tan
"""

import math
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib.pyplot import *
from ast import literal_eval
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from elephant.conversion import BinnedSpikeTrain
from elephant import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
#import numba
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
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
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    #y = np.where(np.isin(identities, id))[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def popu_fr_onetrial(neuron_ids,marker_start,marker_end,fr_bin):
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketrain(neuron_ids[j])
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

# ------- non-square matrix -------

def reduce_dimension(count,bin_size,n_components): # 默认: 0.1 感觉改bin_size影响不大，改firing rate的bin size影响较大
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

def covmatr_spectrum(data):
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

# ------- square matrix -------

def eigen(data):
    print(f'trunc_squa_mat: {data.shape}')
    eigenvalues, eigenvectors = np.linalg.eig(data)
    magnitudes = np.abs(eigenvalues)
    return eigenvalues,eigenvectors,magnitudes

def eigen_value_plot(eigenvalues,mode,marker,state):
    plt.scatter(eigenvalues.real, eigenvalues.imag, c=mode, marker=marker,s=100,label=state)   # 绘制特征值在复平面上的位置

def eigen_value_spectrum(evalue1,evalue2,evalue3,evalue4,num,time_interval):
    #画复平面
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
    eigen_value_plot(evalue1,'g','o','before pertur 1')
    eigen_value_plot(evalue2,'b','^','before pertur 2')
    eigen_value_plot(evalue3,'r','*','after pertur 1')
    eigen_value_plot(evalue4,'yellow','x','after pertur 2')
    plt.legend()
    plt.title(f'Spectrum Analysis')
    plt.savefig(save_path+f"/{region}_trial_{num}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()  #每个trial画完后清空图片'

def compare_stages(prev_stage, curr_stage, decimal_precision=4):
    """
    比较两个阶段中哪些行向量相同/不同
    :param prev_stage: 前一阶段的特征向量数组 (67x67)
    :param curr_stage: 当前阶段的特征向量数组 (67x67)
    :param decimal_precision: 浮点数精度（小数点后保留位数，None表示不处理）
    """
    # 将前一阶段的向量转换为元组并存入集合（用于快速查找）
    prev_vectors = set()
    for i in range(prev_stage.shape[0]):
        vector = prev_stage[i]
        if decimal_precision is not None:
            vector = np.round(vector, decimals=decimal_precision)
        prev_vectors.add(tuple(vector.flatten()))
    
    # 比较当前阶段的每个向量
    same_indices = []
    same_vectors = []  
    diff_indices = []
    diff_vectors = []

    for j in range(curr_stage.shape[0]):
        curr_vector = curr_stage[j]  #取出一个当前阶段的特征向量
        original_vector = curr_vector.copy()  # 保留原始向量（未四舍五入前）
        # 精度处理（仅用于比较）
        if decimal_precision is not None:
            curr_vector_rounded = np.round(curr_vector, decimals=decimal_precision) #四舍五入
            curr_tuple = tuple(curr_vector_rounded.flatten())
        else:
            curr_tuple = tuple(curr_vector.flatten()) 
        
        # 判断并存储
        if curr_tuple in prev_vectors:
            same_indices.append(j)   #记录相同特征向量所在的当前阶段的行索引
            same_vectors.append(original_vector)  # 存储未四舍五入的原始向量
        else:
            diff_indices.append(j)
            diff_vectors.append(original_vector)
        if curr_tuple in prev_vectors:
            same_indices.append(j)   #记录不同特征向量所在的当前阶段的行索引
        else:
            diff_indices.append(j)  
    
    # 转换为NumPy数组
    same_vectors = np.array(same_vectors) if same_vectors else np.empty((0, curr_stage.shape[1]))
    diff_vectors = np.array(diff_vectors) if diff_vectors else np.empty((0, curr_stage.shape[1]))
    
    return same_indices, same_vectors, diff_indices, diff_vectors

def plot_comparison_heatmap(num, same_vectors, diff_vectors, stage_name, vmin=None, vmax=None):
    #plt.figure(figsize=(15, 12))
    """
    绘制相同/不同向量的热图
    :param same_vectors: 相同向量数组 (n_same, 67)
    :param diff_vectors: 不同向量数组 (n_diff, 67)
    :param stage_name: 阶段名称（用于标题）
    :param vmin/vmax: 颜色范围统一用全局最小最大值
    """
    # 合并数据（相同在上，不同在下）
    data_to_plot = []
    if len(same_vectors) > 0:
        data_to_plot.append(same_vectors)
    if len(diff_vectors) > 0:
        data_to_plot.append(diff_vectors)
    
    if len(data_to_plot) == 0:
        print(f"{stage_name}无数据可绘制")
        return
    
    full_data = np.vstack(data_to_plot)
    n_same = same_vectors.shape[0]
    n_total = full_data.shape[0]
    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 6))
    # 绘制热图
    im = ax.imshow(full_data, aspect='auto', cmap='viridis', 
                  vmin=np.min(full_data) if vmin is None else vmin,
                  vmax=np.max(full_data) if vmax is None else vmax)
    # 添加分界线
    if n_same > 0 and n_same < n_total:
        ax.axhline(n_same - 0.5, color='white', linestyle='--', linewidth=2)
        ax.text(67/2, n_same/2, 'Same Vectors', 
                ha='center', va='center', color='white', fontsize=12)
        ax.text(67/2, n_same + (n_total - n_same)/2, 'Different Vectors',
                ha='center', va='center', color='white', fontsize=12)
    # 装饰图形
    ax.set_title(f"{stage_name} Comparison (Total: {n_total} vectors)", fontsize=14)
    ax.set_xlabel("Feature Dimension", fontsize=12)
    ax.set_ylabel("Vectors", fontsize=12)
    ax.set_xticks(np.arange(0, 67, 5))
    fig.colorbar(im, ax=ax, label='Feature Value')
    plt.tight_layout()
    plt.savefig(save_path + f"/{region}_trial_{num}_{stage_name}_eigen_heatmap.png")
    plt.clf()

def compare_eigen_stage(evector1,evector2,evector3,evector4,num):
    d2_diff, d3_diff, d4_diff = np.zeros_like(evector1), np.zeros_like(evector1), np.zeros_like(evector1)
    if evector1.shape[0] != 0 and evector2.shape[0] != 0: 
        s2_indices, s2_same, d2_indices, d2_diff = compare_stages(evector1, evector2)
    if evector2.shape[0] != 0 and evector3.shape[0] != 0: 
        s3_indices, s3_same, d3_indices, d3_diff = compare_stages(evector2, evector3)
    if evector3.shape[0] != 0 and evector4.shape[0] != 0: 
        s4_indices, s4_same, d4_indices, d4_diff = compare_stages(evector3, evector4)
    return d2_diff,d3_diff,d4_diff

def stout_eva2evc(mags,vectors,condi):
    if condi == 'eigv_big0':
        indices = np.flatnonzero(mags)
    elif condi == 'eigv_0to1':
        indices = np.flatnonzero(mags <= 1)
    elif condi == 'eigv_big1':
        indices = np.flatnonzero(mags > 1)
    else:
        raise ValueError(f"Invalid condition: {condi}. Expected 'eigv_big0', 'eigv_0to1', or 'eigv_big1'.")
    norm_biger1 = vectors[indices]
    if norm_biger1.dtype == 'complex': norm_biger1 = np.abs(norm_biger1)
    return norm_biger1

def filter_eva0(value,vector):
    if value.dtype != 'complex':
        indices = np.flatnonzero(value)
        filt = vector[indices]
    else:
        filt = vector
    return filt

def plot_matrix_sim_diff(data,content):
    # 1. 标准化数据
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    print(data_std.shape)
    # 2. 计算相似度矩阵和距离矩阵
    cosine_sim = cosine_similarity(data_std)
    euclidean_dist = euclidean_distances(data_std)

    # 3. 可视化热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cosine_sim, cmap='viridis')
    plt.title('Cosine Similarity Matrix')
    plt.savefig(save_path + f"/{region}_{content}_Cosine_Similarity_Matrix.png", dpi=200)
    plt.clf()

    # 4. 聚类热图
    sns.clustermap(cosine_sim, method='average', cmap='viridis', figsize=(12, 12))
    plt.title('Clustered Heatmap')
    plt.savefig(save_path + f"/{region}_{content}_Clustered_Heatmap.png", dpi=200)
    plt.clf()

    # 三维PCA降维 
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_std)  # 形状 (46, 3)
    explained_var_ratio = pca.explained_variance_ratio_ * 100      # 计算主成分解释方差比
    # 生成46种不同颜色（使用 'gist_ncar' 颜色映射，支持更多颜色）
    colors = cm.gist_ncar(np.linspace(0, 1, data_std.shape[0]))  # 形状 (46, 4)
    # 生成样本标签（假设标签为 'Sample_0' 到 'Sample_45'）
    labels = [f'Sample_{i}' for i in range(data_std.shape[0])]  # 替换为实际标签
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图（每个点不同颜色）
    scatter = ax.scatter(
        data_pca[:, 0],  # PC1
        data_pca[:, 1],  # PC2
        data_pca[:, 2],  # PC3
        s=60,            # 点大小
        alpha=0.8,       # 透明度
        c=colors,        # 颜色数组
        edgecolor='k',   # 边缘颜色
        linewidth=0.5    # 边缘线宽
    )

    # 为每个点添加文本标签
    for i in range(data_std.shape[0]):
        ax.text(
            data_pca[i, 0] + 0.05,  # x坐标微调避免重叠
            data_pca[i, 1] + 0.05,  # y坐标微调
            data_pca[i, 2] + 0.05,  # z坐标微调
            labels[i],
            fontsize=8,            # 字体大小
            color=colors[i],       # 文本颜色与点颜色一致
            ha='left',             # 水平对齐方式
            va='bottom'            # 垂直对齐方式
        )

    ax.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1f}%)', fontsize=12, labelpad=10)
    ax.set_ylabel(f'PC2 ({explained_var_ratio[1]:.1f}%)', fontsize=12, labelpad=10)
    ax.set_zlabel(f'PC3 ({explained_var_ratio[2]:.1f}%)', fontsize=12, labelpad=10)
    ax.set_title('3D PCA Projection with Labels', fontsize=16, pad=20)

    ax.view_init(elev=25, azim=30)
    plt.tight_layout()
    plt.savefig(save_path + f"/{region}_{content}_3D_PCA.png", dpi=300, bbox_inches='tight')
    plt.clf()

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
    time_interval = neuron_num  #单位ms
    print(f'each truncated time length: {time_interval} ms')
    # pertur前后各两个方阵
    two_seg_bin_num = time_interval*2*fr_bin/1000  
    num = 1
    evalues_list, evectors_list, mags_list = [], [], []
    stan_out_eigvc_list, s2_diff_list, s3_diff_list, s4_diff_list = [], [], [], []
    # -- enumarate trials --
    for i in np.arange(0,len(events['push_on'])-1):
        # -- marker --
        #trail_start = events['push_on'].iloc[i]   trial_end = events['push_off'].iloc[i]
        pert_start = events['pert_on'].iloc[i]
        pert_end = events['pert_off'].iloc[i]
        reward_on = events['reward_on'].iloc[i]
        # pert_start != 0 means there's perturbation in this trial
        # reward_on != 0 means there's a success pushing trial
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:
            # -- truncate data --
            data = popu_fr_onetrial(popu_id,pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006,fr_bin)  # pert 前0.218s 后0.218s
            print(f"neuron_fr_mat: {data.shape}")
            # -- compute each stage eigen vector and vector --
            evalue1, evector1, mag1 = eigen(data[:,0:time_interval])  # each evalue is a complex number
            evalue2, evector2, mag2 = eigen(data[:,time_interval:time_interval*2])
            evalue3, evector3, mag3 = eigen(data[:,time_interval*2:time_interval*3]) 
            evalue4, evector4, mag4 = eigen(data[:,time_interval*3:time_interval*4])
            # -- save eigen result --
            # 将每个 stage 的复数数组合并为一个二维数组 (4, 数组长度)
            evalues_trial = np.array([evalue1, evalue2, evalue3, evalue4], dtype=np.complex64)
            evectors_trial = np.array([evector1, evector2, evector3, evector4], dtype=np.complex64)
            # 合并 mag 标量为一个一维数组 (4,)
            mags_trial = np.array([mag1, mag2, mag3, mag4], dtype=np.float32)
            # 添加到列表
            evalues_list.append(evalues_trial)
            evectors_list.append(evectors_trial)
            mags_list.append(mags_trial)
            
            # eigen value spectrum
            eigen_value_spectrum(evalue1,evalue2,evalue3,evalue4,num,time_interval) 
            '''
            # compare eigen vector in different stage, filter those eigenvalue=0, stack trials different stage emerging eigen vector
            filt_evc1 = filter_eva0(evalue1, evector1)
            filt_evc2 = filter_eva0(evalue2, evector2)
            filt_evc3 = filter_eva0(evalue3, evector3)
            filt_evc4 = filter_eva0(evalue4, evector4)
            s2_diff, s3_diff, s4_diff = compare_eigen_stage(filt_evc1,filt_evc2,filt_evc3,filt_evc4,num)
            if not np.all(s2_diff == 0): s2_diff_list.append(s2_diff)
            if not np.all(s3_diff == 0): s3_diff_list.append(s3_diff)
            if not np.all(s4_diff == 0): s4_diff_list.append(s4_diff)
            
            # For those eigen value norm >= 1 in stage 3, collecting them
            norm_biger1 = stout_eva2evc(mag1,evector1,eigv_condi)  
            print(norm_biger1.shape)
            stan_out_eigvc_list.append(norm_biger1)
            '''
            num = num + 1
    '''
    evalues = np.array(evalues_list)    # 形状 (num_trials, 4, K)
    evectors = np.array(evectors_list)  # 同上
    mags = np.array(mags_list)          # 形状 (num_trials, 4)

    np.save(save_path + f"/{region}_evalues.npy", evalues)
    np.save(save_path + f"/{region}_evectors.npy", evectors)
    np.save(save_path + f"/{region}_mags.npy", mags)
    
    stan_out_eigvc = np.vstack(stan_out_eigvc_list)
    np.save(save_path + f"/{region}_{eigv_condi}_eigvc.npy", stan_out_eigvc)
    if stan_out_eigvc.shape[0] > 2: 
        plot_matrix_sim_diff(stan_out_eigvc,eigv_condi)  
    
    s2_diff_np = np.vstack(s2_diff_list)
    s3_diff_np = np.vstack(s3_diff_list)
    s4_diff_np = np.vstack(s4_diff_list)
    if s2_diff_np.shape[0] > 2: 
        plot_matrix_sim_diff(s2_diff_np,'stage2vs1')

    if s3_diff_np.shape[0] > 2: 
        plot_matrix_sim_diff(s3_diff_np,'stage3vs2')

    if s4_diff_np.shape[0] > 2: 
        plot_matrix_sim_diff(s4_diff_np,'stage4vs3')
    '''
main()

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