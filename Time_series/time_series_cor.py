"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/29/2025
data from: Xinrong Tan
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
np.set_printoptions(threshold=np.inf)

# 对一个区域的所有neuron每个trial的spike times矩阵进行time series correlation analysis
# 相似度计算允许事件序列在时间轴上平移。例如，若序列A的事件在序列B的事件之后固定时间出现，算法会找到最佳平移量使得两序列的事件重叠最多
# 非对角线上的亮色块表示两序列在某个时间位移下事件高度重叠。例如，若第2行第5列高亮，说明序列2与5有相似的时序模式。
# 该方法对于长短序列差很大的情况存在bias，因此仅对有发放的neuron进行计算
# ------- NEED CHANGE -------
mice = 'mice_1'  # 10-13 mice_1, 14-17 mice_2, 18-22 mice_3
session_name = '20230511'
region = 'Lobules IV-V'   #Simple lobule  Lobule III  Lobules IV-V  Interposed nucleus  
mode = 0  # mode 0 after per 0~100ms,mode 1 after per 100~200ms,mode -1 before per -100~0ms,
trunc_interval = 50  #单位ms

# ------- NO NEED CHANGE -------
### path
main_path = f'/data1/zhangyuhao/xinrong_data/NP1/{mice}/{session_name}/Sorted'
sorted_path = main_path + '/xinrong_sorted'
save_path = f'/home/zhangyuhao/Desktop/Result/Online_Move_Correct/time_series/{session_name}'
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

def raster_plot_neurons(spike_times,neu_id):
    nu_num = len(spike_times)
    for i in range(0,nu_num):
        plt.plot(spike_times[i] , np.repeat(i,len(spike_times[i])), '|', color='gray') 
    plt.title('spike train') 
    plt.xlabel("time")
    #plt.savefig(f"{save_path}/Ras_{region}_trial{trial_num}_{trunc_interval}_{mode}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path}/Ras_{region}_{neu_id}_alltrials_{trunc_interval}_{mode}.png", dpi=600, bbox_inches='tight')
    #plt.savefig(f"{save_path}/Ras_{region}_neuron{neu_id}_{trunc_interval}_{mode}.png", dpi=600, bbox_inches='tight')
    plt.clf()

def raster_plot_neurons_colors(spike_times,col,column):
    nu_num = len(spike_times)
    for i in range(0, nu_num):
        plt.plot(spike_times[i] , np.repeat(i+column,len(spike_times[i])), '|', color=col) 
    plt.title('spike train') 
    plt.xlabel("time")

def has_shift_similarity(S, T, epsilon=0.0):
    """
    判断两个事件时刻点序列是否存在平移后的相似性。
    
    参数：
    S (list): 第一个事件时刻点列表，元素为浮点数或整数。
    T (list): 第二个事件时刻点列表，元素为浮点数或整数。
    epsilon (float): 允许的时间误差范围。
    
    返回：
    bool: 若存在平移量 δ 使得两序列在 ε 误差内匹配，返回 True，否则返回 False。
    """
    S_sorted = sorted(S)
    T_sorted = sorted(T)
    
    # 生成候选平移量 δ 并统计出现次数
    delta_counts = {}
    for s in S_sorted:
        for t in T_sorted:
            delta = t - s
            delta_counts[delta] = delta_counts.get(delta, 0) + 1
    
    # 按出现次数降序排序候选 δ
    candidate_deltas = sorted(delta_counts.keys(), key=lambda x: -delta_counts[x])
    
    # 定义二分查找验证函数
    def check_match(source, target_sorted, delta):
        for point in source:
            target = point + delta
            left, right = 0, len(target_sorted)
            while left < right:
                mid = (left + right) // 2
                if target_sorted[mid] < target - epsilon:
                    left = mid + 1
                else:
                    right = mid
            if left >= len(target_sorted) or target_sorted[left] > target + epsilon:
                return False
        return True
    
    # 检查每个候选 δ
    for delta in candidate_deltas:
        # 正向检查：S 平移 δ 后是否全匹配 T
        if not check_match(S_sorted, T_sorted, delta):
            continue
        # 反向检查：T 平移 -δ 后是否全匹配 S
        if not check_match(T_sorted, S_sorted, -delta):
            continue
        return True
    
    return False

def popu_sptime_trial(neuron_ids,start,end):  # marker_start,marker_end单位是s
    popu_sptime = []
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketrain(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > start) & (spike_times < end)]
        if len(spike_times_trail) > 2:
            align_nu_times = spike_times_trail - start
            popu_sptime.append(align_nu_times)
    return popu_sptime

def compute_event_similarity(event_times, epsilon=0, norm_method='adaptive', significance_test=False):
    """
    改进版时间相似性计算函数
    :param event_times: 事件时间序列列表
    :param epsilon: 时间窗口容差（允许的事件时间偏移）
    :param norm_method: 归一化方法（'min', 'mean', 'geometric', 'adaptive'）
    :param significance_test: 是否启用统计显著性检验
    :return: 相似性矩阵
    """
    n = len(event_times)
    similarity = np.zeros((n, n))
    
    # 计算全局时间范围（用于显著性检验）
    all_times = np.concatenate([seq for seq in event_times if len(seq) > 0])
    time_min = all_times.min() if len(all_times) > 0 else 0
    time_max = all_times.max() if len(all_times) > 0 else 0
    time_range = max(time_max - time_min + 1, 1)  # 防止除零
    
    for i in range(n):
        s = np.array(event_times[i], dtype=np.int64)
        for j in range(i, n):
            t = np.array(event_times[j], dtype=np.int64)
            
            # 处理对角线元素
            if i == j:
                similarity[i][j] = 1.0
                continue
                
            # 处理空序列
            len_s, len_t = len(s), len(t)
            if len_s == 0 or len_t == 0:
                sim = 1.0 if (len_s == 0 and len_t == 0) else 0.0
                similarity[i][j] = sim
                similarity[j][i] = sim
                continue
            
            # 核心计算逻辑
            time_diffs = s[:, None] - t[None, :]  # 生成时间差矩阵
            diffs_flatten = time_diffs.ravel()
            
            if epsilon == 0:
                # 精确匹配模式
                unique_diffs, counts = np.unique(diffs_flatten, return_counts=True)
                max_overlap = counts.max() if len(counts) > 0 else 0
            else:
                # 时间窗口容差模式
                valid_matches = np.abs(diffs_flatten[:, None] - diffs_flatten) <= epsilon
                max_overlap = valid_matches.sum(axis=1).max() if len(diffs_flatten) > 0 else 0
            
            # 分层归一化策略
            if norm_method == 'adaptive':
                if len_s <= 5 and len_t <= 5:
                    denominator = min(len_s, len_t)
                elif (len_s <= 5 or len_t <= 5) and (len_s > 20 or len_t > 20):
                    denominator = (len_s + len_t) / 2
                else:
                    denominator = np.sqrt(len_s * len_t)
            else:
                denominator = {
                    'min': min(len_s, len_t),
                    'mean': (len_s + len_t) / 2,
                    'geometric': np.sqrt(len_s * len_t),
                    'max': max(len_s, len_t)
                }[norm_method]
            
            sim = max_overlap / denominator if denominator != 0 else 0.0
            
            # 统计显著性检验
            if significance_test:
                lambda_ = (len_s * len_t) / time_range
                if lambda_ > 0:
                    p_value = 1 - poisson.cdf(max_overlap - 1, lambda_)
                    if p_value >= 0.05:
                        sim = 0.0  # 不显著时置零
            
            similarity[i][j] = sim
            similarity[j][i] = sim
    
    return similarity

def plot_similarity_heatmap(sim_matrix, trial_i):
    """可视化热图生成函数"""
    ax = sns.heatmap(
        sim_matrix,
        cmap='viridis',
        vmin=0, vmax=1,
        square=True,
        cbar_kws={'shrink': 0.8},
        xticklabels=False,
        yticklabels=False
    )
    cbar = ax.collections[0].colorbar
    cbar.set_label('Similarity Score', rotation=270, labelpad=15)
    plt.title(f"Time Similarity Matrix ({mode})", fontsize=14, pad=20)
    plt.xlabel("Time Series Index", labelpad=15)
    plt.ylabel("Time Series Index", labelpad=15)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{region}_trial{trial_i}_{trunc_interval}_{mode}.png", dpi=600, bbox_inches='tight')
    plt.clf()

def main():
    plt.figure(figsize=(12, 10))
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    if region not in ch['area_name'].values:
        print("region not exist")
        return
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    popu_id = np.array(neurons['cluster_id']).astype(int)
    num = 1
    collection = []
    column = 0
    n_colors = len(events['push_on']) - 1
    cmap = plt.cm.get_cmap('gist_ncar', n_colors)  # 使用高区分度的colormap
    # -- enumarate trials --
    for i in np.arange(0,len(events['push_on'])-1):
        color = cmap(i)  # 直接按索引取色
        # -- marker --
        #trail_start = events['push_on'].iloc[i]   trial_end = events['push_off'].iloc[i]
        pert_start = events['pert_on'].iloc[i]  #单位s
        pert_end = events['pert_off'].iloc[i]   #单位s
        reward_on = events['reward_on'].iloc[i] #单位s
        # pert_start != 0 means there's perturbation in this trial
        # reward_on != 0 means there's a success pushing trial
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:   # beacuse perturbation length is 0.14s~0.16s randomly
            # -- truncate data --
            t1 = pert_start + trunc_interval * mode / 1000  #单位s
            t2 = t1 + trunc_interval / 1000  #单位s
            data = popu_sptime_trial(popu_id,t1,t2)
            column = column + len(data)
            raster_plot_neurons_colors(data,color,column)
            #if len(data) > 0: collection.append(data)
            #sim_matrix_sigtest = compute_event_similarity(data, significance_test=True)
            #plot_similarity_heatmap(sim_matrix_sigtest,i)
            '''
            if len(data) > 1:
                for idx1 in range(len(data)):
                    for idx2 in range(idx1 + 1, len(data)):
                        S = data[idx1]
                        T = data[idx2]
                        print(f"Comparing S (index {idx1}) and T (index {idx2}): {has_shift_similarity(S, T)}") 
            ''' 
    '''  
    # Flatten the collection
    collection = [item for sublist in collection for item in sublist]
    print(collection)
    raster_plot_neurons(collection,'allrespondneurons')
    '''
    plt.savefig(f"{save_path}/Ras_{region}_neurons_coloralltrials_{trunc_interval}_{mode}.png", dpi=600, bbox_inches='tight')

def neuron_trials_tempora():
    plt.figure(figsize=(12, 10))
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    if region not in ch['area_name'].values:
        print("region not exist")
        return
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    popu_id = np.array(neurons['cluster_id']).astype(int)
    num = 1
    #遍历该区域所有neuron
    for j in range(len(popu_id)): 
        neu_trs_sptime = []
        #遍历所有trials
        for i in np.arange(0,len(events['push_on'])-1): 
            # -- marker --
            #trail_start = events['push_on'].iloc[i]   trial_end = events['push_off'].iloc[i]
            pert_start = events['pert_on'].iloc[i]  #单位s
            pert_end = events['pert_off'].iloc[i]   #单位s
            reward_on = events['reward_on'].iloc[i] #单位s
            # 筛选有pertubation且成功的trials
            # pert_start != 0 means there's perturbation in this trial
            # reward_on != 0 means there's a success pushing trial
            if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:   # beacuse perturbation length is 0.14s~0.16s randomly
                # -- truncate data --
                t1 = pert_start + trunc_interval * mode / 1000  #单位s
                t2 = t1 + trunc_interval / 1000  #单位s
                spike_times = singleneuron_spiketrain(popu_id[j])
                spike_times_trail = spike_times[(spike_times > t1) & (spike_times < t2)]
                if len(spike_times_trail) > 2:
                    align_nu_times = spike_times_trail - t1
                    neu_trs_sptime.append(align_nu_times)

        if len(neu_trs_sptime) > 0: raster_plot_neurons(neu_trs_sptime,j)

#neuron_trials_tempora()

main()