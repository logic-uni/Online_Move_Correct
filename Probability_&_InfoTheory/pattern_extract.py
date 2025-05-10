"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 03/20/2025
data from: Xinrong Tan
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import math
import csv
import pandas as pd
import neo
import quantities as pq
from collections import Counter
from matplotlib.pyplot import *
from ast import literal_eval
from elephant.conversion import BinnedSpikeTrain
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import pdist, squareform
from math import log
from ast import literal_eval
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mutual_info_score
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)   

# ------- NEED CHANGE -------
mice = 'mice_1'  # 10-13 mice_1, 14-17 mice_2, 18-22 mice_3
session_name = '20230510'
region = 'Interposed nucleus'   #Simple lobule  Lobule III  Lobules IV-V  Interposed nucleus  

# ------- NO NEED CHANGE -------
### path
main_path = f'/data1/zhangyuhao/xinrong_data/NP1/{mice}/{session_name}/Sorted'
sorted_path = main_path + '/xinrong_sorted'
save_path = f'/home/zhangyuhao/Desktop/Result/Online_Move_Correct/Pattern_extract/{session_name}'
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
    spike_times=np.zeros(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def popu_fr_onetrial(neuron_ids,marker_start,marker_end,fr_bin):
    for j in range(len(neuron_ids)): #第j个neuron
        spike_times = singleneuron_spiketimes(neuron_ids[j])
        spike_times_trail = spike_times[(spike_times > marker_start) & (spike_times < marker_end)]
        spiketrain = neo.SpikeTrain(spike_times_trail,units='sec',t_start=marker_start, t_stop=marker_end)
        fr = BinnedSpikeTrain(spiketrain, bin_size=fr_bin*pq.ms,tolerance=None)
        fr_binar = fr.binarize()  # 对于可能在1ms内出现两个spike的情况，强制置为该bin下即1ms只能有一个spike
        one_neruon = fr_binar.to_array().astype(int)[0]
        #print(one_neruon)
        if j == 0:
            neurons = one_neruon
        else:
            neurons = np.vstack((neurons, one_neruon))
    return neurons

# ------- Pattern Extraction -------

def Pattern_Entropy_trialaverage(data):
    # about bin 1 bit = 1 msec 
    # Statistics pattern all trials
    result_dic={}
    for j in range(0,len(data)):
        trial=data[j]  # get a trial
        for i in range(0,len(trial)-len(trial)%8,8):  # delete end bits that can't be divide by 8
            a = np.array(trial[i:i+8])                # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
            str1 = ''.join(str(z) for z in a)         # array to str
            if str1 not in result_dic:                # use dic to statistic, key = str, value = number of times
                result_dic[str1]=1
            else:
                result_dic[str1]+=1
    '''
    #delete pattern name contain number > 1 and probability so small that can ignore
    str2='2'
    for i in list(result_dic.keys()):
        if str2 in i:
            del result_dic[i]
    '''
    #compute probability
    total=sum(result_dic.values())
    p={k: v / total for k, v in result_dic.items()}
    del result_dic['00000000']
    total_del0=sum(result_dic.values())
    p_del0={k: v / total_del0 for k, v in result_dic.items()}
    '''
    #sorted keys:s
    s0=['00000000']
    s1=[]
    s2=[]
    for i in p.keys():
        if i.count('1')==1:
            s1.append(i)
        if i.count('1')>1:
            s2.append(i)
    s1=sorted(s1)
    s2=sorted(s2)
    s=s0+s1+s2
    sort_p = {key: p[key] for key in s}
    print(sort_p)
    '''
    #del 0 sorted keys:s
    s1=[]
    s2=[]
    for i in p_del0.keys():
        if i.count('1')==1:
            s1.append(i)
        if i.count('1')>1:
            s2.append(i)
    s1=sorted(s1)
    s2=sorted(s2)
    s=s1+s2
    sort_p = {key: p_del0[key] for key in s}
    print(sort_p)
    # information entropy
    h=0
    for i in p:
        h = h - p[i]*log(p[i],2)
    print('Shannon Entropy=%f'%h)
    #plot
    x=list(sort_p.keys())
    y=list(sort_p.values())

    plt.bar(x, y)
    plt.xticks(x, rotation=90, fontsize=10)
    plt.yticks(fontsize=16)
    #plt.ylim(0,0.08)
    plt.ylabel("Probability of pattern", fontsize=16)
    plt.show()

def Pattern_Entropy_eachtrial(data,trial_num,time_interval,stage): 
    # about bin 1 bit = 1 msec 
    # Statistics pattern all neruons
    result_dic={}
    for j in range(0,len(data)):
        data_neuron=data[j]  # get a neuron
        for i in range(0,len(data_neuron)-len(data_neuron)%8,8):  # delete end bits that can't be divide by 8
            a = np.array(data_neuron[i:i+8])                      # slice into 8 bit,  1 byte(1字节)(1B) = 8 bit(8比特)(8位二进制)；1KB = 1024B; 1MB = 1024KB; 1GB = 1024MB
            str1 = ''.join(str(z) for z in a)                     # array to str
            if str1 not in result_dic:                            # use dic to statistic, key = str, value = number of times
                result_dic[str1]=1
            else:
                result_dic[str1]+=1
    
    #compute probability
    total=sum(result_dic.values())
    p={k: v / total for k, v in result_dic.items()}
    if '00000000' in result_dic:
        del result_dic['00000000']
    total_del0=sum(result_dic.values())
    p_del0={k: v / total_del0 for k, v in result_dic.items()}
    #del 0 sorted keys:s
    s1=[]
    s2=[]
    for i in p_del0.keys():
        if i.count('1')==1:
            s1.append(i)
        if i.count('1')>1:
            s2.append(i)
    s1=sorted(s1)
    s2=sorted(s2)
    s=s1+s2
    sort_p = {key: p_del0[key] for key in s}
    # information entropy
    h=0
    for i in p:
        h = h - p[i]*log(p[i],2)
    # plot
    x=list(sort_p.keys())
    y=list(sort_p.values())
    plt.bar(x, y)
    plt.xticks(x, rotation=90, fontsize=10)
    plt.yticks(fontsize=16)
    plt.text(0.95, 0.95, f'Stage {stage}\n\nTrial {trial_num}\n\n{region}\n\nEach Truncated Time Length: {time_interval} ms\n\nShannon Entropy: {h}', 
        ha='right', va='top', transform=plt.gca() .transAxes, fontsize=10)
    plt.ylabel("Probability of pattern", fontsize=16)
    plt.savefig(save_path + f"/{region}_{stage}_trial{trial_num}.png", dpi=300, bbox_inches='tight')
    plt.clf()

def cluster(data,trial_num):
    # 计算汉明距离矩阵
    distance_matrix = squareform(pdist(data, metric='hamming'))

    # t-SNE降维
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        init="random",  # 或移除 init 参数（某些版本默认允许）
        random_state=42,
        perplexity=30
    )
    projected = tsne.fit_transform(distance_matrix)

    # 可视化
    plt.scatter(projected[:, 0], projected[:, 1], s=5)
    plt.title('t-SNE Projection')
    plt.savefig(save_path + f"/dis_{region}_trial{trial_num}.png", dpi=300, bbox_inches='tight')
    plt.clf()

    # K-means聚类（假设K=3）
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(projected)

    # 评估轮廓系数
    score = silhouette_score(projected, clusters)
    print(f"Silhouette Score: {score:.2f}")

    # 绘制聚类结果
    plt.scatter(projected[:, 0], projected[:, 1], c=clusters, cmap='viridis', s=5)
    plt.title('K-means Clustering')
    plt.savefig(save_path + f"/cluster_{region}_trial{trial_num}.png", dpi=300, bbox_inches='tight')
    plt.clf()

    # 分析簇特征
    for cluster in np.unique(clusters):
        print(f"Cluster {cluster}:")
        print("Mean of each bit:", data[clusters == cluster].mean(axis=0))

def patt_syllable_eachtr(data,trial_num,time_interval,stage): 
    token_length = 20
    # about bin 1 bit = 1 msec 
    # Statistics pattern all neruons
    result_dic={}
    for j in range(0,len(data)):
        data_neuron=data[j]  # get a neuron
        for i in range(0,len(data_neuron),token_length):  # delete end bits that can't be divide by 8
            a = np.array(data_neuron[i:i+token_length])                    
            str1 = ''.join(str(z) for z in a)            # array to str
            if str1 not in result_dic:                   # use dic to statistic, key = str, value = number of times
                result_dic[str1]=1
            else:
                result_dic[str1]+=1
    
    #compute probability
    pure_zero = '0' * token_length
    if pure_zero in result_dic: del result_dic[pure_zero]
    total = sum(result_dic.values())
    p={k: v / total for k, v in result_dic.items()}
    #sorted
    s1=[]
    s2=[]
    for i in p.keys():
        if i.count('1')==1:
            s1.append(i)
        if i.count('1')>1:
            s2.append(i)
    s1=sorted(s1)
    s2=sorted(s2)
    s=s1+s2
    sort_p = {key: p[key] for key in s}
    # information entropy
    h=0
    for i in p:
        h = h - p[i]*log(p[i],2)
    # plot
    x=list(sort_p.keys())
    y=list(sort_p.values())
    plt.bar(x, y)
    plt.xticks(x, rotation=90, fontsize=10)
    plt.yticks(fontsize=16)
    plt.text(0.95, 0.95, f'Stage {stage}\n\nTrial {trial_num}\n\n{region}\n\nEach Truncated Time Length: {time_interval} ms\n\nShannon Entropy: {h}', 
        ha='right', va='top', transform=plt.gca() .transAxes, fontsize=10)
    plt.ylabel("Probability of pattern(without pure 0)", fontsize=16)
    plt.savefig(save_path + f"/{region}_{stage}_trial{trial_num}.png", dpi=300, bbox_inches='tight')
    plt.clf()

def token_type(data,stage):
    # 分割数据为5个20列的数组
    subarrays = np.split(data, 5, axis=1)

    # 存储统计结果和token信息
    all_counts = []
    all_tokens = []

    for seg_idx, sub in enumerate(subarrays, 1):  # 从1开始计数
        # 计算每行1的数量（基于20列的子数组）
        sums = sub.sum(axis=1)
        counts = np.bincount(sums, minlength=21)[:21]
        all_counts.append(counts)
        
        # 存储当前segment的token信息
        segment_tokens = {t: [] for t in range(4, 11)}
        for target_type in range(4, 11):  # 处理type5-type10
            indices = np.where(sums == target_type)[0]
            for idx in indices:
                # 获取20位token并转换为字符串
                token_20bit = ''.join(sub[idx].astype(int).astype(str))
                segment_tokens[target_type].append(token_20bit)
        all_tokens.append(segment_tokens)

    # 可视化部分
    max_count = max([max(c) for c in all_counts])

    plt.figure(figsize=(12, 18))
    for i, counts in enumerate(all_counts):
        plt.subplot(5, 1, i+1)
        bars = plt.bar(range(21), counts, color='skyblue')
        
        # 添加柱状图数值标签
        for bar in bars:
            if (height := bar.get_height()) > 0:
                plt.text(bar.get_x()+bar.get_width()/2, height,
                        f'{height}', ha='center', va='bottom')
        
        # 添加图表元信息
        plt.title(f'Time Segment {i+1}', fontsize=12)
        plt.xlabel('Number of 1s (Type)', fontsize=10)
        plt.ylabel('Count', fontsize=10)
        plt.xticks(range(0, 21, 1))
        plt.xlim(-0.5, 20.5)
        plt.ylim(0, max_count*1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 在右上角添加token信息
        text_content = []
        for t in range(4, 11):
            tokens = all_tokens[i].get(t, [])
            if tokens:
                token_str = f'Type {t}:\n' + '\n'.join(tokens)
                text_content.append(token_str)
        
        if text_content:
            plt.text(0.98, 0.95, '\n\n'.join(text_content), 
                     transform=plt.gca().transAxes,
                     ha='right', va='top',
                     fontsize=7,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.tight_layout()
    plt.savefig(save_path + f"/tokentype_{region}_{stage}.png", dpi=300, bbox_inches='tight')
    plt.clf()
    return all_counts, all_tokens

def token_type_1or2(data,stage):
    # 分割数据为5个20列的数组
    subarrays = np.split(data, 5, axis=1)

    # 存储统计结果和token信息
    all_counts = []
    all_tokens = []

    for seg_idx, sub in enumerate(subarrays, 1):  # 从1开始计数
        # 计算每行1的数量（基于20列的子数组）
        sums = sub.sum(axis=1)
        counts = np.bincount(sums, minlength=21)[:21]
        all_counts.append(counts)
        
        # 存储当前segment的token信息
        segment_tokens = {t: [] for t in range(0, 2)}
        for target_type in range(0, 2):  # 处理type1-type2
            indices = np.where(sums == target_type)[0]
            for idx in indices:
                # 获取20位token并转换为字符串
                token_20bit = ''.join(sub[idx].astype(int).astype(str))
                segment_tokens[target_type].append(token_20bit)
        all_tokens.append(segment_tokens)

    return all_counts, all_tokens



def create_3d_summary(all_stage_counts, save_path, region):
    fig = plt.figure(figsize=(22, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    # 高级配置参数
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 高对比度颜色
    stage_names = ['Pre-P1', 'Pre-P2', 'Post-P1', 'Post-P2']
    type_spacing = 1.5    # 类型间距增加25%
    time_spacing = 2.0    # 时间间距增加33%
    bar_width = 0.6       # 柱宽减少14%
    bar_depth = 0.4       # 柱深减少14%

    # 动态计算最大计数
    max_count = max(max(sub_counts[1:11]) for stage in all_stage_counts for sub_counts in stage)
    
    # 三维柱状图绘制系统
    for stage_idx in range(4):
        stage_counts = all_stage_counts[stage_idx]
        y_base = stage_idx * 5 * time_spacing  # 阶段基准位置
        
        for seg_idx in range(5):
            counts = stage_counts[seg_idx][1:11]
            y_pos = y_base + seg_idx * time_spacing + 0.5  # 中心对齐
            
            for type_idx, count in enumerate(counts):
                if count > 0:
                    # 计算三维坐标
                    x = (type_idx + 1) * type_spacing
                    y = y_pos
                    
                    # 添加随机偏移避免完全重叠（±0.1范围）
                    x += np.random.uniform(-0.1, 0.1)
                    y += np.random.uniform(-0.1, 0.1)
                    
                    ax.bar3d(
                        x - bar_width/2,
                        y - bar_depth/2,
                        0,
                        bar_width,
                        bar_depth,
                        count,
                        color=colors[stage_idx],
                        alpha=0.85,
                        edgecolor='k',
                        linewidth=0.8,
                        zsort='min'
                    )

    # 坐标轴优化系统
    ax.set_xlabel('Token Type', fontsize=14, labelpad=18)
    ax.set_ylabel('Time Sequence', fontsize=14, labelpad=20)
    ax.set_zlabel('Frequency', fontsize=14, labelpad=15)
    
    # X轴设置
    ax.set_xticks([(i+1)*type_spacing for i in range(10)])
    ax.set_xticklabels(range(1,11), fontsize=12, rotation=45)
    ax.set_xlim(type_spacing*0.8, type_spacing*10.2)
    
    # Y轴阶段标记
    y_ticks = [2.5*time_spacing, 7.5*time_spacing, 
              12.5*time_spacing, 17.5*time_spacing]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(stage_names, 
                      fontsize=13,
                      rotation=15,
                      ha='right',
                      va='center')
    
    # 最佳视角参数
    ax.view_init(elev=42, azim=-115)
    
    # 添加辅助标注
    ax.text2D(0.05, 0.93, 
             f"Data Region: {region}\nMax Frequency: {max_count}",
             transform=ax.transAxes,
             fontsize=11,
             bbox=dict(facecolor='white', alpha=0.9))
    
    plt.title(f'Token Type Distribution Analysis (1-10)\n{region}', 
             fontsize=15, pad=18)
    plt.savefig(save_path + f"/3d_summary_{region}.png", dpi=320, bbox_inches='tight')
    plt.close()

def tokenizer(data,time_interval,stage): 
    token_length = 20
    # 统计数据结构
    token_counts = defaultdict(int)
    token_positions = defaultdict(list)

    # 分割和统计流程
    for row_idx in range(data.shape[0]):
        row = data[row_idx]
        for start in range(0, data.shape[1], token_length):
            end = start + token_length
            token_slice = row[start:end]
            
            if np.all(token_slice == 0):
                continue  # 跳过全零token
                
            token = tuple(token_slice.astype(int).tolist())
            token_counts[token] += 1
            token_pos_in_row = start // token_length
            token_positions[token].append((row_idx, token_pos_in_row))

    # 准备绘图数据
    sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
    binary_strings = [''.join(map(str, token)) for token, _ in sorted_tokens]
    frequencies = [count for _, count in sorted_tokens]
    total_samples = sum(frequencies)  # 计算总样本数
    probabilities = [count / total_samples for count in frequencies]  # 计算概率
    total_tokens = len(sorted_tokens)

    # 动态调整可视化参数
    base_height = 6  # 基础高度
    row_height = 0.4  # 每行高度系数
    figsize_width = 16
    figsize_height = base_height + row_height * total_tokens
    fontsize = max(6, 10 - int(total_tokens/50))  # 根据数量自动缩小字体

    # 创建可视化
    plt.figure(figsize=(figsize_width, figsize_height))
    ax = plt.subplot()
    y_pos = np.arange(total_tokens)

    # 绘制完整数据（使用概率）
    bars = ax.barh(y_pos, probabilities, height=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(binary_strings, fontfamily='monospace', fontsize=fontsize)
    ax.invert_yaxis()  # 最高频在上方

    # 设置x轴为百分比格式
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))

    # 标注坐标信息
    ax.set_xlabel('Probability (%)', fontsize=10)
    ax.set_ylabel(f'{token_length}-bit Tokens', fontsize=10)
    ax.set_title(f'Probability Distribution of {token_length}-bit Tokens\nTotal Samples: {total_samples} (Excluding Zero-Tokens)', 
                fontsize=12, pad=20)

    # 优化网格线
    ax.xaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 添加概率数值标签
    max_prob = max(probabilities)
    for bar, prob in zip(bars, probabilities):
        if prob > 0.05 * max_prob:  # 只显示大于最大概率5%的标签
            label = f"{prob:.2%}" if prob >= 0.0001 else f"{prob:.2e}"
            ax.text(bar.get_width() + 0.002,  # 调整偏移量为概率量级
                    bar.get_y() + bar.get_height()/2,
                    label,
                    va='center',
                    fontsize=fontsize-1,
                    color='#333333')

    plt.tight_layout()
    plt.savefig(save_path + f"/token_{region}_{stage}.png", dpi=300, bbox_inches='tight')
    plt.clf()

    with open(save_path + f"/token_{region}_{stage}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Token', 'Count', 'Probability', 'Positions'])
        writer.writerows([
            [''.join(map(str, token)), count, count/total_samples, positions]
            for (token, count), positions in zip(sorted_tokens, token_positions.values())
        ])
    
def InfoPlot():
    x=['PC d:120','PC d:180','PC d:280','PC d:400','IPN d:1580','IPN d:1820','IPN d:1900','IPN d:1960']
    y=[2.3,3.3,3.6,2.8,0.5,0.5,0.3,0.2]
    plt.bar(x, y)
    plt.title('Quantities of information', fontsize=16)
    plt.xticks(x, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Shannon entropy", fontsize=16)
    plt.show()

def plot_type1_distributions(all_stage_counts, all_stage_tokens, save_path, region):
    # 配置参数
    max_tokens_to_show = 20
    color_palette = plt.cm.tab20c(np.linspace(0, 1, max_tokens_to_show))
    
    # 遍历四个实验阶段
    for stage_idx, (stage_counts, stage_tokens) in enumerate(zip(all_stage_counts, all_stage_tokens)):
        # 阶段元数据
        stage_names = ['Pre-P1', 'Pre-P2', 'Post-P1', 'Post-P2']
        stage_label = stage_names[stage_idx]
        
        # 计算总token数（所有类型）
        total_global = sum([np.sum(sub_counts) for sub_counts in stage_counts])
        
        # 收集所有type1的token
        all_type1_tokens = []
        for seg_tokens in stage_tokens:
            if 1 in seg_tokens:  # 检查是否存在type1
                all_type1_tokens.extend(seg_tokens[1])
        
        # 统计token频率
        token_counter = Counter(all_type1_tokens)
        total_conditional = sum(token_counter.values())
        
        # 处理无type1的情况
        if total_conditional == 0:
            print(f"No type1 tokens found in {stage_label}")
            continue
        
        # 准备概率数据
        sorted_tokens = token_counter.most_common(max_tokens_to_show)
        labels = [t[0] for t in sorted_tokens]
        global_probs = [t[1]/total_global for t in sorted_tokens]
        conditional_probs = [t[1]/total_conditional for t in sorted_tokens]
        
        # 创建可视化面板
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), 
                                      gridspec_kw={'width_ratios': [1, 1]})
        
        # 全局概率分布图
        bars1 = ax1.barh(np.arange(max_tokens_to_show), 
                        global_probs[::-1], 
                        color=color_palette[::-1],
                        edgecolor='k',
                        height=0.8)
        ax1.set_yticks(np.arange(max_tokens_to_show))
        ax1.set_yticklabels(labels[::-1], fontsize=10)
        ax1.set_xlabel('Global Probability (All Tokens)', fontsize=12)
        ax1.set_title(f'Distribution in Global Context\nTotal Tokens: {total_global}', 
                     fontsize=13)
        
        # 添加数值标签
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', 
                    ha='left', va='center', fontsize=9)
        
        # 条件概率分布图
        bars2 = ax2.barh(np.arange(max_tokens_to_show), 
                        conditional_probs[::-1], 
                        color=color_palette[::-1],
                        edgecolor='k',
                        height=0.8)
        ax2.set_yticks([])
        ax2.set_xlabel('Conditional Probability (Type1 Only)', fontsize=12)
        ax2.set_title(f'Distribution within Type1\nTotal Type1: {total_conditional}', 
                     fontsize=13)
        
        # 添加数值标签
        for bar in bars2:
            width = bar.get_width()
            ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', 
                    ha='left', va='center', fontsize=9)
        
        # 公共设置
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        
        plt.suptitle(f'Type1 Token Distribution - {region} ({stage_label})\n'
                    f'Top {max_tokens_to_show} Frequent Patterns', 
                    y=0.98, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_path}/type1_distribution_{stage_label}_{region}.png", 
            dpi=350, bbox_inches='tight')
        plt.close()

def plot_post1_type1_timeseries(all_stage_counts, all_stage_tokens, save_path, region):
    # 提取post1阶段数据（索引2）
    post1_counts = all_stage_counts[2]  # 5个时间段的计数数组
    post1_tokens = all_stage_tokens[2]  # 5个时间段的token字典
    
    # 配置参数
    max_tokens = 10  # 显示前10个高频token
    color_map = plt.cm.tab10(np.linspace(0, 1, max_tokens))
    
    # 创建画布
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.25)
    
    # 遍历每个时间段
    for seg_idx in range(5):
        ax1 = fig.add_subplot(gs[seg_idx, 0])  # 全局概率
        ax2 = fig.add_subplot(gs[seg_idx, 1])  # 条件概率
        
        # 获取当前时间段数据
        seg_counts = post1_counts[seg_idx]
        seg_data = post1_tokens[seg_idx]
        
        # 提取type1的token列表
        type1_tokens = seg_data.get(1, [])
        total_global = np.sum(seg_counts)  # 当前段总token数
        total_type1 = len(type1_tokens)    # type1数量
        
        # 处理无数据情况
        if total_type1 == 0:
            ax1.text(0.5, 0.5, 'No Type1 Tokens', ha='center', va='center')
            ax2.text(0.5, 0.5, 'No Type1 Tokens', ha='center', va='center')
            continue
        
        # 统计token频率
        counter = Counter(type1_tokens)
        sorted_items = counter.most_common(max_tokens)
        
        # 准备概率数据
        labels = [item[0] for item in sorted_items]
        global_probs = [item[1]/total_global for item in sorted_items]
        conditional_probs = [item[1]/total_type1 for item in sorted_items]
        
        # 全局概率分布
        ax1.barh(np.arange(max_tokens), global_probs[::-1], 
                color=color_map[::-1], height=0.8, edgecolor='k')
        ax1.set_yticks(np.arange(max_tokens))
        ax1.set_yticklabels(labels[::-1], fontsize=9)
        ax1.set_xlabel('Global Prob', fontsize=10)
        ax1.set_title(f'Segment {seg_idx+1} - Global\nTotal Tokens: {total_global}', fontsize=11)
        
        # 条件概率分布
        ax2.barh(np.arange(max_tokens), conditional_probs[::-1],
                color=color_map[::-1], height=0.8, edgecolor='k')
        ax2.set_yticks([])
        ax2.set_xlabel('Conditional Prob', fontsize=10)
        ax2.set_title(f'Segment {seg_idx+1} - Type1 Only\nTotal Type1: {total_type1}', fontsize=11)
        
        # 添加数值标签
        for ax in [ax1, ax2]:
            for bar in ax.patches:
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # 公共标注
    plt.suptitle(f'Post-Perturbation 1 Type1 Token Distribution Evolution - {region}\n'
                f'Top {max_tokens} Frequent Patterns per Segment', 
                y=0.98, fontsize=14)
    plt.savefig(f"{save_path}/post1_type1_evolution_{region}.png", 
               dpi=350, bbox_inches='tight')
    plt.close()

def pattern_extract():
    fr_bin = 1
    #plt.figure(figsize=(40, 30))
    plt.figure(figsize=(8, 5))
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    if region not in ch['area_name'].values:
        print("region not exist")
        return
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    popu_id = np.array(neurons['cluster_id']).astype(int)
    time_interval = 100  #单位ms
    print(f'each truncated time length: {time_interval} ms')
    # pertur前后各两个matrix
    two_seg_bin_num = time_interval*2*fr_bin/1000  
    num = 1
    bef_per1_trs, bef_per2_trs, aft_per1_trs, aft_per2_trs = [], [], [], []
    # -- enumarate trials --
    for i in np.arange(0,len(events['push_on'])-1):
        # -- marker --
        #trail_start = events['push_on'].iloc[i]   trial_end = events['push_off'].iloc[i]
        pert_start = events['pert_on'].iloc[i]
        pert_end = events['pert_off'].iloc[i]
        reward_on = events['reward_on'].iloc[i]
        # pert_start != 0 means there's perturbation in this trial
        # reward_on != 0 means there's a success pushing trial
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16:  # beacuse perturbation length is 0.14s~0.16s randomly
            # -- truncate data --
            data = popu_fr_onetrial(popu_id,pert_start-two_seg_bin_num,pert_start+two_seg_bin_num+0.0006,fr_bin)  # pert 前200ms 后200ms
            print(f"neuron_fr_mat: {data.shape}")
            bef_per_1 = data[:,0:time_interval]
            bef_per_2 = data[:,time_interval:time_interval*2]
            aft_per_1 = data[:,time_interval*2:time_interval*3]
            aft_per_2 = data[:,time_interval*3:time_interval*4]

            bef_per1_trs.append(bef_per_1)
            bef_per2_trs.append(bef_per_2)
            aft_per1_trs.append(aft_per_1)
            aft_per2_trs.append(aft_per_2)

            #cluster(data,num)
            #patt_syllable_eachtr(bef_per_1,num,time_interval,'before_pertur_1')
            #patt_syllable_eachtr(bef_per_2,num,time_interval,'before_pertur_2')
            #patt_syllable_eachtr(aft_per_1,num,time_interval,'after_pertur_1') 
            #patt_syllable_eachtr(aft_per_2,num,time_interval,'after_pertur_2')
            num = num + 1

    bef_per1_trs_arr = np.vstack(bef_per1_trs)
    bef_per2_trs_arr = np.vstack(bef_per2_trs)
    aft_per1_trs_arr = np.vstack(aft_per1_trs)
    aft_per2_trs_arr = np.vstack(aft_per2_trs)
    
    print(bef_per1_trs_arr.shape)
    np.save(save_path + f"/{region}_bef_per1.npy", bef_per1_trs_arr)
    print(bef_per2_trs_arr.shape)
    np.save(save_path + f"/{region}_bef_per2.npy", bef_per2_trs_arr)
    print(aft_per1_trs_arr.shape)
    np.save(save_path + f"/{region}_aft_per1.npy", aft_per1_trs_arr)
    print(aft_per2_trs_arr.shape)
    np.save(save_path + f"/{region}_aft_per2.npy", aft_per2_trs_arr)

    '''
    # 存储统计全部token counts
    all_stage_counts = []
    for data, stage in [
        (bef_per1_trs_arr, 'bef_pert_1'),
        (bef_per2_trs_arr, 'bef_pert_2'),
        (aft_per1_trs_arr, 'aft_pert_1'),
        (aft_per2_trs_arr, 'aft_pert_2')
    ]:
        counts, tokens = token_type(data, stage)
        all_stage_counts.append(counts)
    np.save(save_path + f"/{region}_all_stage_counts.npy", all_stage_counts)'
    
    # 存储统计 type1 token & counts
    all_stage_counts = []
    all_stage_tokens = []
    for data, stage in [
        (bef_per1_trs_arr, 'bef_pert_1'),
        (bef_per2_trs_arr, 'bef_pert_2'),
        (aft_per1_trs_arr, 'aft_pert_1'),
        (aft_per2_trs_arr, 'aft_pert_2')
    ]:
        counts, tokens = token_type_1or2(data, stage)
        all_stage_counts.append(counts)
        all_stage_tokens.append(tokens)
    #plot_type1_distributions(all_stage_counts, all_stage_tokens, save_path, region)
    plot_post1_type1_timeseries(all_stage_counts, all_stage_tokens, save_path, region)
    '''
    #create_3d_summary(all_stage_counts, save_path, region)
    #token_type(bef_per1_trs_arr,'bef_pert_1')
    #token_type(bef_per2_trs_arr,'bef_pert_2')
    #token_type(aft_per1_trs_arr,'aft_pert_1')
    #token_type(aft_per2_trs_arr,'aft_pert_2')
    #tokenizer(bef_per2_trs_arr,time_interval,'bef_pert_2')
    #tokenizer(aft_per1_trs_arr,time_interval,'aft_pert_1')

pattern_extract()