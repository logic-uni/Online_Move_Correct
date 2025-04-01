"""
# coding: utf-8
@author: Yuhao Zhang
Last updated : 03/27/2025
data from: Xinrong Tan
"""
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import expon
np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

# ------- NEED CHANGE -------
mice = 'mice_1'  # 10-13 mice_1, 14-17 mice_2, 18-22 mice_3
session_name = '20230511'
region = 'Lobules IV-V'   #Simple lobule  Lobule III  Lobules IV-V  Interposed nucleus
mode = -1  # mode 0 after per 0~100ms,mode 1 after per 100~200ms,mode -1 before per -100~0ms,
trunc_interval = 100  #单位ms  每个neuron在20ms/50ms的截断下即使是所有trial也无法得到显著统计

# ------- NO NEED CHANGE -------
### parameter
fr_filter = 1         # 1  firing rate > 1
cutoff_distr = 250           # 250ms/None  cutoff_distr=0.25代表截断ISI分布大于0.25s的
histo_bin_num = 100          # 统计图bin的个数
### path
main_path = f'/data1/zhangyuhao/xinrong_data/NP1/{mice}/{session_name}/Sorted'
sorted_path = main_path + '/xinrong_sorted'
save_path = f'/home/zhangyuhao/Desktop/Result/Online_Move_Correct/ISI/{session_name}'
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

# get single neuron spike train
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def fitting(neuron_intervals,lambdas):
    ## 画完当前neuron所有trial的hist之后，对当前neuron整体的ISI拟合指数分布
    if len(neuron_intervals) != 0 and cutoff_distr == None:
        params = expon.fit(neuron_intervals, floc=0)  # 固定起始点为0
        lambda_fit = 1 / params[1]  # 拟合的lambda值
        # 绘制拟合的指数分布曲线
        x = np.linspace(0, max(neuron_intervals), 100)
        plt.plot(x, expon.pdf(x, *params), 'b-', lw=0.2)
        lambdas.append(lambda_fit)  # 保存lambda值 each lambda represent a neurons firing strength in different run trials
    if cutoff_distr == None:
        plt.xlabel('Inter-spike Interval (s)')
        plt.ylabel('Probability Density')
        plt.title(f'ISI Distribution_{region}')
        # 画完所有的ISI分布后 计算平均 lambda 值
        lambda_mean = np.mean(lambdas)
        neuron_infig_num = len(lambdas)
        # 在图的右上角显示平均的lambda值
        plt.text(0.95, 0.95, f'{mice}\n\nfiring filter: 1 spike/s\n\nneurons num after filter: {neuron_infig_num}\n\nMean λ: {lambda_mean:.2f}\n\nhisto bin: 20\n\n{mode} trials', ha='right', va='top', transform=plt.gca().transAxes)
        # 插入子图表示lambda的分布
        left, bottom, width, height = 0.6,0.25,0.25,0.25
        plt.axes([left,bottom,width,height])
        plt.hist(lambdas, bins=50, density=False, alpha=0.6)
        plt.title(f'lambda distribution')
        plt.xlabel('lambda value')
        plt.ylabel('Prob.')

def ISI(spike_times,unit_id):
    neuron_intervals = []  # 用于存储对应mode下所有trial的spike interval
    for j in range(len(events['pert_on'])): #逐行遍历每个trial
        reset_on = events['reset_on'].iloc[j]  #单位s
        reset_off = events['reset_off'].iloc[j] #单位s
        push_on = events['push_on'].iloc[j] #单位s
        pert_start = events['pert_on'].iloc[j] #单位s
        pert_end = events['pert_off'].iloc[j] #单位s
        push_off = events['push_off'].iloc[j] #单位s
        reward_on = events['reward_on'].iloc[j] #单位s
        
        if pert_start != 0 and reward_on != 0 and pert_end-pert_start > 0.14 and pert_end-pert_start < 0.16: # 提取成功的trial
            t1 = pert_start + trunc_interval * mode / 1000  #单位s
            t2 = t1 + trunc_interval / 1000  #单位s
            spike_times_trail = spike_times[(spike_times > t1) & (spike_times < t2)]
            if len(spike_times_trail) > (fr_filter*(t2-t1)):   #筛掉发放率低的 在这个truncation下必须有1spike/s
                intervals = np.diff(spike_times_trail)  # 计算时间间隔
                intervals = intervals * 1000   # 转为ms单位
                if cutoff_distr != None:
                    intervals = intervals[(intervals > 0.000999999999) & (intervals <= cutoff_distr)]
                    if len(intervals) != 0:  #截断后可能导致没有interval    
                        neuron_intervals.extend(intervals)  #单个neuron所有trial的interval放到一个数组里
    
    if len(neuron_intervals) != 0: 
        #  plot each neuron - trials ISI
        plt.hist(neuron_intervals, bins=histo_bin_num, density=False,alpha=0.6)
        plt.title(f'ISI distrbution_neuron_id_{unit_id}')
        plt.xlabel('Inter-spike Interval (ms)')
        plt.ylabel('Counts')
        #plt.hist(neuron_intervals, bins=100, density=False, alpha=0.6,color=neuron_color)
        plt.text(0.95, 0.95, f'{session_name}\n\nfiring filter: 1 spike/s\n\nhisto bin: 100\n\ndistr cutoff > {cutoff_distr}\n\n{mode}',
            ha='right', va='top', transform=plt.gca().transAxes)
        plt.savefig(save_path+f"/{region}_neuron_{unit_id}_{trunc_interval}_{mode}.png",dpi=600,bbox_inches = 'tight')
        plt.clf()
    
    return neuron_intervals

def collect_ISI_data(spike_times, unit_id, modes, histo_bin_num=100, cutoff_distr=None):
    #收集不同 mode 下的 ISI 分布数据
    all_modes_data = {}  # 存储所有 mode 的直方图数据（键: mode, 值: (counts, bins)）
    for mode in modes:
        neuron_intervals = []  # 存储当前 mode 的所有 interval
        for j in range(len(events['pert_on'])):
            reset_on = events['reset_on'].iloc[j]
            reset_off = events['reset_off'].iloc[j]
            push_on = events['push_on'].iloc[j]
            pert_start = events['pert_on'].iloc[j]
            pert_end = events['pert_off'].iloc[j]
            push_off = events['push_off'].iloc[j]
            reward_on = events['reward_on'].iloc[j]
            if pert_start != 0 and reward_on != 0 and 0.14 < (pert_end - pert_start) < 0.16:
                t1 = pert_start + trunc_interval * mode / 1000  
                t2 = t1 + trunc_interval / 1000
                spike_times_trail = spike_times[(spike_times > t1) & (spike_times < t2)]
                
                if len(spike_times_trail) > (fr_filter * (t2 - t1)):
                    intervals = np.diff(spike_times_trail) * 1000  # 转为ms
                    if cutoff_distr is not None:
                        intervals = intervals[(intervals > 0.001) & (intervals <= cutoff_distr)]
                    if len(intervals) > 0:
                        neuron_intervals.extend(intervals)

        # 生成直方图数据（不绘图）
        counts, bins = np.histogram(neuron_intervals, bins=histo_bin_num, density=False)
        all_modes_data[mode] = (counts, bins)
    
    return all_modes_data

def plot_3d_isi(all_modes_data, unit_id):
    """
    绘制三维 ISI 分布图
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 统一所有 mode 的 bins（取第一个 mode 的 bins 作为基准）
    bins = list(all_modes_data.values())[0][1]
    x = (bins[:-1] + bins[1:]) / 2  # 取中点作为 x 轴坐标
    modes = sorted(all_modes_data.keys())
    
    # 为每个 mode 绘制直方图
    for mode in modes:
        counts, _ = all_modes_data[mode]
        y = np.full_like(x, mode)  # y 轴为 mode 值
        ax.bar3d(x, y, np.zeros_like(counts), 
                 dx=(bins[1]-bins[0])*0.8,  # 控制条宽
                 dy=0.5,  # 控制 mode 间距
                 dz=counts, 
                 shade=True)
    
    ax.set_xlabel('ISI (ms)', fontsize=10)
    ax.set_ylabel('Mode', fontsize=10)
    ax.set_zlabel('Counts', fontsize=10)
    ax.set_title(f'3D ISI Distribution - Neuron {unit_id}\n{session_name}', fontsize=12)
    plt.savefig(f"{save_path}/{region}_neuron_{unit_id}_3d_isi.png", dpi=600, bbox_inches='tight')
    plt.close()

def main():
    popu_intervals = []
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    if region not in ch['area_name'].values:
        print("region not exist")
        return
    site_id = ch[ch['area_name'] == region].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    popu_id = np.array(neurons['cluster_id']).astype(int)
    print(popu_id)
    neuron_colors = plt.cm.hsv(np.linspace(0, 1, len(popu_id)))
    plt.figure(figsize=(10, 6))
    for i in range(len(popu_id)):
        spike_times = singleneuron_spiketrain(popu_id[i])
        #color = neuron_colors[i]
        neuron_intervals = ISI(spike_times,popu_id[i])
        popu_intervals.extend(neuron_intervals)
    '''
    # plot all neurons trials ISI
    plt.hist(popu_intervals, bins=histo_bin_num, density=False,alpha=0.6)
    plt.xlabel('Inter-spike Interval (ms)')
    plt.ylabel('Counts')
    plt.title(f'ISI distrbution_popu')
    #plt.hist(neuron_intervals, bins=100, density=False, alpha=0.6,color=neuron_color)
    plt.text(0.95, 0.95, f'{session_name}\n\nfiring filter: 1 spike/s\n\nhisto bin: 100\n\ndistr cutoff > {cutoff_distr}\n\n{mode}',
        ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(save_path+f"/{region}_popu_{trunc_interval}_{mode}.png",dpi=600,bbox_inches = 'tight')
    plt.clf()
    '''
        
main()