"""
# coding: utf-8
@author: Yuhao Zhang
Last updated : 11/15/2024
data from: Xinrong Tan
"""
import neo
import quantities as pq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import expon

np.set_printoptions(threshold=np.inf)
np.seterr(divide='ignore',invalid='ignore')

### path
mice_name = '20230511'
main_path = r'E:\xinrong\mice_1\20230511\cage1-2-R-2_g0\cage1-2-R-2_g0_imec0'
fig_save_path = r'C:\Users\zyh20\Desktop\Perturbation_analysis\ISI\20230511\during_pertur\Lobules IV-V'

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
def singleneuron_spiketrain(id):
    x = np.where(identities == id)
    y=x[0]
    spike_times=np.empty(len(y))
    for i in range(0,len(y)):
        z=y[i]
        spike_times[i]=times[z]/sample_rate
    return spike_times

def ISI_sub(spike_times,fr_fil,cutoff_distr,mode,neuron_color):
    neuron_intervals = []  # 用于存储该neuron的所有时间间隔
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
            # trancation
            if mode == 'before_reset':
                ## during reset
                t1 = reset_on-0.5 
                t2 = reset_on # unit sec
            elif mode == 'after_reset':
                ## during reset
                t1 = reset_on 
                t2 = reset_on+0.5  # unit sec
            elif mode == 'before_pertur':
                ## during perturbation
                t1 = pert_on-0.5  # unit sec
                t2 = pert_on # unit sec
            elif mode == 'after_pertur':
                ## during perturbation
                t1 = pert_on  # unit sec
                t2 = pert_on+0.5 # unit sec
            spike_times_trail = spike_times[(spike_times > t1) & (spike_times < t2)]
            if len(spike_times_trail)/(t2-t1) > fr_fil: 
                intervals = np.diff(spike_times_trail)  # 计算时间间隔
                # 绘制时间间隔的直方图
                if cutoff_distr != None:
                    intervals = intervals[intervals <= cutoff_distr]
                    intervals = intervals[intervals > 9.99999999e-04]
                    if len(intervals) != 0:  #截断后可能导致没有interval    
                        neuron_intervals.extend(intervals)  #单个neuron所有trial的interval放到一个数组里
    
    plt.hist(neuron_intervals, bins=100, density=False, alpha=0.6,color=neuron_color)
    return neuron_intervals

def ISI(region_neurons_id,fr_filter,mice_type,mode,cutoff_distr=None):
    lambdas = []
    neuron_colors = plt.cm.hsv(np.linspace(0, 1, len(region_neurons_id)))
    print(neuron_colors)
    plt.figure(figsize=(10, 6))
    for i in range(len(region_neurons_id)):
        spike_times = singleneuron_spiketrain(region_neurons_id[i])
        # 每个neuron多个trial分别画histogram
        color = neuron_colors[i]
        neuron_intervals = ISI_sub(spike_times,fr_filter,cutoff_distr,mode,color)
        '''
        ## 画完当前neuron所有trial的hist之后，对当前neuron整体的ISI拟合指数分布
        if len(neuron_intervals) != 0 and cutoff_distr == None:
            params = expon.fit(neuron_intervals, floc=0)  # 固定起始点为0
            lambda_fit = 1 / params[1]  # 拟合的lambda值
            # 绘制拟合的指数分布曲线
            x = np.linspace(0, max(neuron_intervals), 100)
            plt.plot(x, expon.pdf(x, *params), 'b-', lw=0.2)
            lambdas.append(lambda_fit)  # 保存lambda值 each lambda represent a neurons firing strength in different run trials
        '''
        print(neuron_intervals)
        plt.xlim(-0.01,0.26) #add this when needed
        plt.xlabel('Inter-spike Interval (s)')
        plt.ylabel('Counts')
        plt.title(f'ISI Distribution_{region_name}')
        plt.text(0.95, 0.95, f'{mice_name}\n\nfiring filter: 1 spike/s\n\nhisto bin: 100\n\ndistr cutoff > {cutoff_distr}\n\n{mode} trials', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(fig_save_path+f"/{mode}_trials_{region_name}_cutoff_{cutoff_distr}.png",dpi=600,bbox_inches = 'tight')
        
def main(region_name,mice_name,fr_filter,mode,cutoff_distr):
    # get neuron ids
    ch['area_site'] = ch['area_site'].apply(literal_eval)
    site_id = ch[ch['area_name'] == region_name].iloc[0]['area_site']
    neurons = cluster_info[cluster_info['ch'].isin(site_id)]
    neuron_ids = np.array(neurons['cluster_id']).astype(int)
    print(neuron_ids)
    # 计算ISI分布
    ISI(neuron_ids,fr_filter,mice_name,mode,cutoff_distr)

region_name = 'Lobules IV-V'
main(region_name,mice_name,fr_filter=1,mode='before_pertur',cutoff_distr=0.25)
# fr_filter=1 滤掉fr小于1spike/s
# cutoff_distr=0.25代表截断ISI分布大于0.25s的, cutoff_distr=None代表不截断
# mode = 'before_reset' 'after_reset' 'before_pertur' 'after_pertur':

'''
    if cutoff_distr == None:
        plt.xlabel('Inter-spike Interval (s)')
        plt.ylabel('Probability Density')
        plt.title(f'ISI Distribution_{region_name}')
        # 画完所有的ISI分布后 计算平均 lambda 值
        lambda_mean = np.mean(lambdas)
        neuron_infig_num = len(lambdas)
        # 在图的右上角显示平均的lambda值
        plt.text(0.95, 0.95, f'{mice_name}\n\nfiring filter: 1 spike/s\n\nneurons num after filter: {neuron_infig_num}\n\nMean λ: {lambda_mean:.2f}\n\nhisto bin: 20\n\n{mode} trials', ha='right', va='top', transform=plt.gca().transAxes)
        # 插入子图表示lambda的分布
        left, bottom, width, height = 0.6,0.25,0.25,0.25
        plt.axes([left,bottom,width,height])
        plt.hist(lambdas, bins=50, density=False, alpha=0.6)
        plt.title(f'lambda distribution')
        plt.xlabel('lambda value')
        plt.ylabel('Prob.')
    else:
'''