"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 10/07/2024
data from: Xinrong Tan
data collected: 05/11/2023
spikeGLX sample rate : 10593.220339
"""

import numpy as np
import pandas as pd
resample_rate = 1059.321823

main_path = '/data1/zhangyuhao/xinrong_data/NP1/mice_3/20230522/cage1-3-L-2_g0'
marker_path = main_path+'/resample_behavioral_data.csv'
save_path = main_path+'/event_series.csv'

def get_rising_time(array):
    index = np.array([],dtype=int)
    #Binarization
    for i in range(len(array)):
        if array[i]>2:
            array[i]=1
        else:
            array[i]=0
    for m in range(len(array)):
        #Rising edge detection
        if array[m]-array[m-1]==1:
            index = np.append(index,m)  
    return index

def gettime(array,start_index,end_index):
    this_trial = array[start_index:end_index]
    time = np.array([0,0],dtype=int)
    #Binarization
    for i in range(len(this_trial)):
        if this_trial[i]>2:
            this_trial[i]=1
        else:
            this_trial[i]=0
    for m in range(len(this_trial)):
        #Rising edge detection
        if this_trial[m]-this_trial[m-1]==1:
            time[0] = (m+start_index)
        #Falling edge detection
        if this_trial[m-1]-this_trial[m]==1:
            time[1] = (m+start_index)
    return time

def Behav2eventser():
    data = pd.read_csv(marker_path)
    trial_index = get_rising_time(data['Trial_initiation_signal'].to_numpy())   ##这里只能提取上升沿，不能提取下降沿，因为这里是一个冲击
    rocker_reset=data['Rocker_initialization'].to_numpy()
    behavior_process=data['Behavior_process'].to_numpy()
    Resistance_marker=data['Resistance_marker'].to_numpy()
    event_series=np.zeros((len(trial_index)-1, 10))  #最后一个trial程序上不好处理，扔掉不要了
    reward=data['Reward_signal'].to_numpy()
    for trial_num in range(0,len(trial_index)-1): #最后一个trial程序上不好处理，扔掉不要了
        # reorganize each trial,each row represent a trial
        # each trial contains 10 time point: trial_start, reset_on, reset_off, push_on, pert_on, pert_off, push_off, reward_on, reward_off, trial_end
        start_index = trial_index[trial_num]
        end_index = trial_index[trial_num+1]
        reset_onoff = gettime(rocker_reset,start_index,end_index)
        push_onoff = gettime(behavior_process,start_index,end_index)
        pert_onoff = gettime(Resistance_marker,push_onoff[0],push_onoff[1])  ##perturbation必须限制在推杆期间，因为原始程序可能使得pert时间在 推杆前 或 推杆到达底部后
        reward_onoff = gettime(reward,push_onoff[1],end_index)
        
        event_series[trial_num,0] = start_index/resample_rate
        event_series[trial_num,1] = reset_onoff[0]/resample_rate
        event_series[trial_num,2] = reset_onoff[1]/resample_rate
        event_series[trial_num,3] = push_onoff[0]/resample_rate
        event_series[trial_num,4] = pert_onoff[0]/resample_rate
        event_series[trial_num,5] = pert_onoff[1]/resample_rate
        event_series[trial_num,6] = push_onoff[1]/resample_rate
        event_series[trial_num,7] = reward_onoff[0]/resample_rate
        event_series[trial_num,8] = reward_onoff[1]/resample_rate
        event_series[trial_num,9] = end_index/resample_rate

    event_series = event_series.astype(float)
    df=pd.DataFrame(event_series,columns = ['trial_start', 'reset_on', 'reset_off', 'push_on', 'pert_on', 'pert_off', 'push_off', 'reward_on', 'reward_off','trial_end'])
    pd.DataFrame(df).to_csv(save_path)

Behav2eventser()

'''
old code
useless

pertmarker_path = main_path+'/events-rocker-perturbation.csv'
def pertmarker():
    data = pd.read_csv(pertmarker_path)
    pertmarker=data['pertur_marker'].to_numpy()
    return pertmarker

j=m=n=0
for i in range(trial_times-1):  
    event_series[i,1] = reset_onoff[j]    # reset_on
    event_series[i,2] = reset_onoff[j+1]  # reset_off
    event_series[i,3] = push_onoff[j]     # push_on
    event_series[i,6] = push_onoff[j+1]   # push_off
    j=j+2  # event_series 被填满后就自动退出循环了,因此不用设置j上限
    #遍历reward数组,如果在trial区间内,就存入,如果不在,就记0 
    #遍历有效性证明: 如果reward_on不在,则reward_off一定不在,如果reward_on在,则reward_off一定在
    while event_series[i,0]<reward_onoff[m] and reward_onoff[m]<event_series[i+1,0] and m<len(reward_onoff)-2:
        event_series[i,7] = reward_onoff[m]    # reward_on
        event_series[i,8] = reward_onoff[m+1]   # reward_off
        m=m+2
    while event_series[i,0]<pert_onoff[n] and pert_on[n]<event_series[i+1,0] and n<len(pert_on)-1:
        event_series[i,4] = pert_on[n]    # pert_on
        event_series[i,5] = pert_on[n]+0.15   # pert_off
        n=n+1
'''