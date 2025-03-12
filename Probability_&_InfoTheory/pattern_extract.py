"""
# coding: utf-8
@author: Yuhao Zhang
last updated: 10/04/2024
data from: Xinrong Tan
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import log
import math
import warnings
import scipy.io as sio
import pandas as pd
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)   

def Pattern_Entropy(data,id):

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

    #save to csv
    my_list = [[key, value] for key, value in sort_p.items()]
    with open('C:/Users/zyh20/Desktop/csv/-0.1to5WT.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(my_list)

    #plot
    x=list(sort_p.keys())
    y=list(sort_p.values())

    plt.bar(x, y)
    plt.title('Encoding pattern distribution, Neuron id %d'%id, fontsize=16)
    plt.xticks(x, rotation=90, fontsize=10)
    plt.yticks(fontsize=16)
    #plt.ylim(0,0.08)
    plt.ylabel("Probability of pattern", fontsize=16)
    plt.show()
    
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

def Pattern_Entropy_eachtrial(data): 

    # about bin 1 bit = 1 msec 
    # Statistics pattern all trials
    figsize=int(math.sqrt(len(data))+1)

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
        print(sort_p)

        # information entropy
        h=0
        for i in p:
            h = h - p[i]*log(p[i],2)
        print('Shannon Entropy=%f'%h)

        #plot
        plt.subplot(figsize,figsize,j+1)
        x=list(sort_p.keys())
        y=list(sort_p.values())

        plt.bar(x, y)
        plt.axis('off')
        
    plt.show()
    


def InfoPlot():
    x=['PC d:120','PC d:180','PC d:280','PC d:400','IPN d:1580','IPN d:1820','IPN d:1900','IPN d:1960']
    y=[2.3,3.3,3.6,2.8,0.5,0.5,0.3,0.2]

    plt.bar(x, y)
    plt.title('Quantities of information', fontsize=16)
    plt.xticks(x, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Shannon entropy", fontsize=16)
    plt.show()

def trialstart_pattern_extract():
    
    event_series=marker()
    trialstart=np.array([],dtype=float)
    for i in event_series: 
        trialstart=np.append(trialstart,i[0])

    binar_pert=binary_spiketrain(348,trialstart,0,0.2)
    
    Pattern_Entropy_eachtrial(binar_pert)

def reseton_pattern_extract():
    
    event_series=marker()
    reseton=np.array([],dtype=float)
    for i in event_series: 
        reseton=np.append(reseton,i[1])

    binar_pert=binary_spiketrain(348,reseton,-0.2,0)
    
    Pattern_Entropy_eachtrial(binar_pert)

def push_pattern_extract():
    
    event_series=marker()
    push=np.array([],dtype=float)
    for i in event_series: 
        push=np.append(push,i[3])

    binar_pert=binary_spiketrain(348,push,0,0.2)
    
    Pattern_Entropy_eachtrial(binar_pert)

def pert_pattern_extract():

    event_series=marker()
    pert=np.array([],dtype=float)
    for i in event_series:  
        if i[4]!=0:
            pert=np.append(pert,i[4])
    binar_pert=binary_spiketrain(348,pertmarker(),-0.2,0)
    
    Pattern_Entropy_eachtrial(binar_pert)

trialstart_pattern_extract()

#Pattern_Entropy(binary_spiketrain(348,marker()),348)