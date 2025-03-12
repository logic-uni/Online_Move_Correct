"""
# coding: utf-8
@author: Yuhao Zhang
last updated : 11/04/2024
data from: Xinrong Tan
"""

import numpy as np
from scipy.stats import chi2_contingency

# 假设 data 是一个二维数组，每行是一个试验
def compute_transition_matrix(data, num_states):
    transition_matrix = np.zeros((num_states, num_states))
    
    # 统计转移频率
    for sequence in data:
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            transition_matrix[current_state, next_state] += 1

    # 计算转移概率
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums != 0)
    
    return transition_matrix

# 卡方检验函数
def chi_square_test(transition_matrix):
    # 期望矩阵假设每个状态转移的概率相等
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    expected = np.divide(row_sums, transition_matrix.shape[1])

    # 进行卡方检验
    chi2_stat, p_value, _, _ = chi2_contingency(transition_matrix, correction=False)
    
    return chi2_stat, p_value

# 示例使用
num_states = 3  # 假设有 3 个离散状态
data = np.array([[0, 1, 2, 1, 0], 
                 [1, 2, 0, 1, 2], 
                 [2, 1, 0, 2, 1]])

transition_matrix = compute_transition_matrix(data, num_states)
chi2_stat, p_value = chi_square_test(transition_matrix)

print("转移矩阵：")
print(transition_matrix)
print(f"卡方统计量: {chi2_stat}, p 值: {p_value}")
