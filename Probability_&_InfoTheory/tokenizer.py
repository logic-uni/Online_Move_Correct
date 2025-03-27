import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
import time
import math

# 参数配置
min_len = 10    # 最小模式长度
max_len = 20    # 最大模式长度
min_support = 5  # 最小出现次数阈值（动态剪枝）
n_jobs = cpu_count()  # 使用全部CPU核心

file_path = '/home/zhangyuhao/Desktop/Result/Online_Move_Correct/Pattern_extract/'
data = np.load(file_path + '/20230511/Interposed nucleus_aft_per1.npy')

# -------------------------------------------------------------------
# 步骤1：生成高频模式字典（按信息熵降序排列）
# -------------------------------------------------------------------
def generate_substrings(binary_str):
    """生成非全零子序列"""
    substrings = []
    n = len(binary_str)
    for l in range(min_len, max_len + 1):
        for i in range(n - l + 1):
            token = binary_str[i:i+l]
            if '1' in token:
                substrings.append(token)
    return substrings

def process_row(row):
    binary_str = ''.join(row.astype(str))
    return generate_substrings(binary_str)

with Pool(n_jobs) as pool:
    all_substrings = pool.map(process_row, data)

counter = Counter()
for subs in all_substrings:
    counter.update(subs)
filtered_patterns = {k: v for k, v in counter.items() if v >= min_support}

# 计算模式的熵值
def calculate_entropy(pattern):
    """计算二进制模式的信息熵"""
    count_0 = pattern.count('0')
    count_1 = len(pattern) - count_0
    p0 = count_0 / len(pattern)
    p1 = count_1 / len(pattern)
    entropy = 0.0
    if p0 > 0:
        entropy -= p0 * math.log2(p0)
    if p1 > 0:
        entropy -= p1 * math.log2(p1)
    return entropy

# 按熵值降序 → 长度降序 → 出现次数降序排列
patterns_with_metadata = [
    (p, calculate_entropy(p), len(p), counter[p]) 
    for p in filtered_patterns
]
patterns_sorted = sorted(
    patterns_with_metadata,
    key=lambda x: (-x[1], -x[2], -x[3])
)

# 提取排序后的模式列表
sorted_patterns = [p[0] for p in patterns_sorted]

# -------------------------------------------------------------------
# 步骤2：定义替换函数（优先选择高熵模式）
# -------------------------------------------------------------------
def replace_row(row):
    binary_str = ''.join(row.astype(str))
    n = len(binary_str)
    covered = [False] * n
    selected = []
    i = 0
    while i < n:
        best_match = None
        # 遍历所有模式，寻找最高熵的可行匹配
        for pattern in sorted_patterns:
            p_len = len(pattern)
            end = i + p_len
            if end > n:
                continue
            if binary_str[i:end] == pattern and not any(covered[i:end]):
                best_match = pattern
                break  # 已排序，第一个匹配即最优
        if best_match:
            selected.append(best_match)
            # 标记覆盖位置
            for j in range(i, i + len(best_match)):
                covered[j] = True
            i += len(best_match)
        else:
            i += 1
    return selected

# -------------------------------------------------------------------
# 步骤3：并行替换并输出
# -------------------------------------------------------------------
start_time = time.time()
with Pool(n_jobs) as pool:
    replaced_matrix = pool.map(replace_row, data)

# 打印前5行结果（二进制模式）
print("替换后矩阵（前5行）:")
for i in range(5):
    row = replaced_matrix[i]
    print(f"行 {i+1}: 模式={row}")

# -------------------------------------------------------------------
# 步骤4：统计信息量最大的20个模式（排除单'1'）
# -------------------------------------------------------------------
all_patterns = []
for row in replaced_matrix:
    all_patterns.extend(row)

# 过滤仅含一个'1'的模式
filtered = [p for p in all_patterns if p.count('1') > 1]

# 统计频率
pattern_counter = Counter(filtered)

# 打印结果（按熵值排序的原始元组列表）
print("\n信息量最大的20个模式（按熵值排序）:")
print("格式: [模式] 熵值 | 长度 | 总出现次数 | 替换后出现次数")
for pattern_info in patterns_sorted[:20]:
    pattern = pattern_info[0]
    entropy = pattern_info[1]
    length = pattern_info[2]
    total_count = pattern_info[3]  # 原始数据中的总出现次数
    replaced_count = pattern_counter.get(pattern, 0)  # 替换后的出现次数
    print(f"{pattern}: 熵={entropy:.3f} bits | 长={length} | 原始出现={total_count} | 替换后出现={replaced_count}")

print(f"\n总耗时: {time.time() - start_time:.2f}秒")