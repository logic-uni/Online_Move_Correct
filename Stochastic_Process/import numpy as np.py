import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
save_path = f'/home/zhangyuhao/Desktop/Result/Online_Move_Correct/time_series'
def plot_similarity_heatmap(sim_matrix, title="Time Similarity Matrix"):
    """
    绘制相似性矩阵热图
    :param sim_matrix: 相似性矩阵，shape=(n, n)
    :param title: 图表标题
    """
    plt.figure(figsize=(12, 10))
    
    # 创建热图（关闭默认颜色条，后续手动添加）
    ax = sns.heatmap(
        sim_matrix,
        cmap='viridis',          # 使用清晰的颜色映射
        vmin=0, vmax=1,          # 固定颜色范围
        square=True,             # 保持单元格为正方形
        cbar_kws={'shrink': 0.8},# 调整颜色条大小
        xticklabels=False,       # 隐藏x轴标签
        yticklabels=False        # 隐藏y轴标签
    )
    
    # 添加颜色条标签
    cbar = ax.collections[0].colorbar
    cbar.set_label('Similarity Score', rotation=270, labelpad=15)
    
    # 设置标题和坐标轴标签
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Time Series Index", labelpad=15)
    plt.ylabel("Time Series Index", labelpad=15)
    
    # 优化布局显示
    plt.tight_layout()
    plt.savefig(save_path+f"/test.png",dpi=600,bbox_inches = 'tight')


# ----------------- 示例用法 -------------------
if __name__ == "__main__":
    # 生成示例相似性矩阵（替换为实际数据）
    np.random.seed(42)
    demo_sim = np.random.rand(20, 20) * 0.8
    demo_sim = (demo_sim + demo_sim.T) / 2  # 保证对称
    np.fill_diagonal(demo_sim, 1)          # 对角线设为1
    print(demo_sim)
    # 绘制热图
    plot_similarity_heatmap(demo_sim, "Demo Similarity Matrix")