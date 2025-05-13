import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

# 全局字体配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建字体属性对象（用于需要显式指定字体的地方）
chinese_font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')

def plot_results(estimated: dict, true_angles: dict, snr: float):
    """实现MATLAB的plotFunction可视化功能"""
    plt.figure(figsize=(16, 9))
    
    # 真实角度绘制

    angles_true = list(true_angles.values())
    indices = [i+1 for i, val in enumerate(angles_true) if val != []]
    angles_true = [val for val in angles_true if val != []]
    angles_true = np.array(angles_true) * 180 / np.pi  # 转换为度
    plt.scatter(indices, angles_true, c='b', marker='s', label='真实值', s=80, edgecolors='k')
    plt.xlim(1,len(true_angles)+1)
    plt.ylim(-90,90)
    plt.title('真实角度分布', fontproperties=chinese_font)
    plt.xlabel('信源编号', fontproperties=chinese_font)
    plt.ylabel('角度(度)', fontproperties=chinese_font)
    plt.xticks(range(len(true_angles)))
    plt.legend(prop=chinese_font)
    plt.grid(True)
    
    # 估计值绘制

    angles_est = list(estimated.values())
    indices_est = [i+1 for i, val in enumerate(angles_est) if (np.array(val).size > 0)]
    angles_est = [val for val in angles_est if (np.array(val).size > 0)]
    angles_est = np.array(angles_est) * 180 / np.pi  # 转换为度
    plt.scatter(indices_est, angles_est, facecolors='none',  marker='o', label='估计值', s=100, linewidths=2, edgecolors='r')
    plt.title(f'估计结果 (SNR={snr}dB)', fontproperties=chinese_font)
    plt.xlabel('信源编号', fontproperties=chinese_font)
    plt.ylabel('角度(度)', fontproperties=chinese_font)
    plt.xticks(range(len(estimated)+1))
    plt.legend(prop=chinese_font)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
