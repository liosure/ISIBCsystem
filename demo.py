from isibc_targetid import TargetIdentifier, plot_results
import numpy as np

def main():
    # 参数设置（与MATLAB demo保持一致）
    num_sources = 3  # 随机生成3个信源
    ti = TargetIdentifier(K=8, M=100, N=10, snr=10)
    
    # 生成随机入射角度（与MATLAB demo完全一致）
    np.random.seed(42)  # 固定随机种子确保可重复性
    theta_true = np.random.uniform(-0.25, 0.25, num_sources)
    
    # 生成雷达信号
    data = ti.generate_signal(theta_true, num_sources)
    
    # 执行目标识别
    result = ti.estimate_sources(data)
    
    # 准备可视化数据
    true_angles_deg = {f'BD{i}': np.arcsin(theta*2)*180/np.pi for i, theta in enumerate(theta_true)}
    plot_results(result['angles'], true_angles_deg, ti.snr)

if __name__ == "__main__":
    main()
