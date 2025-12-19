"""
复现Lead time与误差对比图
作者: Claude Code
日期: 2025-12-20
描述: 使用虚拟数据复现LSTM、GAT、GCN、GSAGE、Hpyer-GSAGE五种模型随预测时长变化的误差曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证可复现
np.random.seed(42)


def generate_virtual_data():
    """
    生成虚拟数据

    Returns:
        lead_times: Lead time数组 [1, 2, 3, 4, 5, 6] 小时
        errors_dict: 包含5个模型误差数据的字典
    """
    lead_times = np.array([1, 2, 3, 4, 5, 6])

    # 基于图表估计的数值生成虚拟数据
    # LSTM (蓝色实线+叉): 误差最大，增长最快
    errors_lstm = np.array([0.65, 0.88, 1.02, 1.15, 1.24, 1.32])

    # GAT (橙色虚线+方块): 误差次之
    errors_gat = np.array([0.62, 0.83, 0.97, 1.10, 1.19, 1.27])

    # GSAGE (绿色点划线+圆): 误差再次
    errors_gsage = np.array([0.58, 0.78, 0.92, 1.04, 1.13, 1.21])

    # Hpyer-GSAGE (红色点划线+圆): 误差最小
    errors_hpyer_gsage = np.array([0.55, 0.74, 0.87, 0.99, 1.08, 1.16])

    # GCN (紫色虚线+菱形): 误差介于GAT和GSAGE之间
    errors_gcn = np.array([0.60, 0.80, 0.94, 1.07, 1.16, 1.24])

    # 添加少量随机噪声使曲线更真实
    noise_scale = 0.01
    errors_lstm += np.random.normal(0, noise_scale, len(lead_times))
    errors_gat += np.random.normal(0, noise_scale, len(lead_times))
    errors_gsage += np.random.normal(0, noise_scale, len(lead_times))
    errors_hpyer_gsage += np.random.normal(0, noise_scale, len(lead_times))
    errors_gcn += np.random.normal(0, noise_scale, len(lead_times))

    errors_dict = {
        'LSTM': errors_lstm,
        'GAT': errors_gat,
        'GAT-LSTM': errors_gcn,
        'XGBoost': errors_gsage,
        'myGNN': errors_hpyer_gsage
    }

    return lead_times, errors_dict


def plot_lead_time_comparison(lead_times, errors_dict, save_path):
    """
    绘制Lead time对比图

    Args:
        lead_times: Lead time数组
        errors_dict: 误差字典
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 定义线型和标记
    line_styles = {
        'LSTM': {'linestyle': '-', 'marker': 'x', 'linewidth': 2.5,
                 'markersize': 10, 'color': '#1f77b4'},
        'GAT': {'linestyle': '--', 'marker': 's', 'linewidth': 2.5,
                'markersize': 8, 'color': '#ff7f0e'},
        'GAT-LSTM': {'linestyle': '--', 'marker': 'D', 'linewidth': 2.5,
                     'markersize': 7, 'color': '#9467bd'},
        'XGBoost': {'linestyle': '-.', 'marker': 'o', 'linewidth': 2.5,
                    'markersize': 8, 'color': '#2ca02c'},
        'myGNN': {'linestyle': '-.', 'marker': 'o', 'linewidth': 2.5,
                  'markersize': 8, 'color': '#d62728'}
    }

    # 绘制五条曲线
    for model_name, errors in errors_dict.items():
        style = line_styles[model_name]
        ax.plot(lead_times, errors,
                label=model_name,
                linestyle=style['linestyle'],
                marker=style['marker'],
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                color=style['color'],
                markeredgewidth=2,
                markerfacecolor=style['color'] if model_name != 'LSTM' else 'none',
                markeredgecolor=style['color'])

    fontsize = 16
    # 设置坐标轴
    ax.set_xlabel('Lead time (days)', fontsize=fontsize)
    ax.set_ylabel('MAE (°C)', fontsize=fontsize)

    # 设置刻度
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_ylim(0.5, 1.4)
    ax.set_yticks(np.arange(0.5, 1.5, 0.2))

    # 设置刻度字体大小
    ax.tick_params(axis='both', labelsize=fontsize)

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # 添加图例
    legend = ax.legend(fontsize=fontsize, loc='upper left', frameon=True,
                      fancybox=False)


    # # 设置边框
    # for spine in ax.spines.values():
    #     spine.set_linewidth(1.5)
    #     spine.set_color('black')

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
               )
    print(f"[完成] 图表已保存至: {save_path}")

    # 关闭图表
    plt.close()


def main():
    """主函数"""
    # 设置路径
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(exist_ok=True)
    save_path = result_dir / 'lead_time_comparison.png'

    # 生成虚拟数据
    print("正在生成虚拟数据...")
    lead_times, errors_dict = generate_virtual_data()

    # 打印数据摘要
    print("\n数据摘要:")
    print(f"Lead times: {lead_times}")
    for model_name, errors in errors_dict.items():
        print(f"{model_name:15s}: {errors}")

    # 绘制图表
    print("\n正在绘制图表...")
    plot_lead_time_comparison(lead_times, errors_dict, save_path)

    # 统计信息
    print("\n模型性能对比 (平均误差):")
    for model_name, errors in errors_dict.items():
        avg_error = np.mean(errors)
        print(f"{model_name:15s}: {avg_error:.4f}")

    print("\n[完成] 复现完成!")


if __name__ == '__main__':
    main()
