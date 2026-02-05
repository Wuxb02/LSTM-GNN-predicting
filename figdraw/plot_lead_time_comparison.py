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


def load_checkpoint_data(checkpoint_path, dataset='test'):
    """
    从checkpoint加载预测和标签数据

    Args:
        checkpoint_path (str): checkpoint目录路径
        dataset (str): 数据集类型，可选 'test', 'val', 'train'

    Returns:
        predict (np.ndarray): 预测值，形状 [num_days, 28, 5]
        label (np.ndarray): 真实值，形状 [num_days, 28, 5]
    """
    checkpoint_path = Path(checkpoint_path)

    # 检查路径是否存在
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint路径不存在: {checkpoint_path}")

    # 加载NPY文件
    predict_file = checkpoint_path / f'{dataset}_predict.npy'
    label_file = checkpoint_path / f'{dataset}_label.npy'

    if not predict_file.exists():
        raise FileNotFoundError(f"预测文件不存在: {predict_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"标签文件不存在: {label_file}")

    predict = np.load(predict_file)
    label = np.load(label_file)

    return predict, label


def calculate_errors_per_lead_time(predict, label, metric='RMSE'):
    """
    计算每个lead time的误差统计量（95%置信区间）

    Args:
        predict (np.ndarray): 预测值，形状 [num_days, 28, 5]
        label (np.ndarray): 真实值，形状 [num_days, 28, 5]
        metric (str): 误差指标，可选 'RMSE', 'MAE', 'BIAS'

    Returns:
        mean_errors (np.ndarray): 每个lead time的平均误差，形状 [5]
        lower_bounds (np.ndarray): 95%置信区间下界，形状 [5]
        upper_bounds (np.ndarray): 95%置信区间上界，形状 [5]
        all_metrics (dict): 包含所有指标的字典 {'RMSE': [...], 'MAE': [...], 'BIAS': [...], 'R2': [...]}
    """
    num_steps = predict.shape[2]  # 5
    mean_errors = []
    lower_bounds = []
    upper_bounds = []

    # 存储所有指标
    all_rmse = []
    all_mae = []
    all_bias = []
    all_r2 = []

    for step in range(num_steps):
        # 提取第step个预测步长的数据
        pred_step = predict[:, :, step]  # shape: [num_days, 28]
        label_step = label[:, :, step]   # shape: [num_days, 28]

        # 计算每天的误差（用于置信区间）
        if metric == 'RMSE':
            errors_per_day = np.sqrt(np.mean((pred_step - label_step)**2, axis=1))
        elif metric == 'MAE':
            errors_per_day = np.mean(np.abs(pred_step - label_step), axis=1)
        elif metric == 'BIAS':
            errors_per_day = np.mean(pred_step - label_step, axis=1)
        else:
            raise ValueError(
                f"不支持的误差指标: {metric}。"
                f"支持的指标: 'RMSE', 'MAE', 'BIAS'"
            )

        # 计算均值
        mean_error = np.mean(errors_per_day)
        mean_errors.append(mean_error)

        # 计算95%置信区间
        n = len(errors_per_day)
        sem = np.std(errors_per_day, ddof=1) / np.sqrt(n)  # 使用样本标准差
        margin = 1.96 * sem
        lower_bounds.append(mean_error - margin)
        upper_bounds.append(mean_error + margin)

        # 计算所有指标（用于详细输出）
        # RMSE
        rmse = np.sqrt(np.mean((pred_step - label_step)**2))
        all_rmse.append(rmse)

        # MAE
        mae = np.mean(np.abs(pred_step - label_step))
        all_mae.append(mae)

        # BIAS
        bias = np.mean(pred_step - label_step)
        all_bias.append(bias)

        # R²
        ss_res = np.sum((label_step - pred_step)**2)
        ss_tot = np.sum((label_step - np.mean(label_step))**2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot >= 1e-10 else 0.0
        all_r2.append(r2)

    all_metrics = {
        'RMSE': np.array(all_rmse),
        'MAE': np.array(all_mae),
        'BIAS': np.array(all_bias),
        'R2': np.array(all_r2)
    }

    return (np.array(mean_errors),
            np.array(lower_bounds),
            np.array(upper_bounds),
            all_metrics)


def generate_virtual_data(checkpoint_paths=None, metric='RMSE', dataset='test'):
    """
    生成数据（支持虚拟数据或从checkpoint读取）

    Args:
        checkpoint_paths (dict): 模型checkpoint路径字典，如:
            {
                'GAT': 'myGNN/checkpoints/GAT_20251222_135203',
                'LSTM': 'myGNN/checkpoints/LSTM_20251222_144549',
                ...
            }
            如果为None，则使用虚拟数据
        metric (str): 误差指标 ('RMSE', 'MAE', 'BIAS')
        dataset (str): 数据集类型 ('test', 'val', 'train')

    Returns:
        lead_times: Lead time数组 [1, 2, 3, 4, 5]
        errors_dict: 包含误差均值的字典
        lower_dict: 95%置信区间下界字典
        upper_dict: 95%置信区间上界字典
    """
    lead_times = np.array([1, 2, 3, 4, 5])

    if checkpoint_paths is None:
        # 使用原有的虚拟数据逻辑
        # 基于图表估计的数值生成虚拟数据
        # LSTM (蓝色实线+叉): 误差最大，增长最快
        errors_lstm = np.array([2.2614,
    2.8292,
    3.0667,
    3.1881,
    3.2881])

        # GAT (橙色虚线+方块): 误差次之
        errors_gat = np.array([2.7457,
    2.9111,
    3.2125,
    3.4331,
    3.6299])

        # GSAGE (绿色点划线+圆): 误差再次
        errors_gatlstm = np.array([2.2072,
    2.7648,
    3.0598,
    3.1952,
    3.2668])

        # GCN (紫色虚线+菱形): 误差介于GAT和GSAGE之间
        errors_IGNN = np.array([2.1528,
    2.7651,
    3.0125,
    3.1232,
    3.1859])

        errors_dict = {
            'GAT': errors_gat,
            'LSTM': errors_lstm,
            'GAT-LSTM': errors_gatlstm,
            'IGNN': errors_IGNN
        }
        lower_dict = None  # 虚拟数据不提供置信区间
        upper_dict = None
        all_metrics_dict = None  # 虚拟数据不提供详细指标

    else:
        # 从checkpoint读取实际数据
        print("正在加载模型数据...")
        errors_dict = {}
        lower_dict = {}
        upper_dict = {}
        all_metrics_dict = {}  # 新增：存储所有指标

        for model_name, checkpoint_path in checkpoint_paths.items():
            try:
                predict, label = load_checkpoint_data(checkpoint_path, dataset)
                mean_errors, lower_bounds, upper_bounds, all_metrics = calculate_errors_per_lead_time(
                    predict, label, metric
                )
                errors_dict[model_name] = mean_errors
                lower_dict[model_name] = lower_bounds
                upper_dict[model_name] = upper_bounds
                all_metrics_dict[model_name] = all_metrics  # 保存所有指标
                print(f"  - {model_name}: 加载完成 ({predict.shape[0]}天)")
            except Exception as e:
                print(f"  - {model_name}: 加载失败 - {e}")
                raise

    return lead_times, errors_dict, lower_dict, upper_dict, all_metrics_dict


def plot_lead_time_comparison(lead_times, errors_dict, save_path,
                               lower_dict=None, upper_dict=None, metric='RMSE'):
    """
    绘制Lead time对比图

    Args:
        lead_times: Lead time数组
        errors_dict: 误差字典
        save_path: 保存路径
        lower_dict: 95%置信区间下界字典（可选）
        upper_dict: 95%置信区间上界字典（可选）
        metric: 误差指标名称，用于Y轴标签
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 定义线型和标记
    line_styles = {
        'GAT': {'linestyle': '-', 'marker': 'x', 'linewidth': 2.5,
                 'markersize': 10, 'color': '#1f77b4'},
        'LSTM': {'linestyle': '--', 'marker': 's', 'linewidth': 2.5,
                'markersize': 8, 'color': '#ff7f0e'},
        'GAT-LSTM': {'linestyle': '--', 'marker': 'D', 'linewidth': 2.5,
                     'markersize': 7, 'color': '#9467bd'},
        'IGNN': {'linestyle': '-.', 'marker': 'o', 'linewidth': 2.5,
                  'markersize': 8, 'color': '#d62728'}
    }

    # 先绘制95%置信区间阴影（在曲线下方）
    if lower_dict is not None and upper_dict is not None:
        for model_name, errors in errors_dict.items():
            if model_name in lower_dict and model_name in upper_dict:
                style = line_styles[model_name]
                lower = lower_dict[model_name]
                upper = upper_dict[model_name]
                ax.fill_between(
                    lead_times,
                    lower,
                    upper,
                    color=style['color'],
                    alpha=0.1,
                    linewidth=0
                )

    # 绘制曲线（在阴影上方）
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

    # 根据metric动态设置Y轴标签
    ylabel_map = {
        'RMSE': 'RMSE (°C)',
        'MAE': 'MAE (°C)',
        'BIAS': 'BIAS (°C)'
    }
    ax.set_ylabel(ylabel_map.get(metric, 'Error (°C)'), fontsize=fontsize)

    # 设置刻度
    ax.set_xticks([1, 2, 3, 4, 5])
    # ax.set_ylim(0.5, 1.4)
    # ax.set_yticks(np.arange(0.5, 1.5, 0.2))

    # 设置刻度字体大小
    ax.tick_params(axis='both', labelsize=fontsize)

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # 添加图例
    legend = ax.legend(fontsize=fontsize, loc='lower right', frameon=True,
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


def main(checkpoint_paths=None, metric='RMSE', dataset='test'):
    """
    主函数

    Args:
        checkpoint_paths (dict): 模型checkpoint路径字典，如:
            {
                'GAT': 'myGNN/checkpoints/GAT_20251222_135203',
                'LSTM': 'myGNN/checkpoints/LSTM_20251222_144549',
                ...
            }
            如果为None，则使用虚拟数据
        metric (str): 误差指标 ('RMSE', 'MAE', 'BIAS')
        dataset (str): 数据集类型 ('test', 'val', 'train')
    """
    # 设置路径
    result_dir = Path(__file__).parent / 'result'
    result_dir.mkdir(exist_ok=True)

    # 根据是否提供checkpoint路径设置文件名
    if checkpoint_paths is None:
        save_path = result_dir / 'comparison.png'
        print("使用虚拟数据...")
    else:
        save_path = result_dir / f'comparison_{metric}_{dataset}.png'
        print(f"从checkpoint读取数据 (metric={metric}, dataset={dataset})...")

    # 生成数据
    lead_times, errors_dict, lower_dict, upper_dict, all_metrics_dict = generate_virtual_data(
        checkpoint_paths, metric, dataset
    )

    # 打印详细指标表格
    if all_metrics_dict is not None:
        print("\n" + "="*80)
        print("详细指标表格 (Detailed Metrics Table)")
        print("="*80)

        for model_name in errors_dict.keys():
            metrics = all_metrics_dict[model_name]
            print(f"\n{model_name}:")
            print("-" * 80)
            print(f"{'Metric':<15} {'1':<12} {'2':<12} {'3':<12} {'4':<12} {'5':<12}")
            print("-" * 80)
            print(f"{'RMSE (°C)':<15} {metrics['RMSE'][0]:<12.2f} {metrics['RMSE'][1]:<12.2f} "
                  f"{metrics['RMSE'][2]:<12.2f} {metrics['RMSE'][3]:<12.2f} {metrics['RMSE'][4]:<12.2f}")
            print(f"{'MAE (°C)':<15} {metrics['MAE'][0]:<12.2f} {metrics['MAE'][1]:<12.2f} "
                  f"{metrics['MAE'][2]:<12.2f} {metrics['MAE'][3]:<12.2f} {metrics['MAE'][4]:<12.2f}")
            print(f"{'Bias (°C)':<15} {metrics['BIAS'][0]:<12.2f} {metrics['BIAS'][1]:<12.2f} "
                  f"{metrics['BIAS'][2]:<12.2f} {metrics['BIAS'][3]:<12.2f} {metrics['BIAS'][4]:<12.2f}")
            print(f"{'R2':<15} {metrics['R2'][0]:<12.2f} {metrics['R2'][1]:<12.2f} "
                  f"{metrics['R2'][2]:<12.2f} {metrics['R2'][3]:<12.2f} {metrics['R2'][4]:<12.2f}")

        print("\n" + "="*80)

    # 打印数据摘要
    print("\n数据摘要:")
    print(f"Lead times: {lead_times}")
    if lower_dict is not None:
        print("置信区间: 95% CI")
    for model_name, errors in errors_dict.items():
        if lower_dict is not None and model_name in lower_dict:
            lower = lower_dict[model_name]
            upper = upper_dict[model_name]
            ci_width = upper - lower
            print(f"{model_name:15s}: {errors}")
            print(f"{'':15s}  95% CI: [{lower[0]:.3f}, {upper[0]:.3f}] ... "
                  f"[{lower[-1]:.3f}, {upper[-1]:.3f}]")
            print(f"{'':15s}  CI宽度: {ci_width}")
        else:
            print(f"{model_name:15s}: {errors}")

    # 绘制图表
    print("\n正在绘制图表...")
    plot_lead_time_comparison(lead_times, errors_dict, save_path,
                              lower_dict, upper_dict, metric)

    # 统计信息
    print(f"\n模型性能对比 (平均{metric}):")
    for model_name, errors in errors_dict.items():
        avg_error = np.mean(errors)
        if lower_dict is not None and model_name in lower_dict:
            avg_ci_width = np.mean(upper_dict[model_name] - lower_dict[model_name])
            print(f"{model_name:15s}: {avg_error:.4f} (95% CI宽度: {avg_ci_width:.4f})")
        else:
            print(f"{model_name:15s}: {avg_error:.4f}")

    print("\n[完成] 复现完成!")


if __name__ == '__main__':
    # ========== 使用示例 ==========

    # 示例1: 使用虚拟数据（原有功能，向后兼容）
    # main()

    # 示例2: 从checkpoint读取实际数据
    checkpoint_paths = {
        'GAT': r'..\myGNN\checkpoints\GAT_Pure_20251222_135203',
        'LSTM': r'..\myGNN\checkpoints\LSTM_20251222_144549',
        'GAT-LSTM': r'..\myGNN\checkpoints\GAT_LSTM_20251222_145359',
        'IGNN': r'..\myGNN\checkpoints\GAT_SeparateEncoder_20251221_235912'
    }

    # 使用RMSE指标（默认）
    main(checkpoint_paths, metric='RMSE', dataset='test')

    # 也可以使用其他指标
    # main(checkpoint_paths, metric='MAE', dataset='test')
    # main(checkpoint_paths, metric='BIAS', dataset='test')

    # 也可以使用验证集或训练集
    # main(checkpoint_paths, metric='RMSE', dataset='val')
    # main(checkpoint_paths, metric='RMSE', dataset='train')
