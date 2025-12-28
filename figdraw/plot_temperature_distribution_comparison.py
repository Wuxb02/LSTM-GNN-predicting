"""
温度分布对比绘图脚本（固定配置版）

作者: Claude Code
日期: 2025-12-22
描述: 对比两个模型在验证集上的温度预测分布，支持指定站点和温度范围

使用方法:
    1. 修改下面的配置区域
    2. 直接运行: python figdraw/plot_temperature_distribution_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde


# ==================== 配置区域 ====================
# 修改这里的配置来对比不同的模型

# 模型1的checkpoint目录（必填）
FOLDER1 = r'myGNN\checkpoints\GAT_SeparateEncoder_20251221_235912'

# 模型2的checkpoint目录（必填）
FOLDER2 = r'myGNN\checkpoints\GAT_SeparateEncoder_ADA'

# 模型1的显示名称（用于图例）
MODEL1_NAME = 'MSE Loss'

# 模型2的显示名称（用于图例）
MODEL2_NAME = 'ETW Loss'

# 指定站点ID（0-27），None表示所有站点平均
STATION = None  # None 或 0-27之间的整数

# 温度范围 (min, max)，None表示自动计算
TEMP_RANGE = (30, 40)  # None 或 (min, max) 元组

# 输出文件路径（None表示自动保存到figdraw/outputs/）
OUTPUT = None  # None 或字符串路径

# ================================================

fontsize=16
ticksize=14
# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
COLOR_LABEL = '#4472C4'  # 蓝色
COLOR_MODEL1 = '#4BAE4B'  # 橙色
COLOR_MODEL2 = '#FF8114' # 绿色

# 图表DPI
DPI = 300


def load_validation_data(checkpoint_dir, step_idx=0):
    """
    加载验证集数据

    Args:
        checkpoint_dir: checkpoint文件夹路径
        step_idx: 预测步长索引（默认0即step 1）

    Returns:
        val_predict: 预测值数组 [347, 28]
        val_label: 标签值数组 [347, 28]

    Raises:
        FileNotFoundError: 如果数据文件不存在
        ValueError: 如果数据形状不正确
    """
    # 确保使用绝对路径
    if not Path(checkpoint_dir).is_absolute():
        # 相对于脚本所在目录的父目录（项目根目录）
        project_root = Path(__file__).parent.parent
        checkpoint_path = project_root / checkpoint_dir
    else:
        checkpoint_path = Path(checkpoint_dir)

    # 检查文件是否存在
    pred_file = checkpoint_path / 'val_predict.npy'
    label_file = checkpoint_path / 'val_label.npy'

    if not pred_file.exists():
        raise FileNotFoundError(
            f"预测文件不存在: {pred_file}\n"
            f"请检查路径: {checkpoint_path}"
        )

    if not label_file.exists():
        raise FileNotFoundError(
            f"标签文件不存在: {label_file}"
        )

    # 加载数据
    val_predict = np.load(pred_file)
    val_label = np.load(label_file)

    # 检查数据形状
    if val_predict.ndim != 3:
        raise ValueError(
            f"预测数据形状错误: {val_predict.shape}，期望3维"
        )

    if val_label.ndim != 3:
        raise ValueError(
            f"标签数据形状错误: {val_label.shape}，期望3维"
        )

    # 提取指定步长的数据
    val_predict = val_predict[:, :, step_idx]  # [347, 28]
    val_label = val_label[:, :, step_idx]      # [347, 28]

    print(f"[OK] 加载数据成功: {checkpoint_path.name}")
    print(f"  数据形状: {val_predict.shape}")
    print(f"  温度范围: [{val_predict.min():.2f}, "
          f"{val_predict.max():.2f}] C")

    return val_predict, val_label


def prepare_plot_data(val_predict1, val_predict2, val_label,
                      station_id=None, temp_range=None):
    """
    准备绘图数据

    Args:
        val_predict1: 模型1预测值 [347, 28]
        val_predict2: 模型2预测值 [347, 28]
        val_label: 标签值 [347, 28]
        station_id: 可选，站点ID（None表示所有站点平均）
        temp_range: 可选，温度范围元组(min, max)，用于截断数据

    Returns:
        label_values: 标签温度值的1维数组（截断后）
        pred1_values: 模型1预测温度值的1维数组（截断后）
        pred2_values: 模型2预测温度值的1维数组（截断后）
    """
    if station_id is None:
        # 所有站点数据flatten
        label_values = val_label.flatten()
        pred1_values = val_predict1.flatten()
        pred2_values = val_predict2.flatten()
        print(f"\n数据准备: 使用所有28个站点的数据")
    else:
        # 仅提取指定站点数据
        if not (0 <= station_id < val_label.shape[1]):
            raise ValueError(
                f"站点ID错误: {station_id}，"
                f"应在0-{val_label.shape[1]-1}之间"
            )

        label_values = val_label[:, station_id]
        pred1_values = val_predict1[:, station_id]
        pred2_values = val_predict2[:, station_id]
        print(f"\n数据准备: 使用站点 {station_id} 的数据")

    print(f"  原始样本总数: {len(label_values)}")
    print(f"  原始标签温度范围: [{label_values.min():.2f}, "
          f"{label_values.max():.2f}] C")
    print(f"  原始模型1温度范围: [{pred1_values.min():.2f}, "
          f"{pred1_values.max():.2f}] C")
    print(f"  原始模型2温度范围: [{pred2_values.min():.2f}, "
          f"{pred2_values.max():.2f}] C")

    # 如果指定了温度范围，则截断数据
    if temp_range is not None:
        temp_min, temp_max = temp_range
        print(f"\n按温度范围截断数据: [{temp_min}, {temp_max}] C")

        # 找出标签在范围内的索引
        mask = (label_values >= temp_min) & (label_values <= temp_max)

        # 应用mask截断所有数组
        label_values = label_values[mask]
        pred1_values = pred1_values[mask]
        pred2_values = pred2_values[mask]

        print(f"  截断后样本数: {len(label_values)} "
              f"(保留 {len(label_values)/mask.size*100:.1f}%)")

        if len(label_values) == 0:
            raise ValueError(
                f"温度范围 [{temp_min}, {temp_max}] 内没有数据！"
            )

        # 打印截断后的统计信息
        print(f"  截断后标签范围: [{label_values.min():.2f}, "
              f"{label_values.max():.2f}] C")
        print(f"  截断后模型1范围: [{pred1_values.min():.2f}, "
              f"{pred1_values.max():.2f}] C")
        print(f"  截断后模型2范围: [{pred2_values.min():.2f}, "
              f"{pred2_values.max():.2f}] C")

    return label_values, pred1_values, pred2_values


def plot_temperature_distribution(label_values, pred1_values, pred2_values,
                                   model1_name, model2_name,
                                   temp_range, save_path):
    """
    绘制温度分布对比图（使用KDE平滑曲线）

    Args:
        label_values: 标签温度值
        pred1_values: 模型1预测温度值
        pred2_values: 模型2预测温度值
        model1_name: 模型1显示名称
        model2_name: 模型2显示名称
        temp_range: 温度范围元组(min, max)，None表示自动计算
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=DPI)

    # 确定x轴范围
    if temp_range:
        # 使用指定的温度范围
        xlim = (temp_range[0], temp_range[1])
    else:
        # 自动计算合理范围
        all_data = np.concatenate([label_values, pred1_values, pred2_values])
        temp_min = np.floor(all_data.min() / 5) * 5
        temp_max = np.ceil(all_data.max() / 5) * 5
        xlim = (temp_min, temp_max)

    # 生成平滑的x轴点
    x_smooth = np.linspace(xlim[0], xlim[1], 300)

    # 使用KDE生成平滑曲线
    kde_label = gaussian_kde(label_values, bw_method='scott')
    kde_pred1 = gaussian_kde(pred1_values, bw_method='scott')
    kde_pred2 = gaussian_kde(pred2_values, bw_method='scott')

    # 计算KDE密度值
    y_label = kde_label(x_smooth)
    y_pred1 = kde_pred1(x_smooth)
    y_pred2 = kde_pred2(x_smooth)

    # 绘制三条平滑曲线
    ax.plot(x_smooth, y_label, color=COLOR_LABEL, linewidth=2.5,
            alpha=0.8, label='Observation')
    ax.plot(x_smooth, y_pred1, color=COLOR_MODEL1, linewidth=2.5,
            alpha=0.8, label=model1_name)
    ax.plot(x_smooth, y_pred2, color=COLOR_MODEL2, linewidth=2.5,
            alpha=0.8, label=model2_name)

    # 设置坐标轴范围
    ax.set_xlim(xlim)

    # 设置标签和标题
    ax.set_xlabel('Temperature (°C)', fontsize=fontsize)
    ax.set_ylabel('Density', fontsize=fontsize)

    # 网格和图例
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='best', fontsize=fontsize)
    ax.tick_params(labelsize=ticksize)
    

    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[OK] 绘图保存成功: {save_path}")


def main():
    """
    主函数，使用顶部配置区域的参数
    """
    print("=" * 60)
    print("温度分布对比绘图脚本")
    print("=" * 60)

    # 显示配置
    print(f"\n模型1: {MODEL1_NAME}")
    print(f"  路径: {FOLDER1}")
    print(f"\n模型2: {MODEL2_NAME}")
    print(f"  路径: {FOLDER2}")
    print(f"\n站点设置: {'所有站点平均' if STATION is None else f'站点 {STATION}'}")

    if TEMP_RANGE:
        print(f"温度范围: [{TEMP_RANGE[0]}, {TEMP_RANGE[1]}] C")
    else:
        print("温度范围: 自动计算")

    # 加载数据
    print("\n" + "=" * 60)
    print("加载模型1数据")
    print("=" * 60)
    val_predict1, val_label1 = load_validation_data(FOLDER1)

    print("\n" + "=" * 60)
    print("加载模型2数据")
    print("=" * 60)
    val_predict2, val_label2 = load_validation_data(FOLDER2)

    # 验证标签一致性
    if not np.allclose(val_label1, val_label2):
        print("\n[WARNING] 两个模型的标签数据不完全一致")
        print("  使用模型1的标签数据")

    # 准备绘图数据
    print("\n" + "=" * 60)
    print("准备绘图数据")
    print("=" * 60)
    label_values, pred1_values, pred2_values = prepare_plot_data(
        val_predict1, val_predict2, val_label1, STATION, TEMP_RANGE
    )

    # 确定输出路径
    if OUTPUT:
        save_path = Path(OUTPUT)
    else:
        output_dir = Path(__file__).parent / 'result'
        output_dir.mkdir(exist_ok=True)

        # 构造文件名
        station_suffix = (f'_station{STATION}'
                          if STATION is not None
                          else '_all_stations')
        filename = (f'temp_dist_comparison_{MODEL1_NAME}_'
                   f'{MODEL2_NAME}{station_suffix}.png')
        save_path = output_dir / filename

    # 绘图
    print("\n" + "=" * 60)
    print("绘制温度分布图")
    print("=" * 60)
    plot_temperature_distribution(
        label_values, pred1_values, pred2_values,
        MODEL1_NAME, MODEL2_NAME,
        TEMP_RANGE, save_path
    )

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"\n输出文件: {save_path}")


if __name__ == '__main__':
    main()
