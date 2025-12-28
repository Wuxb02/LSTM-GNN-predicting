"""
模型对比可视化脚本

用于对比两个GNN模型在验证集上的预测结果。

功能:
- 加载两个checkpoint的验证集数据
- 支持指定天数范围
- 支持单个站点或所有站点平均
- 生成三条曲线对比图 (Ground Truth + Model 1 + Model 2)

作者: GNN气温预测项目
日期: 2025
"""

import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')

fontsize = 16
ticksize = 14


@dataclass
class ComparisonConfig:
    """对比脚本配置类"""
    # 模型路径
    model1_dir: str
    model2_dir: str
    model1_name: str = "Model 1"  # 图例显示名称
    model2_name: str = "Model 2"

    # 数据范围
    day_start: int = 0
    day_end: Optional[int] = None  # None表示到最后
    station_id: Optional[int] = None  # None表示所有站点平均
    pred_step: int = 0  # 预测步长索引

    # 输出配置
    output_path: str = 'figdraw/result/model_comparison.png'
    dpi: int = 300
    figsize: Tuple[int, int] = (12, 10)

    # 绘图样式
    show_grid: bool = True
    grid_alpha: float = 0.3
    line_width: float = 2.5
    marker_size: int = 3


def load_validation_data(checkpoint_dir: str) -> Dict:
    """
    加载验证集数据并进行完整性检查

    Args:
        checkpoint_dir: checkpoint目录路径

    Returns:
        dict: {
            'predictions': np.ndarray [num_samples, num_stations, pred_len],
            'labels': np.ndarray [num_samples, num_stations, pred_len],
            'time': np.ndarray [num_samples] (可选),
            'shape_info': dict 数据形状信息
        }

    Raises:
        FileNotFoundError: 必需文件不存在
        ValueError: 数据形状不匹配
    """
    ckpt_path = Path(checkpoint_dir)

    # 检查目录存在性
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")

    # 加载必需文件
    pred_file = ckpt_path / 'val_predict.npy'
    label_file = ckpt_path / 'val_label.npy'
    time_file = ckpt_path / 'val_time.npy'

    if not pred_file.exists():
        raise FileNotFoundError(f"缺失预测文件: {pred_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"缺失标签文件: {label_file}")

    predictions = np.load(pred_file)
    labels = np.load(label_file)

    # 数据形状验证
    if predictions.shape != labels.shape:
        raise ValueError(
            f"预测和标签形状不匹配: "
            f"predictions={predictions.shape}, labels={labels.shape}"
        )

    # 可选加载时间信息
    time_data = None
    if time_file.exists():
        time_data = np.load(time_file)

    return {
        'predictions': predictions,
        'labels': labels,
        'time': time_data,
        'shape_info': {
            'num_samples': predictions.shape[0],
            'num_stations': predictions.shape[1],
            'pred_len': predictions.shape[2]
        }
    }


def extract_predictions(
    data: Dict,
    day_start: int = 0,
    day_end: Optional[int] = None,
    pred_step: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    提取指定天数范围和预测步长的数据

    Args:
        data: load_validation_data()返回的数据字典
        day_start: 起始天数索引（包含）
        day_end: 结束天数索引（不包含，None表示到最后）
        pred_step: 预测步长索引

    Returns:
        predictions_slice: [num_days, num_stations]
        labels_slice: [num_days, num_stations]

    Raises:
        ValueError: 索引超出边界
    """
    predictions = data['predictions']
    labels = data['labels']
    num_samples = data['shape_info']['num_samples']
    pred_len = data['shape_info']['pred_len']

    # 验证pred_step
    if pred_step < 0 or pred_step >= pred_len:
        raise ValueError(
            f"pred_step={pred_step} 超出范围 [0, {pred_len-1}]"
        )

    # 处理day_end
    if day_end is None:
        day_end = num_samples

    # 验证天数范围
    if day_start < 0 or day_start >= num_samples:
        raise ValueError(
            f"day_start={day_start} 超出范围 [0, {num_samples-1}]"
        )
    if day_end <= day_start or day_end > num_samples:
        raise ValueError(
            f"day_end={day_end} 无效，应满足 {day_start} < day_end <= {num_samples}"
        )

    # 提取数据 [num_days, num_stations]
    pred_slice = predictions[day_start:day_end, :, pred_step]
    label_slice = labels[day_start:day_end, :, pred_step]

    return pred_slice, label_slice


def compute_station_average(
    data: np.ndarray,
    station_id: Optional[int] = None
) -> np.ndarray:
    """
    计算指定站点或所有站点的平均值

    Args:
        data: [num_days, num_stations] 数据数组
        station_id: None=所有站点平均，整数=指定站点

    Returns:
        result: [num_days] 一维数组

    Raises:
        ValueError: station_id超出范围
    """
    num_stations = data.shape[1]

    if station_id is None:
        # 所有站点平均
        return np.mean(data, axis=1)  # [num_days]
    else:
        # 指定站点
        if station_id < 0 or station_id >= num_stations:
            raise ValueError(
                f"station_id={station_id} 超出范围 [0, {num_stations-1}]"
            )
        return data[:, station_id]  # [num_days]


def plot_comparison(
    config: ComparisonConfig,
    labels: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    day_start: int
) -> None:
    """
    绘制三条曲线对比图（简洁模式）

    Args:
        config: 配置对象
        labels: [num_days] 真实标签
        pred1: [num_days] 模型1预测
        pred2: [num_days] 模型2预测
        day_start: 起始天数索引（用于x轴标签）
    """
    num_days = len(labels)
    x_axis = np.arange(day_start, day_start + num_days)
    # 创建图表
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    plt.axhline(y=35,
                linestyle='--',
                linewidth=config.line_width,
                color='black',
                alpha=0.3
                )
    # 绘制三条曲线（参考用户图片样式）
    # Ground Truth: 黑色圆圈
    ax.plot(
        x_axis, labels,
        color='#4472C4',
        linestyle='-',
        linewidth=config.line_width,
        # marker='o',
        markersize=config.marker_size,
        label='Observation',
        alpha=0.9
    )

    # Model 1: 蓝色三角形
    ax.plot(
        x_axis, pred1,
        color='#4BAE4B',  # matplotlib默认蓝
        linestyle='-',
        linewidth=config.line_width,
        # marker='^',
        markersize=config.marker_size,
        label=config.model1_name,
        alpha=0.85
    )

    # Model 2: 红色方块
    ax.plot(
        x_axis, pred2,
        color='#FF8114',  # matplotlib默认红
        linestyle='-',
        linewidth=config.line_width,
        # marker='s',
        markersize=config.marker_size,
        label=config.model2_name,
        alpha=0.85
    )

    # 样式设置
    ax.set_xlabel('Day of year', fontsize=fontsize)
    ax.set_ylabel('Temperature (°C)', fontsize=fontsize)

    # # 动态标题
    # station_text = (
    #     f'All Stations Average' if config.station_id is None
    #     else f'Station {config.station_id}'
    # )
    # ax.set_title(
    #     f'Model Comparison - {station_text} (Step {config.pred_step + 1})',
    #     fontsize=14,
    #     fontweight='bold',
    #     pad=15
    # )

    # 图例
    ax.legend(
        loc='best',
        frameon=True,
        framealpha=0.9,
        fontsize=fontsize,
        edgecolor='gray'
    )

    # 网格
    if config.show_grid:
        ax.grid(True, alpha=config.grid_alpha, linestyle='--', linewidth=0.5)

    # 坐标轴样式
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=ticksize)

    # 保存
    plt.tight_layout()
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()

    print(f"[OK] Comparison plot saved to: {output_path}")


def main():
    """主函数：完整的执行流程"""

    # ============ 配置区域 ============
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent.parent

    config = ComparisonConfig(
        # 模型路径（使用绝对路径）
        model1_dir=str(script_dir / 'myGNN' / 'checkpoints' /
                       'GAT_SeparateEncoder_20251221_235912'),
        model2_dir=str(script_dir / 'myGNN' / 'checkpoints' /
                       'GAT_SeparateEncoder_ADA'),
        model1_name='MSE Loss',
        model2_name='ETW Loss',

        # 数据范围
        day_start=150,
        day_end=250,  # None表示到最后
        station_id=6,  # None表示所有站点平均
        pred_step=0,

        # 输出配置
        output_path=str(script_dir / 'figdraw' /
                        'result' / 'model_comparison.png'),
        dpi=300,
        figsize=(12, 6)
    )
    # ==================================

    print("=" * 60)
    print("GNN模型对比脚本")
    print("=" * 60)

    try:
        # Step 1: 加载两个模型的数据
        print(f"\n[1/4] 加载模型数据...")
        print(f"  - Model 1: {config.model1_dir}")
        data1 = load_validation_data(config.model1_dir)
        print(f"    [OK] Data shape: {data1['predictions'].shape}")

        print(f"  - Model 2: {config.model2_dir}")
        data2 = load_validation_data(config.model2_dir)
        print(f"    [OK] Data shape: {data2['predictions'].shape}")

        # 验证两个模型数据形状一致
        if data1['predictions'].shape != data2['predictions'].shape:
            raise ValueError(
                f"两个模型数据形状不一致: "
                f"{data1['predictions'].shape} vs {data2['predictions'].shape}"
            )

        # Step 2: 提取指定范围数据
        print(f"\n[2/4] 提取数据范围...")
        print(f"  - 天数范围: [{config.day_start}, {config.day_end or 'end'})")
        print(f"  - 预测步长: {config.pred_step}")

        pred1, labels = extract_predictions(
            data1, config.day_start, config.day_end, config.pred_step
        )
        pred2, _ = extract_predictions(
            data2, config.day_start, config.day_end, config.pred_step
        )
        print(f"    [OK] Extracted shape: {pred1.shape}")

        # Step 3: 计算站点平均
        print(f"\n[3/4] Computing station data...")
        station_text = (
            "All stations average" if config.station_id is None
            else f"Station {config.station_id}"
        )
        print(f"  - Selection: {station_text}")

        labels_avg = compute_station_average(labels, config.station_id)
        pred1_avg = compute_station_average(pred1, config.station_id)
        pred2_avg = compute_station_average(pred2, config.station_id)
        print(f"    [OK] Final shape: {labels_avg.shape}")

        # Step 4: 绘制对比图
        print(f"\n[4/4] 绘制对比图...")
        plot_comparison(
            config, labels_avg, pred1_avg, pred2_avg, config.day_start
        )

        # 输出统计信息
        print("\n" + "=" * 60)
        print("Statistics:")
        print(f"  - Comparison days: {len(labels_avg)}")
        print(
            f"  - Ground truth range: [{labels_avg.min():.2f}, {labels_avg.max():.2f}] C")
        print(f"  - {config.model1_name} RMSE: "
              f"{np.sqrt(np.mean((pred1_avg - labels_avg)**2)):.4f} C")
        print(f"  - {config.model2_name} RMSE: "
              f"{np.sqrt(np.mean((pred2_avg - labels_avg)**2)):.4f} C")
        print("=" * 60)
        print("\n[OK] Comparison completed!")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("  Please check if checkpoint paths are correct")
    except ValueError as e:
        print(f"\n[ERROR] {e}")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
