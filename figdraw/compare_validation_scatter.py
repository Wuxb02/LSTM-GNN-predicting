"""
验证集散点图对比脚本

用于对比两个GNN模型在验证集上的预测效果。

功能:
- 加载两个checkpoint的验证集数据
- 将3D数据转换为1D数组用于散点图
- 使用test.py的绘图代码绘制散点图对比
- 输出统计信息（RMSE、高温样本数量等）

作者: GNN气温预测项目
日期: 2025
"""

from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 16

@dataclass
class ComparisonConfig:
    """对比脚本配置类"""
    # 模型路径
    model1_dir: str
    model2_dir: str
    model1_name: str = "Model 1"
    model2_name: str = "Model 2"

    # 数据转换配置
    pred_step: int = 0  # 预测步长索引（0-4对应第1-5天）

    # 绘图配置
    output_path: str = 'figdraw/result/validation_scatter_comparison.png'
    dpi: int = 300
    figsize: Tuple[int, int] = (8, 8)


def load_validation_data(checkpoint_dir: str) -> Dict:
    """
    加载验证集数据及动态阈值表

    Args:
        checkpoint_dir: checkpoint目录路径

    Returns:
        dict: {
            'predictions': np.ndarray [N, 28, 5],
            'labels':      np.ndarray [N, 28, 5],
            'threshold_map': np.ndarray [365, 28],
            'val_time':    np.ndarray [N]  全局时间索引
        }

    Raises:
        FileNotFoundError: 必需文件不存在
        AssertionError: 数据形状不匹配
    """
    ckpt_path = Path(checkpoint_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")

    for fname in ('val_predict.npy', 'val_label.npy',
                  'threshold_map.npy', 'val_time.npy'):
        if not (ckpt_path / fname).exists():
            raise FileNotFoundError(f"缺失文件: {ckpt_path / fname}")

    predictions = np.load(ckpt_path / 'val_predict.npy')
    labels = np.load(ckpt_path / 'val_label.npy')
    threshold_map = np.load(ckpt_path / 'threshold_map.npy')  # [365, 28]
    val_time = np.load(ckpt_path / 'val_time.npy')            # [N]

    assert predictions.shape == labels.shape, \
        f"预测和标签形状不匹配: {predictions.shape} vs {labels.shape}"
    assert predictions.ndim == 3 and predictions.shape[1] == 28 \
        and predictions.shape[2] == 5, \
        f"数据形状异常: {predictions.shape}，期望 (N, 28, 5)"

    return {
        'predictions': predictions,
        'labels': labels,
        'threshold_map': threshold_map,
        'val_time': val_time,
    }


# 2018年验证集在全局数据集中的起始索引（2010–2017共2922天）
_VAL_YEAR_START = 2922


def flatten_predictions(
    data: Dict,
    pred_step: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将3D数据转换为1D数组，并生成每样本对应的站点分位数阈值

    Args:
        data: load_validation_data()返回的数据字典
        pred_step: 预测步长索引（0-4）

    Returns:
        predictions_1d: [N*28]
        labels_1d: [N*28]
        thresholds_1d: [N*28] 从checkpoint动态阈值表查得的逐样本阈值

    Raises:
        ValueError: pred_step超出范围
    """
    predictions = data['predictions']    # [N, 28, 5]
    labels = data['labels']              # [N, 28, 5]
    threshold_map = data['threshold_map']  # [365, 28]
    val_time = data['val_time']          # [N]

    if pred_step < 0 or pred_step >= 5:
        raise ValueError(f"pred_step={pred_step} 超出范围 [0, 4]")

    # 每个样本的预测目标日期对应的0-based doy
    # future_global_idx = val_time[i] + pred_step
    # doy_0based = future_global_idx - _VAL_YEAR_START
    doy_indices = np.clip(
        val_time + pred_step - _VAL_YEAR_START, 0, 364
    ).astype(int)  # [N]

    # 查表得到 [N, 28] 的逐样本阈值
    thresholds_2d = threshold_map[doy_indices, :]  # [N, 28]

    # 提取指定步长的数据
    pred_step_data = predictions[:, :, pred_step]  # [N, 28]
    label_step_data = labels[:, :, pred_step]      # [N, 28]

    # 行优先展平为1D：时间步在外层，站点在内层
    pred_1d = pred_step_data.flatten()
    label_1d = label_step_data.flatten()
    thresholds_1d = thresholds_2d.flatten()        # [N*28]

    return pred_1d, label_1d, thresholds_1d


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray,
    thresholds_1d: np.ndarray,
    config: ComparisonConfig
) -> None:
    """
    绘制散点图对比

    Args:
        y_true: 真实值 [N]
        y_pred_model1: 模型1预测值 [N]
        y_pred_model2: 模型2预测值 [N]
        thresholds_1d: 每样本对应站点的90分位阈值 [N]
        config: 配置对象
    """
    fig, ax_main = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    color_base = "#3681b6"
    color_prop = "#F87C7CCE"

    # --- 主图：散点图 ---
    ax_main.scatter(y_true, y_pred_model1, c=color_base, alpha=0.3, s=20,
                    label=config.model1_name, marker='o')
    ax_main.scatter(y_true, y_pred_model2, c=color_prop, alpha=0.3, s=20,
                    label=config.model2_name, marker='^')

    # 辅助线
    # 1. 计算所有数据的最小值和最大值
    all_data = np.concatenate([y_true, y_pred_model1, y_pred_model2])
    data_min = np.min(all_data)
    data_max = np.max(all_data)

    # 2. 添加一点缓冲 (比如 5%)
    padding = (data_max - data_min) * 0.05
    plot_min = data_min - padding
    plot_max = data_max + padding

    # 3. 绘制对角线
    ax_main.plot([plot_min, plot_max], [plot_min, plot_max],
                 color='gray', linestyle='--', lw=2)

    # 4. 坐标轴范围
    ax_main.set_xlim(plot_min, plot_max)
    ax_main.set_ylim(plot_min, plot_max)

    # 设置坐标轴 (显式关闭格网)
    ax_main.grid(False)
    ax_main.set_xlabel('Observed Temperature (°C)')
    ax_main.set_ylabel('Predicted Temperature (°C)')
    ax_main.legend(loc='upper left', frameon=True, framealpha=0.9)

    # --- 插图：高温残差分布（基于各站点90分位阈值）---
    mask_high = y_true >= thresholds_1d
    residuals_model1 = y_true[mask_high] - y_pred_model1[mask_high]
    residuals_model2 = y_true[mask_high] - y_pred_model2[mask_high]

    # 调整位置：通过 bbox_to_anchor 的第一个参数(x)向左移
    ax_inset = inset_axes(ax_main, width="35%", height="35%", loc=4,
                          bbox_to_anchor=(-0.0, 0.08, 1, 1),
                          bbox_transform=ax_main.transAxes)

    sns.kdeplot(residuals_model1, ax=ax_inset, color=color_base, fill=True,
                alpha=0.3, linewidth=2)
    sns.kdeplot(residuals_model2, ax=ax_inset, color=color_prop, fill=True,
                alpha=0.3, linewidth=2)

    ax_inset.axvline(0, color='k', linestyle='-', linewidth=1)

    # 插图样式设置
    ax_inset.grid(False)
    ax_inset.set_xlabel('Residual')
    ax_inset.set_ylabel('Density')
    ax_inset.set_xlim(-3, 5)
    ax_inset.tick_params(axis='both', which='major')

    # ========== 高温区域拟合直线 ==========
    if mask_high.sum() > 0:
        y_true_high = y_true[mask_high]
        y_pred1_high = y_pred_model1[mask_high]
        y_pred2_high = y_pred_model2[mask_high]

        coeffs1 = np.polyfit(y_true_high, y_pred1_high, 1)
        poly1 = np.poly1d(coeffs1)

        coeffs2 = np.polyfit(y_true_high, y_pred2_high, 1)
        poly2 = np.poly1d(coeffs2)

        leg = ax_main.legend(loc='upper left', frameon=True)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([60])
            if hasattr(handle, 'set_linewidth'):
                handle.set_linewidth(3)

    plt.tight_layout()

    # 保存
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
    plt.close()

    print(f"[OK] 对比图已保存至: {output_path}")


def main():
    """主函数"""
    # 获取项目根目录
    script_dir = Path(__file__).parent.parent

    # ============ 配置区域 ============
    config = ComparisonConfig(
        # 模型路径
        model1_dir=str(script_dir / 'myGNN' / 'checkpoints mean' /
                       'GAT_SeparateEncoder_MSE'),
        model2_dir=str(script_dir / 'myGNN' / 'checkpoints mean' /
                       'GAT_SeparateEncoder_WEIGHTED'),
        model1_name='MSE Loss',
        model2_name='Weighted Loss',

        # 数据转换配置
        pred_step=0,  # 第1步预测（0-4对应第1-5天）

        # 绘图配置
        output_path=str(script_dir / 'figdraw' / 'result' /
                        'validation_scatter_comparison.png'),
        dpi=300,
        figsize=(10, 8)
    )
    # ==================================

    print("=" * 70)
    print("验证集模型对比脚本")
    print("=" * 70)

    try:
        # Step 1: 加载数据
        print("\n[1/4] 加载模型数据...")
        print(f"  - 模型1: {config.model1_dir}")
        data1 = load_validation_data(config.model1_dir)
        print(f"    [OK] 数据形状: {data1['predictions'].shape}")

        print(f"  - 模型2: {config.model2_dir}")
        data2 = load_validation_data(config.model2_dir)
        print(f"    [OK] 数据形状: {data2['predictions'].shape}")

        # Step 2: 转换数据
        print(f"\n[2/4] 转换数据...")
        print(f"  - 预测步长: 第{config.pred_step + 1}天")
        pred1_1d, labels_1d, thresholds_1d = flatten_predictions(
            data1, config.pred_step)
        pred2_1d, _, _ = flatten_predictions(data2, config.pred_step)
        print(f"    [OK] 转换后形状: {pred1_1d.shape}")
        print(f"    [OK] 阈值范围: [{thresholds_1d.min():.2f}, "
              f"{thresholds_1d.max():.2f}] °C")

        # Step 3: 绘制对比图
        print(f"\n[3/4] 绘制散点图...")
        plot_scatter_comparison(labels_1d, pred1_1d, pred2_1d,
                                thresholds_1d, config)

        # Step 4: 输出统计信息
        mask_high = labels_1d >= thresholds_1d
        print(f"\n[4/4] 统计信息:")
        print(f"  - 样本数量: {len(labels_1d)}")
        print(f"  - 温度范围: [{labels_1d.min():.2f}, "
              f"{labels_1d.max():.2f}] °C")
        print(f"  - 高温样本 (>=各站点90分位阈值): {mask_high.sum()}")

        rmse1 = np.sqrt(np.mean((pred1_1d - labels_1d)**2))
        rmse2 = np.sqrt(np.mean((pred2_1d - labels_1d)**2))
        print(f"  - {config.model1_name} RMSE: {rmse1:.4f} °C")
        print(f"  - {config.model2_name} RMSE: {rmse2:.4f} °C")
        print(f"  - RMSE改进: {((rmse1 - rmse2) / rmse1 * 100):+.2f}%")

        # 高温样本RMSE
        if mask_high.sum() > 0:
            rmse1_high = np.sqrt(
                np.mean((pred1_1d[mask_high] - labels_1d[mask_high])**2))
            rmse2_high = np.sqrt(
                np.mean((pred2_1d[mask_high] - labels_1d[mask_high])**2))
            print(f"  - {config.model1_name} 高温RMSE: {rmse1_high:.4f} °C")
            print(f"  - {config.model2_name} 高温RMSE: {rmse2_high:.4f} °C")
            print(f"  - 高温RMSE改进: "
                  f"{((rmse1_high - rmse2_high) / rmse1_high * 100):+.2f}%")

        print("\n" + "=" * 70)
        print("[OK] 对比完成！")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
