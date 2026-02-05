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
    threshold: float = 35.0  # 高温阈值
    output_path: str = 'figdraw/result/validation_scatter_comparison.png'
    dpi: int = 300
    figsize: Tuple[int, int] = (8, 8)


def load_validation_data(checkpoint_dir: str) -> Dict:
    """
    加载验证集数据

    Args:
        checkpoint_dir: checkpoint目录路径

    Returns:
        dict: {
            'predictions': np.ndarray [347, 28, 5],
            'labels': np.ndarray [347, 28, 5]
        }

    Raises:
        FileNotFoundError: 必需文件不存在
        AssertionError: 数据形状不匹配
    """
    ckpt_path = Path(checkpoint_dir)

    # 检查目录存在性
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint目录不存在: {checkpoint_dir}")

    # 检查文件存在性
    pred_file = ckpt_path / 'val_predict.npy'
    label_file = ckpt_path / 'val_label.npy'

    if not pred_file.exists():
        raise FileNotFoundError(f"缺失文件: {pred_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"缺失文件: {label_file}")

    # 加载数据
    predictions = np.load(pred_file)
    labels = np.load(label_file)

    # 验证形状
    assert predictions.shape == labels.shape, \
        f"预测和标签形状不匹配: {predictions.shape} vs {labels.shape}"
    assert predictions.shape == (347, 28, 5), \
        f"数据形状异常: {predictions.shape}，期望 (347, 28, 5)"

    return {
        'predictions': predictions,
        'labels': labels
    }


def flatten_predictions(
    data: Dict,
    pred_step: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将3D数据转换为1D数组

    Args:
        data: load_validation_data()返回的数据字典
        pred_step: 预测步长索引（0-4）

    Returns:
        predictions_1d: [9716] (347×28)
        labels_1d: [9716]

    Raises:
        ValueError: pred_step超出范围
    """
    predictions = data['predictions']  # [347, 28, 5]
    labels = data['labels']

    # 验证pred_step
    if pred_step < 0 or pred_step >= 5:
        raise ValueError(f"pred_step={pred_step} 超出范围 [0, 4]")

    # 提取指定步长的数据
    pred_step_data = predictions[:, :, pred_step]  # [347, 28]
    label_step_data = labels[:, :, pred_step]

    # 展平为1D数组
    pred_1d = pred_step_data.flatten()  # [9716]
    label_1d = label_step_data.flatten()

    return pred_1d, label_1d


def plot_scatter_comparison(
    y_true: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray,
    config: ComparisonConfig
) -> None:
    """
    绘制散点图对比（完全复用test.py代码）

    Args:
        y_true: 真实值 [N]
        y_pred_model1: 模型1预测值 [N]
        y_pred_model2: 模型2预测值 [N]
        config: 配置对象
    """
    THRESHOLD = config.threshold

    # ========== 以下代码直接复制自test.py第30-95行 ==========
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

    # 4. 同时更新坐标轴范围以匹配 (这步很重要，否则可能线画了但图没显示全)
    ax_main.set_xlim(plot_min, plot_max)
    ax_main.set_ylim(plot_min, plot_max)
    ax_main.axvline(x=THRESHOLD, color='gray', linestyle='--', lw=2)
    ax_main.axhline(y=THRESHOLD, color='gray', linestyle='--', lw=2)
    ax_main.fill_between([THRESHOLD, 43], THRESHOLD, 43, color='orange',alpha=0.05)

    # 设置坐标轴 (显式关闭格网)
    ax_main.grid(False)
    ax_main.set_xlabel('Observed Temperature (°C)')
    ax_main.set_ylabel('Predicted Temperature (°C)')
    ax_main.legend(loc='upper left', frameon=True, framealpha=0.9)

    # --- 插图：高温残差分布 ---
    mask_high = y_true >= THRESHOLD
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
    ax_inset.grid(False)  # 插图也关闭格网
    # ax_inset.spines['top'].set_visible(False)
    # ax_inset.spines['right'].set_visible(False)
    ax_inset.set_xlabel('Residual')
    ax_inset.set_ylabel('Density')
    ax_inset.set_xlim(-3, 5)
    ax_inset.tick_params(axis='both', which='major')

    # ========== 复制结束 ==========

    # ========== 新增：高温区域拟合直线和R²值 ==========
    # 在高温区域（>threshold）绘制拟合直线
    if mask_high.sum() > 0:
        # 提取高温区域数据
        y_true_high = y_true[mask_high]
        y_pred1_high = y_pred_model1[mask_high]
        y_pred2_high = y_pred_model2[mask_high]

        # 对模型1进行线性拟合
        # y_pred = a * y_true + b
        coeffs1 = np.polyfit(y_true_high, y_pred1_high, 1)
        poly1 = np.poly1d(coeffs1)

        # 对模型2进行线性拟合
        coeffs2 = np.polyfit(y_true_high, y_pred2_high, 1)
        poly2 = np.poly1d(coeffs2)

        # 计算R²值
        def calculate_r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot)

        r2_model1 = calculate_r2(y_true_high, y_pred1_high)
        r2_model2 = calculate_r2(y_true_high, y_pred2_high)

        # 绘制拟合直线（仅在高温区域）
        x_fit = np.linspace(THRESHOLD, 43, 100)
        y_fit1 = poly1(x_fit)
        y_fit2 = poly2(x_fit)

        # 绘制模型1的拟合线
        ax_main.plot(x_fit, y_fit1, color='darkblue', linestyle='-',
                     linewidth=2.5, alpha=0.8,)
                    #  label=f'{config.model1_name} Fit (R²={r2_model1:.3f})')

        # 绘制模型2的拟合线
        ax_main.plot(x_fit, y_fit2, color='darkred', linestyle='-',
                     linewidth=2.5, alpha=0.8,)
                    #  label=f'{config.model2_name} Fit (R²={r2_model2:.3f})')

        # 更新图例（重新设置以包含拟合线）
        # 1. 生成图例（获取图例对象）
        leg = ax_main.legend(loc='upper left', frameon=True)

        # 2. 遍历图例中的每一个句柄，强制设置为不透明
        for handle in leg.legend_handles:
            # 强制设置不透明度为 1.0 (完全不透明)
            handle.set_alpha(1.0)
            
            # 可选：如果是散点（通过检查是否有 set_sizes 方法），可以把图例里的点放大一点，看得更清
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([60])  # 设置图例中散点的大小
            
            # 可选：如果是线（拟合线），确保线宽一致
            if hasattr(handle, 'set_linewidth'):
                handle.set_linewidth(3)
    # ========== 新增结束 ==========

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
        model1_dir=str(script_dir / 'myGNN' / 'checkpoints' /
                       'GAT_SeparateEncoder_MSE'),
        model2_dir=str(script_dir / 'myGNN' / 'checkpoints' /
                       'GAT_SeparateEncoder_WEIGHTED'),
        model1_name='MSE Loss',
        model2_name='WeightedTrend Loss',

        # 数据转换配置
        pred_step=0,  # 第1步预测（0-4对应第1-5天）

        # 绘图配置
        threshold=35.0,
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
        pred1_1d, labels_1d = flatten_predictions(data1, config.pred_step)
        pred2_1d, _ = flatten_predictions(data2, config.pred_step)
        print(f"    [OK] 转换后形状: {pred1_1d.shape}")

        # Step 3: 绘制对比图
        print(f"\n[3/4] 绘制散点图...")
        plot_scatter_comparison(labels_1d, pred1_1d, pred2_1d, config)

        # Step 4: 输出统计信息
        print(f"\n[4/4] 统计信息:")
        print(f"  - 样本数量: {len(labels_1d)}")
        print(f"  - 温度范围: [{labels_1d.min():.2f}, "
              f"{labels_1d.max():.2f}] °C")
        print(f"  - 高温样本 (>{config.threshold}°C): "
              f"{(labels_1d > config.threshold).sum()}")

        rmse1 = np.sqrt(np.mean((pred1_1d - labels_1d)**2))
        rmse2 = np.sqrt(np.mean((pred2_1d - labels_1d)**2))
        print(f"  - {config.model1_name} RMSE: {rmse1:.4f} °C")
        print(f"  - {config.model2_name} RMSE: {rmse2:.4f} °C")
        print(f"  - RMSE改进: {((rmse1 - rmse2) / rmse1 * 100):+.2f}%")

        # 高温样本RMSE
        mask_high = labels_1d >= config.threshold
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
