"""
myGNN主训练脚本

这是整个框架的统一入口，提供完整的训练流程：
1. 配置管理（config.py）
2. 数据加载（dataset.py）
3. 图构建（distance_graph.py）
4. 模型训练（network_GNN.py）
5. 结果保存

使用方法：
    python train.py

配置修改：
    修改config.py中的参数，无需命令行参数

作者: GNN气温预测项目
日期: 2025
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 无GUI环境下使用

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
from myGNN.config import create_config, print_config, get_feature_indices_for_graph
from myGNN.dataset import create_dataloaders
from myGNN.graph.distance_graph import create_graph_from_config
from myGNN.network_GNN import (
    get_model,
    get_optimizer,
    get_scheduler,
    train,
    val,
    test,
    get_metric,
    get_metrics_per_step,
    get_exp_info,
    get_extreme_metrics,
    get_extreme_metrics_per_step,
)


def setup_seed(seed):
    """设置随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_loss_curves(train_losses, val_losses, best_epoch, save_dir):
    """
    绘制训练和验证损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        best_epoch: 最佳epoch
        save_dir: 保存目录
    """
    plt.figure(figsize=(12, 5))

    # 子图1: 完整训练曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5, alpha=0.8)
    plt.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5, alpha=0.8)
    plt.axvline(
        x=best_epoch,
        color="g",
        linestyle="--",
        linewidth=1.5,
        label=f"Best Epoch ({best_epoch})",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # 子图2: 对数尺度查看细节
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=1.5, alpha=0.8)
    plt.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=1.5, alpha=0.8)
    plt.axvline(
        x=best_epoch,
        color="g",
        linestyle="--",
        linewidth=1.5,
        label=f"Best Epoch ({best_epoch})",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.title(
        "Training and Validation Loss (Log Scale)", fontsize=14, fontweight="bold"
    )
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(save_dir / "loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ 损失曲线图已保存: {save_dir / 'loss_curves.png'}")


def save_loss_history(train_losses, val_losses, best_epoch, save_dir):
    """
    保存详细的loss历史记录到文本文件

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        best_epoch: 最佳epoch
        save_dir: 保存目录
    """
    with open(save_dir / "loss_history.txt", "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("训练损失历史记录\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总训练轮数: {len(train_losses)}\n")
        f.write(f"最佳Epoch: {best_epoch}\n")
        f.write(f"最佳验证损失: {val_losses[best_epoch - 1]:.6f}\n\n")

        # 统计信息
        f.write("【训练损失统计】\n")
        f.write(
            f"  最小值: {min(train_losses):.6f} (Epoch {train_losses.index(min(train_losses)) + 1})\n"
        )
        f.write(
            f"  最大值: {max(train_losses):.6f} (Epoch {train_losses.index(max(train_losses)) + 1})\n"
        )
        f.write(f"  最终值: {train_losses[-1]:.6f}\n")
        f.write(f"  平均值: {np.mean(train_losses):.6f}\n\n")

        f.write("【验证损失统计】\n")
        f.write(
            f"  最小值: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses)) + 1})\n"
        )
        f.write(
            f"  最大值: {max(val_losses):.6f} (Epoch {val_losses.index(max(val_losses)) + 1})\n"
        )
        f.write(f"  最终值: {val_losses[-1]:.6f}\n")
        f.write(f"  平均值: {np.mean(val_losses):.6f}\n\n")

        # 详细记录
        f.write("=" * 80 + "\n")
        f.write("逐轮损失详情\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Improvement':>12}\n"
        )
        f.write("-" * 80 + "\n")

        for i in range(len(train_losses)):
            epoch = i + 1
            train_loss = train_losses[i]
            val_loss = val_losses[i]

            # 计算改善情况
            if i == 0:
                improvement = "-"
            else:
                improvement = val_losses[i - 1] - val_loss
                improvement = f"{improvement:+.6f}"

            # 标记最佳epoch
            marker = " *BEST*" if epoch == best_epoch else ""

            f.write(
                f"{epoch:6d} | {train_loss:12.6f} | {val_loss:12.6f} | {improvement:>12} {marker}\n"
            )

        f.write("=" * 80 + "\n")

    print(f"✓ 损失历史记录已保存: {save_dir / 'loss_history.txt'}")


def _build_threshold_array_from_time(threshold_map, time_indices, pred_len, config):
    """
    根据时间索引和阈值表，构建逐样本逐预测步的阈值数组。

    Args:
        threshold_map: [365, num_stations] 动态阈值表
        time_indices: [num_samples] 每个样本的起始时间索引
        pred_len: 预测步长
        config: 配置对象

    Returns:
        threshold_array: [num_samples, num_stations, pred_len] 逐样本阈值
    """
    from myGNN.dataset import (
        YEAR_BOUNDARIES,
        _get_year_from_idx,
        is_leap_year,
        normalize_doy_for_loss,
    )

    num_samples = len(time_indices)
    num_stations = threshold_map.shape[1]
    target_idx = config.target_feature_idx
    met_data = np.load(config.MetData_fp)

    threshold_array = np.zeros((num_samples, num_stations, pred_len), dtype=np.float32)

    for i in range(num_samples):
        time_idx = int(time_indices[i])
        for step in range(pred_len):
            future_idx = time_idx + step
            raw_doy = int(met_data[future_idx, 0, 28])
            year = _get_year_from_idx(future_idx)
            try:
                doy_0based = normalize_doy_for_loss(year, raw_doy)
                threshold_array[i, :, step] = threshold_map[doy_0based, :]
            except ValueError:
                # 2月29日样本（理论上不应出现，因为被 collate_fn 过滤了）
                threshold_array[i, :, step] = np.median(threshold_map)

    return threshold_array


def create_save_dir(config):
    """
    创建保存目录

    Args:
        config: 配置对象

    Returns:
        save_dir: 保存目录路径
    """
    # 创建保存目录：save_path/模型名_时间戳
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.save_path) / f"{config.exp_model}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def save_results(
    save_dir,
    config,
    arch_config,
    best_epoch,
    train_results,
    val_results,
    test_results,
    stats=None,
):
    """
    保存训练结果

    Args:
        save_dir: 保存目录
        config: 配置对象
        arch_config: 模型架构配置对象
        best_epoch: 最佳epoch
        train_results: 训练集结果字典
        val_results: 验证集结果字典
        test_results: 测试集结果字典
        stats: 数据统计字典（可选，包含threshold_map等）
    """
    # 1. 保存完整配置（包含所有Config和ArchConfig参数）
    config_str = f"""
{"=" * 80}
myGNN训练配置 - 完整记录
{"=" * 80}

【数据配置】
  数据路径: {config.MetData_fp}
  站点信息路径: {config.station_info_fp}
  数据集标识: {config.dataset_num}
  节点数量: {config.node_num}

【数据集划分】
  训练集: 索引 {config.train_start}-{config.train_end - 1} ({config.train_end - config.train_start} 天)
  验证集: 索引 {config.val_start}-{config.val_end - 1} ({config.val_end - config.val_start} 天)
  测试集: 索引 {config.test_start}-{config.test_end - 1} ({config.test_end - config.test_start} 天)

【时间窗口配置】
  历史窗口长度 (hist_len): {config.hist_len} 天
  预测长度 (pred_len): {config.pred_len} 天

【特征配置】
  原始特征维度 (base_feature_dim): {config.base_feature_dim}
  预测目标索引 (target_feature_idx): {config.target_feature_idx}
  选择特征 (feature_indices): {config.feature_indices if config.feature_indices else "所有基础特征(0-23)"}
  时间编码 (add_temporal_encoding): {"启用" if config.add_temporal_encoding else "禁用"}
  时间特征维度 (temporal_features): {config.temporal_features if config.add_temporal_encoding else 0}
  最终输入维度 (in_dim): {config.in_dim}
  数据标准化参数:
    - ta_mean: {config.ta_mean:.6f}
    - ta_std: {config.ta_std:.6f}

【模型配置】
  模型类型 (exp_model): {config.exp_model}

【图结构配置】
  图类型 (graph_type): {config.graph_type}
  {"K近邻数量 (top_neighbors): " + str(config.top_neighbors) if config.graph_type in ["inv_dis", "knn"] else ""}
  {"使用边属性 (use_edge_attr): " + str(config.use_edge_attr) if config.graph_type in ["inv_dis", "knn"] else ""}
  {"空间相似性邻居数 (spatial_sim_top_k): " + str(config.spatial_sim_top_k) if config.graph_type == "spatial_similarity" else ""}
  {"邻域权重系数 (spatial_sim_alpha): " + str(config.spatial_sim_alpha) if config.graph_type == "spatial_similarity" else ""}
  {"使用邻域相似性 (spatial_sim_use_neighborhood): " + str(config.spatial_sim_use_neighborhood) if config.graph_type == "spatial_similarity" else ""}
  {"初始空间邻居数 (spatial_sim_initial_neighbors): " + str(config.spatial_sim_initial_neighbors) if config.graph_type == "spatial_similarity" else ""}

【训练配置】
  批次大小 (batch_size): {config.batch_size}
  最大训练轮数 (epochs): {config.epochs}
  学习率 (lr): {config.lr}
  权重衰减 (weight_decay): {config.weight_decay}
  早停耐心值 (early_stop): {config.early_stop}
  随机种子 (seed): {config.seed}

【优化器配置】
  优化器类型 (optimizer): {config.optimizer}
  {"动量 (momentum): " + str(config.momentum) if config.optimizer == "SGD" else ""}
  {"Betas (betas): " + str(config.betas) if config.optimizer in ["Adam", "AdamW"] else ""}

【学习率调度器配置】
  调度器类型 (scheduler): {config.scheduler if config.scheduler else "不使用"}
  {"StepLR - Step Size (step_size): " + str(config.step_size) if config.scheduler == "StepLR" else ""}
  {"StepLR - Gamma (gamma): " + str(config.gamma) if config.scheduler == "StepLR" else ""}
  {"CosineAnnealingLR - T_max (T_max): " + str(config.T_max) if config.scheduler == "CosineAnnealingLR" else ""}
  {"CosineAnnealingLR - Eta_min (eta_min): " + str(config.eta_min) if config.scheduler == "CosineAnnealingLR" else ""}
  {"ReduceLROnPlateau - Patience (patience): " + str(config.patience) if config.scheduler == "ReduceLROnPlateau" else ""}
  {"ReduceLROnPlateau - Factor (factor): " + str(config.factor) if config.scheduler == "ReduceLROnPlateau" else ""}
  {"MultiStepLR - Milestones (milestones): " + str(config.milestones) if config.scheduler == "MultiStepLR" else ""}
  {"MultiStepLR - Gamma (gamma): " + str(config.gamma) if config.scheduler == "MultiStepLR" else ""}

【损失函数配置】
  损失类型 (loss_type): {config.loss_config.loss_type}
  {"加权趋势损失参数:" if config.loss_config.loss_type == "WeightedTrend" else ""}
  {"  - 警戒阈值 (alert_temp): " + str(config.loss_config.alert_temp) + "°C" if config.loss_config.loss_type == "WeightedTrend" else ""}
  {"  - 漏报权重 (c_under): " + str(config.loss_config.c_under) if config.loss_config.loss_type == "WeightedTrend" else ""}
  {"  - 误报权重 (c_over): " + str(config.loss_config.c_over) if config.loss_config.loss_type == "WeightedTrend" else ""}
  {"  - 高温权重 (c_default_high): " + str(config.loss_config.c_default_high) if config.loss_config.loss_type == "WeightedTrend" else ""}
  {"多阈值加权参数:" if config.loss_config.loss_type == "MultiThreshold" else ""}
  {"  - 温度阈值 (multi_thresholds): " + str(config.loss_config.multi_thresholds) if config.loss_config.loss_type == "MultiThreshold" else ""}
  {"  - 权重列表 (multi_weights): " + str(config.loss_config.multi_weights) if config.loss_config.loss_type == "MultiThreshold" else ""}
  {"季节加权参数:" if config.loss_config.loss_type == "SeasonalWeighted" else ""}
  {"  - 夏季权重 (summer_weight): " + str(config.loss_config.summer_weight) if config.loss_config.loss_type == "SeasonalWeighted" else ""}
  {"  - 冬季权重 (winter_weight): " + str(config.loss_config.winter_weight) if config.loss_config.loss_type == "SeasonalWeighted" else ""}
  {"  - 春秋权重 (spring_fall_weight): " + str(config.loss_config.spring_fall_weight) if config.loss_config.loss_type == "SeasonalWeighted" else ""}


【模型架构配置 (ArchConfig)】
  隐藏层维度 (hid_dim): {arch_config.hid_dim}
  MLP层数 (MLP_layer): {arch_config.MLP_layer}
  激活函数 (AF): {arch_config.AF}
  规范化类型 (norm_type): {arch_config.norm_type}
  使用Dropout (dropout): {arch_config.dropout}

  {"GAT特定参数:" if "GAT" in config.exp_model else ""}
  {"  - GAT层数 (GAT_layer): " + str(arch_config.GAT_layer) if "GAT" in config.exp_model else ""}
  {"  - 注意力头数 (heads): " + str(arch_config.heads) if "GAT" in config.exp_model else ""}
  {"  - 层内Dropout (intra_drop): " + str(arch_config.intra_drop) if "GAT" in config.exp_model else ""}
  {"  - 层间Dropout (inter_drop): " + str(arch_config.inter_drop) if "GAT" in config.exp_model else ""}

  {"GraphSAGE特定参数:" if "SAGE" in config.exp_model else ""}
  {"  - SAGE层数 (SAGE_layer): " + str(arch_config.SAGE_layer) if "SAGE" in config.exp_model else ""}
  {"  - 聚合方式 (aggr): " + str(arch_config.aggr) if "SAGE" in config.exp_model else ""}
  {"  - 层间Dropout (inter_drop): " + str(arch_config.inter_drop) if "SAGE" in config.exp_model else ""}

  LSTM参数:
    - LSTM层数 (lstm_num_layers): {arch_config.lstm_num_layers}
    - LSTM Dropout (lstm_dropout): {arch_config.lstm_dropout}
    - 双向LSTM (lstm_bidirectional): {arch_config.lstm_bidirectional}


【可视化配置】
  自动可视化 (auto_visualize): {config.auto_visualize}
  可视化步长 (viz_pred_steps): {config.viz_pred_steps}
  绘制所有站点 (viz_plot_all_stations): {config.viz_plot_all_stations}
  图表DPI (viz_dpi): {config.viz_dpi}
  使用地理底图 (viz_use_basemap): {config.viz_use_basemap}

【设备配置】
  计算设备 (device): {config.device}

【路径配置】
  保存路径 (save_path): {config.save_path}
  日志路径 (log_path): {config.log_path}

{"=" * 80}
训练结果
{"=" * 80}
最佳Epoch: {best_epoch}
{"=" * 80}
"""
    with open(save_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(config_str)

    # 2. 保存训练集结果
    np.save(save_dir / "train_predict.npy", train_results["predict"])
    np.save(save_dir / "train_label.npy", train_results["label"])
    np.save(save_dir / "train_time.npy", train_results["time"])

    # 3. 保存验证集结果
    np.save(save_dir / "val_predict.npy", val_results["predict"])
    np.save(save_dir / "val_label.npy", val_results["label"])
    np.save(save_dir / "val_time.npy", val_results["time"])

    # 4. 保存测试集结果
    np.save(save_dir / "test_predict.npy", test_results["predict"])
    np.save(save_dir / "test_label.npy", test_results["label"])
    np.save(save_dir / "test_time.npy", test_results["time"])

    # 4. 计算极端值监控指标
    print("\n正在计算极端值监控指标...")

    # 判断是否使用动态阈值
    use_dynamic = getattr(config.loss_config, "use_station_day_threshold", False)
    threshold_map = stats.get("threshold_map") if stats else None

    if use_dynamic and threshold_map is not None:
        # 动态阈值模式：构建逐样本阈值数组
        print("  构建逐样本动态阈值数组...")
        train_thr = _build_threshold_array_from_time(
            threshold_map, train_results["time"], config.pred_len, config
        )
        val_thr = _build_threshold_array_from_time(
            threshold_map, val_results["time"], config.pred_len, config
        )
        test_thr = _build_threshold_array_from_time(
            threshold_map, test_results["time"], config.pred_len, config
        )
        print(
            f"  阈值数组形状: train={train_thr.shape}, val={val_thr.shape}, test={test_thr.shape}"
        )
    else:
        train_thr = val_thr = test_thr = None

    # 计算整体极端值指标
    train_extreme_metrics = get_extreme_metrics(
        train_results["predict"],
        train_results["label"],
        threshold_array=train_thr,
    )
    val_extreme_metrics = get_extreme_metrics(
        val_results["predict"],
        val_results["label"],
        threshold_array=val_thr,
    )
    test_extreme_metrics = get_extreme_metrics(
        test_results["predict"],
        test_results["label"],
        threshold_array=test_thr,
    )

    # 计算按步长分解的极端值指标
    train_extreme_per_step = get_extreme_metrics_per_step(
        train_results["predict"],
        train_results["label"],
        threshold_array=train_thr,
    )
    val_extreme_per_step = get_extreme_metrics_per_step(
        val_results["predict"],
        val_results["label"],
        threshold_array=val_thr,
    )
    test_extreme_per_step = get_extreme_metrics_per_step(
        test_results["predict"],
        test_results["label"],
        threshold_array=test_thr,
    )

    # 5. 保存评估指标 (包含极端值信息)
    metrics_str = f"""
评估指标
{"=" * 80}
训练集:
  RMSE: {train_results["rmse"]:.4f} °C
  MAE:  {train_results["mae"]:.4f} °C
  R²:   {train_results["r2"]:.4f}
  Bias: {train_results["bias"]:+.4f} °C

验证集:
  RMSE: {val_results["rmse"]:.4f} °C
  MAE:  {val_results["mae"]:.4f} °C
  R²:   {val_results["r2"]:.4f}
  Bias: {val_results["bias"]:+.4f} °C

测试集:
  RMSE: {test_results["rmse"]:.4f} °C
  MAE:  {test_results["mae"]:.4f} °C
  R²:   {test_results["r2"]:.4f}
  Bias: {test_results["bias"]:+.4f} °C

指标说明:
  RMSE (均方根误差): 值越小越好，单位为°C
  MAE (平均绝对误差): 值越小越好，单位为°C
  R² (决定系数): 范围[0, 1]，值越接近1越好
    - R² = 1: 完美预测
    - R² = 0: 预测效果等同于使用平均值
    - R² < 0: 预测效果差于使用平均值
  Bias (系统性偏差): 单位为°C
    - Bias > 0: 模型倾向于高估温度
    - Bias = 0: 模型无系统性偏差
    - Bias < 0: 模型倾向于低估温度


{"=" * 80}
极端值监控指标
{"=" * 80}

【样本分布统计】
"""

    # 添加样本分布统计
    if train_extreme_metrics["normal_temp"]:
        metrics_str += f"训练集:\n"
        for ht in train_extreme_metrics["high_temp"]:
            metrics_str += (
                f"  高于90分位: {ht['sample_count']}样本 ({ht['percentage']:.1f}%)\n"
            )
        metrics_str += (
            f"  低于90分位: {train_extreme_metrics['normal_temp']['sample_count']}样本 "
        )
        metrics_str += f"({train_extreme_metrics['normal_temp']['percentage']:.1f}%)\n"

    if val_extreme_metrics["normal_temp"]:
        metrics_str += f"验证集:\n"
        for ht in val_extreme_metrics["high_temp"]:
            metrics_str += (
                f"  高于90分位: {ht['sample_count']}样本 ({ht['percentage']:.1f}%)\n"
            )
        metrics_str += (
            f"  低于90分位: {val_extreme_metrics['normal_temp']['sample_count']}样本 "
        )
        metrics_str += f"({val_extreme_metrics['normal_temp']['percentage']:.1f}%)\n"

        metrics_str += f"测试集:\n"
        for ht in test_extreme_metrics["high_temp"]:
            metrics_str += (
                f"  高于90分位: {ht['sample_count']}样本 ({ht['percentage']:.1f}%)\n"
            )
        metrics_str += (
            f"  低于90分位: {test_extreme_metrics['normal_temp']['sample_count']}样本 "
        )
        metrics_str += f"({test_extreme_metrics['normal_temp']['percentage']:.1f}%)\n"

    # 添加分类性能指标（按数据集分组）
    metrics_str += f"\n【训练集分类性能】\n"
    for ht in train_extreme_metrics["high_temp"]:
        metrics_str += f"  高于90分位:\n"
        metrics_str += f"    RMSE: {ht['rmse']:.4f} °C, MAE: {ht['mae']:.4f} °C, Bias: {ht['bias']:+.4f} °C\n"
        metrics_str += f"    低估率: {ht['underestimate_rate']:.1f}%, 高估率: {ht['overestimate_rate']:.1f}%\n"
        metrics_str += f"    命中率: {ht['hit_rate']:.1f}%, 误报率: {ht['false_alarm_rate']:.1f}%, 漏报率: {ht['miss_rate']:.1f}%\n"
    for lt in train_extreme_metrics["low_temp"]:
        metrics_str += f"  低于90分位:\n"
        metrics_str += f"    RMSE: {lt['rmse']:.4f} °C, MAE: {lt['mae']:.4f} °C, Bias: {lt['bias']:+.4f} °C\n"
        metrics_str += f"    低估率: {lt['underestimate_rate']:.1f}%, 高估率: {lt['overestimate_rate']:.1f}%\n"
        metrics_str += f"    命中率: {lt['hit_rate']:.1f}%, 误报率: {lt['false_alarm_rate']:.1f}%, 漏报率: {lt['miss_rate']:.1f}%\n"

    metrics_str += f"\n【验证集分类性能】\n"
    for ht in val_extreme_metrics["high_temp"]:
        metrics_str += f"  高于90分位:\n"
        metrics_str += f"    RMSE: {ht['rmse']:.4f} °C, MAE: {ht['mae']:.4f} °C, Bias: {ht['bias']:+.4f} °C\n"
        metrics_str += f"    低估率: {ht['underestimate_rate']:.1f}%, 高估率: {ht['overestimate_rate']:.1f}%\n"
        metrics_str += f"    命中率: {ht['hit_rate']:.1f}%, 误报率: {ht['false_alarm_rate']:.1f}%, 漏报率: {ht['miss_rate']:.1f}%\n"
    for lt in val_extreme_metrics["low_temp"]:
        metrics_str += f"  低于90分位:\n"
        metrics_str += f"    RMSE: {lt['rmse']:.4f} °C, MAE: {lt['mae']:.4f} °C, Bias: {lt['bias']:+.4f} °C\n"
        metrics_str += f"    低估率: {lt['underestimate_rate']:.1f}%, 高估率: {lt['overestimate_rate']:.1f}%\n"
        metrics_str += f"    命中率: {lt['hit_rate']:.1f}%, 误报率: {lt['false_alarm_rate']:.1f}%, 漏报率: {lt['miss_rate']:.1f}%\n"

    metrics_str += f"\n【测试集分类性能】\n"
    for ht in test_extreme_metrics["high_temp"]:
        metrics_str += f"  高于90分位:\n"
        metrics_str += f"    RMSE: {ht['rmse']:.4f} °C, MAE: {ht['mae']:.4f} °C, Bias: {ht['bias']:+.4f} °C\n"
        metrics_str += f"    低估率: {ht['underestimate_rate']:.1f}%, 高估率: {ht['overestimate_rate']:.1f}%\n"
        metrics_str += f"    命中率: {ht['hit_rate']:.1f}%, 误报率: {ht['false_alarm_rate']:.1f}%, 漏报率: {ht['miss_rate']:.1f}%\n"
    for lt in test_extreme_metrics["low_temp"]:
        metrics_str += f"  低于90分位:\n"
        metrics_str += f"    RMSE: {lt['rmse']:.4f} °C, MAE: {lt['mae']:.4f} °C, Bias: {lt['bias']:+.4f} °C\n"
        metrics_str += f"    低估率: {lt['underestimate_rate']:.1f}%, 高估率: {lt['overestimate_rate']:.1f}%\n"
        metrics_str += f"    命中率: {lt['hit_rate']:.1f}%, 误报率: {lt['false_alarm_rate']:.1f}%, 漏报率: {lt['miss_rate']:.1f}%\n"

    metrics_str += f"{'=' * 80}\n"

    # 保存 metrics.txt
    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(metrics_str)

    # 6. 保存按预测步长分解的指标
    train_metrics_per_step = get_metrics_per_step(
        train_results["predict"], train_results["label"]
    )
    val_metrics_per_step = get_metrics_per_step(
        val_results["predict"], val_results["label"]
    )
    test_metrics_per_step = get_metrics_per_step(
        test_results["predict"], test_results["label"]
    )

    # 保存为CSV格式（便于分析）
    import pandas as pd

    # 训练集按步长指标
    train_df = pd.DataFrame(train_metrics_per_step)
    train_df.to_csv(save_dir / "train_metrics_per_step.csv", index=False)

    # 验证集按步长指标
    val_df = pd.DataFrame(val_metrics_per_step)
    val_df.to_csv(save_dir / "val_metrics_per_step.csv", index=False)

    # 测试集按步长指标
    test_df = pd.DataFrame(test_metrics_per_step)
    test_df.to_csv(save_dir / "test_metrics_per_step.csv", index=False)

    # 同时保存为可读的txt格式
    metrics_per_step_str = f"""
按预测步长分解的指标
{"=" * 80}
训练集:
"""
    for m in train_metrics_per_step:
        metrics_per_step_str += (
            f"  第{m['step']}步 (+{m['step']}天): "
            f"RMSE={m['rmse']:.4f}°C, "
            f"MAE={m['mae']:.4f}°C, "
            f"R²={m['r2']:.4f}, "
            f"Bias={m['bias']:+.4f}°C\n"
        )

    metrics_per_step_str += "\n验证集:\n"
    for m in val_metrics_per_step:
        metrics_per_step_str += (
            f"  第{m['step']}步 (+{m['step']}天): "
            f"RMSE={m['rmse']:.4f}°C, "
            f"MAE={m['mae']:.4f}°C, "
            f"R²={m['r2']:.4f}, "
            f"Bias={m['bias']:+.4f}°C\n"
        )

    metrics_per_step_str += "\n测试集:\n"
    for m in test_metrics_per_step:
        metrics_per_step_str += (
            f"  第{m['step']}步 (+{m['step']}天): "
            f"RMSE={m['rmse']:.4f}°C, "
            f"MAE={m['mae']:.4f}°C, "
            f"R²={m['r2']:.4f}, "
            f"Bias={m['bias']:+.4f}°C\n"
        )

    metrics_per_step_str += f"{'=' * 80}\n"

    with open(save_dir / "metrics_per_step.txt", "w", encoding="utf-8") as f:
        f.write(metrics_per_step_str)

    # 7. 保存极端值指标文件
    # 7.1 保存整体极端值指标为CSV
    extreme_csv_data = []

    # 训练集 - 高温
    for ht in train_extreme_metrics["high_temp"]:
        extreme_csv_data.append(
            {
                "dataset": "train",
                "temp_type": "high",
                "threshold": ht["threshold"],
                "sample_count": ht["sample_count"],
                "percentage": ht["percentage"],
                "rmse": ht["rmse"],
                "mae": ht["mae"],
                "bias": ht["bias"],
                "underestimate_rate": ht["underestimate_rate"],
                "overestimate_rate": ht["overestimate_rate"],
                "hit_rate": ht["hit_rate"],
                "false_alarm_rate": ht["false_alarm_rate"],
                "miss_rate": ht["miss_rate"],
            }
        )

    # 训练集 - 低温
    for lt in train_extreme_metrics["low_temp"]:
        extreme_csv_data.append(
            {
                "dataset": "train",
                "temp_type": "low",
                "threshold": lt["threshold"],
                "sample_count": lt["sample_count"],
                "percentage": lt["percentage"],
                "rmse": lt["rmse"],
                "mae": lt["mae"],
                "bias": lt["bias"],
                "underestimate_rate": lt["underestimate_rate"],
                "overestimate_rate": lt["overestimate_rate"],
                "hit_rate": lt["hit_rate"],
                "false_alarm_rate": lt["false_alarm_rate"],
                "miss_rate": lt["miss_rate"],
            }
        )

    # 验证集 - 高温
    for ht in val_extreme_metrics["high_temp"]:
        extreme_csv_data.append(
            {
                "dataset": "val",
                "temp_type": "high",
                "threshold": ht["threshold"],
                "sample_count": ht["sample_count"],
                "percentage": ht["percentage"],
                "rmse": ht["rmse"],
                "mae": ht["mae"],
                "bias": ht["bias"],
                "underestimate_rate": ht["underestimate_rate"],
                "overestimate_rate": ht["overestimate_rate"],
                "hit_rate": ht["hit_rate"],
                "false_alarm_rate": ht["false_alarm_rate"],
                "miss_rate": ht["miss_rate"],
            }
        )

    # 验证集 - 低温
    for lt in val_extreme_metrics["low_temp"]:
        extreme_csv_data.append(
            {
                "dataset": "val",
                "temp_type": "low",
                "threshold": lt["threshold"],
                "sample_count": lt["sample_count"],
                "percentage": lt["percentage"],
                "rmse": lt["rmse"],
                "mae": lt["mae"],
                "bias": lt["bias"],
                "underestimate_rate": lt["underestimate_rate"],
                "overestimate_rate": lt["overestimate_rate"],
                "hit_rate": lt["hit_rate"],
                "false_alarm_rate": lt["false_alarm_rate"],
                "miss_rate": lt["miss_rate"],
            }
        )

    # 测试集 - 高温
    for ht in test_extreme_metrics["high_temp"]:
        extreme_csv_data.append(
            {
                "dataset": "test",
                "temp_type": "high",
                "threshold": ht["threshold"],
                "sample_count": ht["sample_count"],
                "percentage": ht["percentage"],
                "rmse": ht["rmse"],
                "mae": ht["mae"],
                "bias": ht["bias"],
                "underestimate_rate": ht["underestimate_rate"],
                "overestimate_rate": ht["overestimate_rate"],
                "hit_rate": ht["hit_rate"],
                "false_alarm_rate": ht["false_alarm_rate"],
                "miss_rate": ht["miss_rate"],
            }
        )

    # 测试集 - 低温
    for lt in test_extreme_metrics["low_temp"]:
        extreme_csv_data.append(
            {
                "dataset": "test",
                "temp_type": "low",
                "threshold": lt["threshold"],
                "sample_count": lt["sample_count"],
                "percentage": lt["percentage"],
                "rmse": lt["rmse"],
                "mae": lt["mae"],
                "bias": lt["bias"],
                "underestimate_rate": lt["underestimate_rate"],
                "overestimate_rate": lt["overestimate_rate"],
                "hit_rate": lt["hit_rate"],
                "false_alarm_rate": lt["false_alarm_rate"],
                "miss_rate": lt["miss_rate"],
            }
        )

    extreme_df = pd.DataFrame(extreme_csv_data)
    extreme_df.to_csv(save_dir / "extreme_metrics.csv", index=False)

    # 7.2 保存按步长分解的极端值指标为CSV
    extreme_per_step_csv = []

    for step_data in train_extreme_per_step:
        step = step_data["step"]
        # 高温
        for ht in step_data["high_temp"]:
            extreme_per_step_csv.append(
                {
                    "dataset": "train",
                    "step": step,
                    "temp_type": "high",
                    "threshold": ht["threshold"],
                    "sample_count": ht["sample_count"],
                    "rmse": ht["rmse"],
                    "mae": ht["mae"],
                    "bias": ht["bias"],
                    "hit_rate": ht["hit_rate"],
                }
            )
        # 低温
        for lt in step_data["low_temp"]:
            extreme_per_step_csv.append(
                {
                    "dataset": "train",
                    "step": step,
                    "temp_type": "low",
                    "threshold": lt["threshold"],
                    "sample_count": lt["sample_count"],
                    "rmse": lt["rmse"],
                    "mae": lt["mae"],
                    "bias": lt["bias"],
                    "hit_rate": lt["hit_rate"],
                }
            )

    for step_data in val_extreme_per_step:
        step = step_data["step"]
        # 高温
        for ht in step_data["high_temp"]:
            extreme_per_step_csv.append(
                {
                    "dataset": "val",
                    "step": step,
                    "temp_type": "high",
                    "threshold": ht["threshold"],
                    "sample_count": ht["sample_count"],
                    "rmse": ht["rmse"],
                    "mae": ht["mae"],
                    "bias": ht["bias"],
                    "hit_rate": ht["hit_rate"],
                }
            )
        # 低温
        for lt in step_data["low_temp"]:
            extreme_per_step_csv.append(
                {
                    "dataset": "val",
                    "step": step,
                    "temp_type": "low",
                    "threshold": lt["threshold"],
                    "sample_count": lt["sample_count"],
                    "rmse": lt["rmse"],
                    "mae": lt["mae"],
                    "bias": lt["bias"],
                    "hit_rate": lt["hit_rate"],
                }
            )

    for step_data in test_extreme_per_step:
        step = step_data["step"]
        # 高温
        for ht in step_data["high_temp"]:
            extreme_per_step_csv.append(
                {
                    "dataset": "test",
                    "step": step,
                    "temp_type": "high",
                    "threshold": ht["threshold"],
                    "sample_count": ht["sample_count"],
                    "rmse": ht["rmse"],
                    "mae": ht["mae"],
                    "bias": ht["bias"],
                    "hit_rate": ht["hit_rate"],
                }
            )
        # 低温
        for lt in step_data["low_temp"]:
            extreme_per_step_csv.append(
                {
                    "dataset": "test",
                    "step": step,
                    "temp_type": "low",
                    "threshold": lt["threshold"],
                    "sample_count": lt["sample_count"],
                    "rmse": lt["rmse"],
                    "mae": lt["mae"],
                    "bias": lt["bias"],
                    "hit_rate": lt["hit_rate"],
                }
            )

    extreme_per_step_df = pd.DataFrame(extreme_per_step_csv)
    extreme_per_step_df.to_csv(save_dir / "extreme_metrics_per_step.csv", index=False)

    # 8. 保存阈值表（如果使用站点-日内动态阈值）
    if stats is not None:
        threshold_map_to_save = stats.get("threshold_map")
        sample_count_to_save = stats.get("threshold_sample_count")
        if threshold_map_to_save is not None:
            np.save(save_dir / "threshold_map.npy", threshold_map_to_save)
            np.save(save_dir / "threshold_sample_count.npy", sample_count_to_save)
            print(f"✓ 阈值表已保存: {save_dir / 'threshold_map.npy'}")

    print(f"\n✓ 结果已保存到: {save_dir}")
    print(f"✓ 按步长分解的指标已保存:")
    print(f"  - {save_dir / 'val_metrics_per_step.csv'}")
    print(f"  - {save_dir / 'test_metrics_per_step.csv'}")
    print(f"  - {save_dir / 'metrics_per_step.txt'}")
    print(f"✓ 极端值监控指标已保存:")
    print(f"  - {save_dir / 'extreme_metrics.csv'}")
    print(f"  - {save_dir / 'extreme_metrics_per_step.csv'}")
    print(f"  - metrics.txt (已包含极端值部分)")


def main():
    """主训练函数"""
    print("=" * 80)
    print("myGNN 气温预测模型训练")
    print("=" * 80)

    # ==================== 1. 加载配置 ====================
    print("\n[1/7] 加载配置...")

    # 🔥 加载配置
    # 配置方式：在config.py的LossConfig类中，修改 self.loss_type = 'MSE' 或 'WeightedTrend'
    config, arch_config = create_config()

    # 打印使用的损失函数类型
    print(f"✓ 损失函数类型: {config.loss_config.loss_type}")
    if config.use_enhanced_training:
        print(f"✓ 使用增强训练流程")
    else:
        print(f"✓ 使用标准训练流程（MSE）")

    # 设置随机种子
    setup_seed(config.seed)

    # 打印配置信息
    print_config(config, arch_config)

    # 创建保存目录
    save_dir = create_save_dir(config)
    print(f"\n✓ 保存目录: {save_dir}")

    # ==================== 2. 构建图结构 ====================
    print("\n[2/7] 构建图结构...")

    # 如果使用空间相似性图或correlation_climate图，需要先加载数据准备特征
    feature_data = None
    if config.graph_type == "spatial_similarity":
        print("  空间相似性图需要特征数据，先加载训练数据...")
        MetData_temp = np.load(config.MetData_fp)

        # 使用辅助函数获取特征索引（与数据加载保持一致）

        feature_indices = get_feature_indices_for_graph(config)

        # 提取训练集对应特征
        train_data_temp = MetData_temp[config.train_start : config.train_end, :, :]
        train_data_temp = train_data_temp[:, :, feature_indices]

        # 时间平均得到每个站点的特征向量
        feature_data = train_data_temp.mean(axis=0)  # [num_stations, num_features]

        print(f"  ✓ 特征数据形状: {feature_data.shape}")
        print(f"    (28个站点 × {len(feature_indices)}个特征)")

    elif config.graph_type == "correlation_climate":
        print("  correlation_climate图需要完整气象数据...")
        feature_data = np.load(
            config.MetData_fp
        )  # [total_len, num_stations, num_features]
        print(f"  ✓ 气象数据形状: {feature_data.shape}")
        print(
            f"    (将使用训练集 [{config.train_start}, {config.train_end}) 计算相关性和统计量)"
        )

    graph = create_graph_from_config(config, feature_data=feature_data)

    # 打印图信息(兼容不同的图对象格式)
    if hasattr(graph, "edge_form"):
        # 旧格式的图对象
        print(f"✓ 图类型: {graph.edge_form}")
        print(f"  节点数: {graph.node_num}")
        print(f"  边数: {graph.edge_index.shape[1]}")
        print(f"  使用边属性: {graph.use_edge_attr}")
    else:
        # PyG Data对象(correlation_climate返回的格式)
        print(f"✓ 图类型: {config.graph_type}")
        print(f"  节点数: {graph.num_nodes}")
        print(f"  边数: {graph.num_edges}")
        edge_attr_status = "是" if graph.edge_attr is not None else "否"
        print(f"  使用边属性: {edge_attr_status}")

    # ==================== 3. 加载数据 ====================
    print("\n[3/7] 加载数据...")
    train_loader, val_loader, test_loader, stats = create_dataloaders(config, graph)

    # 更新配置中的标准化参数
    config.ta_mean = stats["ta_mean"]
    config.ta_std = stats["ta_std"]

    # 三级优先级：站点-日内动态阈值 > 全局动态阈值 > 固定阈值
    loss_cfg = config.loss_config

    if getattr(loss_cfg, "use_station_day_threshold", False):
        threshold_map = stats.get("threshold_map")
        sample_count_map = stats.get("threshold_sample_count")
        if threshold_map is not None:
            print(f"  [阈值] 站点-日内动态阈值: shape={threshold_map.shape}")
            print(
                f"         分位数={loss_cfg.threshold_percentile}, "
                f"窗口=±{loss_cfg.threshold_window_radius}天"
            )
            print(
                f"         阈值范围: [{threshold_map.min():.2f}, "
                f"{threshold_map.max():.2f}]°C"
            )
            print(
                f"         样本数范围: [{sample_count_map.min()}, "
                f"{sample_count_map.max()}]"
            )
        else:
            print("  [阈值] 站点-日内动态阈值未构建，回退到全局动态阈值")
            loss_cfg.use_station_day_threshold = False
            config.loss_config.alert_temp = stats["ta_p90"]
            print(f"  [阈值] 全局动态阈值: {config.loss_config.alert_temp:.3f}°C")
    elif loss_cfg.use_dynamic_threshold:
        config.loss_config.alert_temp = stats["ta_p90"]
        print(
            f"  [阈值] 全局动态阈值: {config.loss_config.alert_temp:.3f}°C "
            f"(训练集90分位数)"
        )
    else:
        print(f"  [阈值] 固定高温阈值: {config.loss_config.alert_temp:.1f}°C")

    print(f"✓ 数据加载完成")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")

    # ==================== 4. 创建模型 ====================
    print("\n[4/7] 创建模型...")
    model = get_model(config, arch_config)

    model = model.to(config.device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ 模型: {config.exp_model}")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  设备: {config.device}")

    # ==================== 5. 设置优化器和调度器 ====================
    print("\n[5/7] 设置优化器和调度器...")
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    print(f"✓ 优化器: {config.optimizer}")
    if scheduler is not None:
        print(f"✓ 调度器: {config.scheduler}")
    else:
        print(f"✓ 调度器: 不使用")

    # ==================== 5.5. 设置损失函数 ====================
    # 根据配置自动选择训练流程
    if config.use_enhanced_training:
        print("\n[5.5/7] 设置增强损失函数...")
        from train_enhanced import (
            get_loss_function,
            train_epoch as train_enhanced,
            validate_epoch,
        )

        # 如果启用站点-日内动态阈值，传递 threshold_map
        threshold_map_for_loss = None
        if getattr(config.loss_config, "use_station_day_threshold", False):
            threshold_map_for_loss = stats.get("threshold_map")

        criterion = get_loss_function(config, threshold_map=threshold_map_for_loss)
        use_enhanced = True
    else:
        print("\n[5.5/7] 使用标准MSE损失函数...")
        criterion = None  # network_GNN.py 中有全局的 criterion
        use_enhanced = False

    # ==================== 6. 训练模型 ====================
    print("\n[6/7] 开始训练...")
    print(get_exp_info(config))

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    best_val_results = None

    train_losses = []
    val_losses = []

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # 训练 - 根据配置选择训练方法
        if use_enhanced:
            # 使用增强训练流程（带加权趋势损失）
            # 参数顺序：model, dataloader, optimizer, scheduler, criterion, config, device
            train_loss = train_enhanced(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                config,
                config.device,
            )
        else:
            # 使用标准训练流程（MSE损失）
            train_loss = train(train_loader, model, optimizer, scheduler, config)
        train_losses.append(train_loss)

        # 验证 - 根据配置选择验证方法
        if use_enhanced:
            # 使用增强验证流程（validate_epoch 现在返回完整的4个值）
            # 参数顺序：model, dataloader, criterion, config, device
            val_loss, val_pred, val_label, val_time = validate_epoch(
                model, val_loader, criterion, config, config.device
            )
        else:
            # 使用标准验证流程
            val_loss, val_pred, val_label, val_time = val(val_loader, model, config)
        val_losses.append(val_loss)

        # ReduceLROnPlateau调度器需要在验证后调用
        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time

        # 打印进度
        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0

            # 保存模型（包含config和graph，保证可解释性分析一致性）
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "arch_config": arch_config,
                "graph": graph,  # 🔑 新增: 保存训练时的图结构
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, save_dir / "best_model.pth")

            # 保存最佳验证结果
            best_val_results = {
                "predict": val_pred,
                "label": val_label,
                "time": val_time,
                "loss": val_loss,
            }

            print(f"  ✓ 最佳模型已保存 (Val Loss: {val_loss:.4f})")
        else:
            patience += 1

        # 早停
        if patience >= config.early_stop:
            print(f"\n早停触发！已连续 {patience} 个epoch无改善")
            break

    print(f"\n训练完成！最佳Epoch: {best_epoch}, 最佳验证损失: {best_val_loss:.4f}")

    # 保存训练曲线数据（numpy格式）
    np.save(save_dir / "train_losses.npy", np.array(train_losses))
    np.save(save_dir / "val_losses.npy", np.array(val_losses))

    # 保存详细的loss历史记录（文本格式）
    save_loss_history(train_losses, val_losses, best_epoch, save_dir)

    # 绘制并保存loss曲线图
    plot_loss_curves(train_losses, val_losses, best_epoch, save_dir)

    # ==================== 7. 测试最佳模型 ====================
    print("\n[6/7] 测试最佳模型...")

    # 安全检查：确保保存了最佳模型
    if best_val_results is None:
        print("⚠ 警告: 训练过程未保存任何结果，可能训练轮数为0或所有epoch都失败")
        print("将使用当前模型进行测试...")
        val_loss, val_pred, val_label, val_time = val(val_loader, model, config)
        best_val_results = {
            "predict": val_pred,
            "label": val_label,
            "time": val_time,
            "loss": val_loss,
        }
        best_epoch = 0
    else:
        # 加载最佳模型（兼容新旧格式）
        checkpoint = torch.load(save_dir / "best_model.pth", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # 测试
    test_loss, test_pred, test_label, test_time = test(test_loader, model, config)

    # ==================== 6.5. 评估训练集 ====================
    print("\n正在评估训练集...")

    # 选择合适的评估函数
    if use_enhanced:
        # 使用增强验证流程
        train_eval_loss, train_pred, train_label, train_time = validate_epoch(
            model, train_loader, criterion, config, config.device
        )
    else:
        # 使用标准验证流程
        # 注意：val()函数实际是通用的评估函数，可用于任何数据集
        train_eval_loss, train_pred, train_label, train_time = val(
            train_loader, model, config
        )

    # 计算详细指标
    train_rmse, train_mae, train_r2, train_bias = get_metric(train_pred, train_label)
    val_rmse, val_mae, val_r2, val_bias = get_metric(
        best_val_results["predict"], best_val_results["label"]
    )
    test_rmse, test_mae, test_r2, test_bias = get_metric(test_pred, test_label)

    # 计算按预测步长分解的指标
    train_metrics_per_step = get_metrics_per_step(train_pred, train_label)
    val_metrics_per_step = get_metrics_per_step(
        best_val_results["predict"], best_val_results["label"]
    )
    test_metrics_per_step = get_metrics_per_step(test_pred, test_label)

    print(f"✓ 训练集评估完成:")
    print(
        f"  RMSE: {train_rmse:.4f} °C, MAE: {train_mae:.4f} °C, "
        f"R²: {train_r2:.4f}, Bias: {train_bias:+.4f} °C"
    )

    # 输出训练集指标
    print(f"\n训练集 (最佳模型 Epoch {best_epoch}):")
    print(f"  整体（所有预测步长平均）:")
    print(
        f"    RMSE: {train_rmse:.4f} °C, MAE: {train_mae:.4f} °C, "
        f"R²: {train_r2:.4f}, Bias: {train_bias:+.4f} °C"
    )
    print(f"  按预测步长分解:")
    for metrics in train_metrics_per_step:
        print(
            f"    第{metrics['step']}步 (+{metrics['step']}天): "
            f"RMSE: {metrics['rmse']:.4f} °C, "
            f"MAE: {metrics['mae']:.4f} °C, "
            f"R²: {metrics['r2']:.4f}, "
            f"Bias: {metrics['bias']:+.4f} °C"
        )

    # 输出验证集指标
    print(f"\n验证集 (最佳模型 Epoch {best_epoch}):")
    print(f"  整体（所有预测步长平均）:")
    print(
        f"    RMSE: {val_rmse:.4f} °C, MAE: {val_mae:.4f} °C, "
        f"R²: {val_r2:.4f}, Bias: {val_bias:+.4f} °C"
    )
    print(f"  按预测步长分解:")
    for metrics in val_metrics_per_step:
        print(
            f"    第{metrics['step']}步 (+{metrics['step']}天): "
            f"RMSE: {metrics['rmse']:.4f} °C, "
            f"MAE: {metrics['mae']:.4f} °C, "
            f"R²: {metrics['r2']:.4f}, "
            f"Bias: {metrics['bias']:+.4f} °C"
        )

    # 输出测试集指标
    print(f"\n测试集:")
    print(f"  整体（所有预测步长平均）:")
    print(
        f"    RMSE: {test_rmse:.4f} °C, MAE: {test_mae:.4f} °C, "
        f"R²: {test_r2:.4f}, Bias: {test_bias:+.4f} °C"
    )
    print(f"  按预测步长分解:")
    for metrics in test_metrics_per_step:
        print(
            f"    第{metrics['step']}步 (+{metrics['step']}天): "
            f"RMSE: {metrics['rmse']:.4f} °C, "
            f"MAE: {metrics['mae']:.4f} °C, "
            f"R²: {metrics['r2']:.4f}, "
            f"Bias: {metrics['bias']:+.4f} °C"
        )

    # 计算并输出极端值监控指标
    print("\n" + "=" * 80)
    print("极端值监控指标")
    print("=" * 80)

    # 判断是否使用动态阈值
    use_dynamic_main = getattr(config.loss_config, "use_station_day_threshold", False)
    threshold_map_main = stats.get("threshold_map")

    if use_dynamic_main and threshold_map_main is not None:
        # 动态阈值模式：构建逐样本阈值数组
        print("  构建逐样本动态阈值数组...")
        train_thr_main = _build_threshold_array_from_time(
            threshold_map_main, train_time, config.pred_len, config
        )
        val_thr_main = _build_threshold_array_from_time(
            threshold_map_main, val_time, config.pred_len, config
        )
        test_thr_main = _build_threshold_array_from_time(
            threshold_map_main, test_time, config.pred_len, config
        )
    else:
        train_thr_main = val_thr_main = test_thr_main = None

    # 计算极端值指标
    train_extreme = get_extreme_metrics(
        train_pred, train_label, threshold_array=train_thr_main
    )
    val_extreme = get_extreme_metrics(
        best_val_results["predict"],
        best_val_results["label"],
        threshold_array=val_thr_main,
    )
    test_extreme = get_extreme_metrics(
        test_pred, test_label, threshold_array=test_thr_main
    )

    # 输出样本分布统计
    print("\n【样本分布统计】")
    print("训练集:")
    for ht in train_extreme["high_temp"]:
        print(f"  高于90分位: {ht['sample_count']}样本 ({ht['percentage']:.1f}%)")
    if train_extreme["normal_temp"]:
        print(
            f"  低于90分位: {train_extreme['normal_temp']['sample_count']}样本 "
            f"({train_extreme['normal_temp']['percentage']:.1f}%)"
        )

    print("验证集:")
    for ht in val_extreme["high_temp"]:
        print(f"  高于90分位: {ht['sample_count']}样本 ({ht['percentage']:.1f}%)")
    if val_extreme["normal_temp"]:
        print(
            f"  低于90分位: {val_extreme['normal_temp']['sample_count']}样本 "
            f"({val_extreme['normal_temp']['percentage']:.1f}%)"
        )

    print("测试集:")
    for ht in test_extreme["high_temp"]:
        print(f"  高于90分位: {ht['sample_count']}样本 ({ht['percentage']:.1f}%)")
    if test_extreme["normal_temp"]:
        print(
            f"  低于90分位: {test_extreme['normal_temp']['sample_count']}样本 "
            f"({test_extreme['normal_temp']['percentage']:.1f}%)"
        )

    # 输出分类性能（按数据集分组）
    print("\n【训练集分类性能】")
    for ht in train_extreme["high_temp"]:
        print(f"  高于90分位:")
        print(
            f"    RMSE: {ht['rmse']:.4f} °C, MAE: {ht['mae']:.4f} °C, Bias: {ht['bias']:+.4f} °C"
        )
        print(
            f"    低估率: {ht['underestimate_rate']:.1f}%, 高估率: {ht['overestimate_rate']:.1f}%"
        )
        print(
            f"    命中率: {ht['hit_rate']:.1f}%, 误报率: {ht['false_alarm_rate']:.1f}%, "
            f"漏报率: {ht['miss_rate']:.1f}%"
        )
    for lt in train_extreme["low_temp"]:
        print(f"  低于90分位:")
        print(
            f"    RMSE: {lt['rmse']:.4f} °C, MAE: {lt['mae']:.4f} °C, Bias: {lt['bias']:+.4f} °C"
        )
        print(
            f"    低估率: {lt['underestimate_rate']:.1f}%, 高估率: {lt['overestimate_rate']:.1f}%"
        )
        print(
            f"    命中率: {lt['hit_rate']:.1f}%, 误报率: {lt['false_alarm_rate']:.1f}%, "
            f"漏报率: {lt['miss_rate']:.1f}%"
        )

    print("\n【验证集分类性能】")
    for ht in val_extreme["high_temp"]:
        print(f"  高于90分位:")
        print(
            f"    RMSE: {ht['rmse']:.4f} °C, MAE: {ht['mae']:.4f} °C, Bias: {ht['bias']:+.4f} °C"
        )
        print(
            f"    低估率: {ht['underestimate_rate']:.1f}%, 高估率: {ht['overestimate_rate']:.1f}%"
        )
        print(
            f"    命中率: {ht['hit_rate']:.1f}%, 误报率: {ht['false_alarm_rate']:.1f}%, "
            f"漏报率: {ht['miss_rate']:.1f}%"
        )
    for lt in val_extreme["low_temp"]:
        print(f"  低于90分位:")
        print(
            f"    RMSE: {lt['rmse']:.4f} °C, MAE: {lt['mae']:.4f} °C, Bias: {lt['bias']:+.4f} °C"
        )
        print(
            f"    低估率: {lt['underestimate_rate']:.1f}%, 高估率: {lt['overestimate_rate']:.1f}%"
        )
        print(
            f"    命中率: {lt['hit_rate']:.1f}%, 误报率: {lt['false_alarm_rate']:.1f}%, "
            f"漏报率: {lt['miss_rate']:.1f}%"
        )

    print("\n【测试集分类性能】")
    for ht in test_extreme["high_temp"]:
        print(f"  高于90分位:")
        print(
            f"    RMSE: {ht['rmse']:.4f} °C, MAE: {ht['mae']:.4f} °C, Bias: {ht['bias']:+.4f} °C"
        )
        print(
            f"    低估率: {ht['underestimate_rate']:.1f}%, 高估率: {ht['overestimate_rate']:.1f}%"
        )
        print(
            f"    命中率: {ht['hit_rate']:.1f}%, 误报率: {ht['false_alarm_rate']:.1f}%, "
            f"漏报率: {ht['miss_rate']:.1f}%"
        )
    for lt in test_extreme["low_temp"]:
        print(f"  低于90分位:")
        print(
            f"    RMSE: {lt['rmse']:.4f} °C, MAE: {lt['mae']:.4f} °C, Bias: {lt['bias']:+.4f} °C"
        )
        print(
            f"    低估率: {lt['underestimate_rate']:.1f}%, 高估率: {lt['overestimate_rate']:.1f}%"
        )
        print(
            f"    命中率: {lt['hit_rate']:.1f}%, 误报率: {lt['false_alarm_rate']:.1f}%, "
            f"漏报率: {lt['miss_rate']:.1f}%"
        )

    print("=" * 80)

    # ==================== 8. 保存结果 ====================
    print("\n[7/7] 保存结果...")

    # 构建训练集结果字典
    best_train_results = {
        "predict": train_pred,
        "label": train_label,
        "time": train_time,
        "loss": train_eval_loss,
        "rmse": train_rmse,
        "mae": train_mae,
        "r2": train_r2,
        "bias": train_bias,
    }

    best_val_results["rmse"] = val_rmse
    best_val_results["mae"] = val_mae
    best_val_results["r2"] = val_r2
    best_val_results["bias"] = val_bias

    test_results = {
        "predict": test_pred,
        "label": test_label,
        "time": test_time,
        "loss": test_loss,
        "rmse": test_rmse,
        "mae": test_mae,
        "r2": test_r2,
        "bias": test_bias,
    }

    save_results(
        save_dir,
        config,
        arch_config,
        best_epoch,
        best_train_results,
        best_val_results,
        test_results,
        stats=stats,
    )

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)

    # ==================== 8. 自动生成可视化（可选） ====================
    if config.auto_visualize:
        print("\n[8/7] 自动生成可视化...")
        print("=" * 80)

        try:
            # 导入可视化函数
            from visualize_results import visualize_checkpoint

            # 调用可视化
            success = visualize_checkpoint(
                checkpoint_dir=str(save_dir),
                output_dir="auto",
                pred_steps=config.viz_pred_steps,
                plot_all_stations=config.viz_plot_all_stations,
                dpi=config.viz_dpi,
                use_basemap=config.viz_use_basemap,
                add_scalebar=False,
                add_north_arrow=False,
                silent=False,
            )

            if success:
                print(f"\n✅ 可视化已自动生成: {save_dir / 'visualizations'}")
            else:
                print(f"\n⚠ 可视化生成失败，请查看上方错误信息")

        except ImportError as e:
            print(f"\n⚠ 无法导入可视化模块: {e}")
            print("  如需自动可视化，请确保visualize_results.py在同一目录")
        except Exception as e:
            print(f"\n⚠ 可视化过程出错: {e}")
            print("  训练结果已保存，您可以稍后手动运行可视化")

        print("=" * 80)

    # 提示用户
    if not config.auto_visualize:
        print("\n💡 提示:")
        print(f"  训练结果已保存到: {save_dir}")
        print(f"  如需生成可视化,可以:")
        print(f"  1. 在config.py中设置 config.auto_visualize = True")
        print(f"  2. 或手动运行:")
        print(f"     python myGNN/visualize_results.py")
        print(f"     并修改配置中的 CHECKPOINT_DIR = '{save_dir.name}'")


if __name__ == "__main__":
    main()
