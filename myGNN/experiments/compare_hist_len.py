"""
历史窗口长度对比实验

扫描 hist_len = 5..30（步长1），对比多个模型在测试集上的 RMSE/MAE/R²，
输出 CSV 并绘制折线图帮助选择最佳历史窗口。

所有超参数继承自 config.py，本脚本只覆盖实验控制变量:
  - hist_len（扫描范围）
  - exp_model（模型切换）
  - use_feature_separation（根据模型自动切换）
  - auto_visualize = False（实验中关闭可视化）

使用方法:
    python myGNN/experiments/compare_hist_len.py

输出:
    myGNN/experiments/results/hist_len_comparison.csv
    myGNN/experiments/results/hist_len_rmse.png

作者: GNN气温预测项目
日期: 2025
"""

import os
import sys
import time
import copy
import traceback
import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

matplotlib.use("Agg")

# ==================== 路径设置 ====================
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from myGNN.config import Config, ArchConfig, get_feature_indices_for_graph
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
)

# ==================== 实验参数（仅实验控制变量） ====================
HIST_LEN_LIST = list(range(5, 31))  # 5, 6, 7, ..., 30

# 模型列表：(模型名, 是否需要特征分离)
# SeparateEncoder 系列需要 use_feature_separation=True
# 其余模型使用 use_feature_separation=False + feature_indices 保证特征一致
MODEL_LIST = [
    ("GAT_SeparateEncoder", True),
    ("GAT_LSTM", True),
    ("GAT_Pure", True),
    ("LSTM", True),
]

# 输出目录
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_config_for_model(model_name, use_separation, hist_len):
    """
    为指定模型和窗口长度创建配置

    所有超参数（lr, epochs, batch_size, optimizer, scheduler 等）
    全部继承 Config() 默认值（即 config.py 中的定义）。
    本函数只覆盖 3 个实验控制变量 + 1 个便利开关。

    Args:
        model_name: 模型名称
        use_separation: 是否启用特征分离
        hist_len: 历史窗口长度

    Returns:
        config, arch_config
    """
    config = Config()
    arch_config = ArchConfig()

    # ---- 实验控制变量（仅这些会被覆盖） ----
    config.hist_len = hist_len
    config.exp_model = model_name
    config.auto_visualize = False  # 批量实验中关闭自动可视化

    return config, arch_config


def run_single_experiment(model_name, use_separation, hist_len, graph):
    """
    运行单次实验：训练 + 测试

    Args:
        model_name: 模型名称
        use_separation: 是否启用特征分离
        hist_len: 历史窗口长度
        graph: 预构建的图结构

    Returns:
        dict: 包含 rmse, mae, r2, bias, best_epoch, train_time_sec
    """
    config, arch_config = create_config_for_model(model_name, use_separation, hist_len)

    setup_seed(config.seed)

    # 加载数据
    train_loader, val_loader, test_loader, stats = create_dataloaders(config, graph)
    config.ta_mean = stats["ta_mean"]
    config.ta_std = stats["ta_std"]

    # 创建模型
    model = get_model(config, arch_config).to(config.device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # 训练
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    t_start = time.time()

    epoch_bar = tqdm(
        range(1, config.epochs + 1),
        desc=f"{model_name}|hist={hist_len}",
        unit="ep",
        leave=False,
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        train_loss = train(train_loader, model, optimizer, scheduler, config,
                           arch_config,
                           epoch=epoch, total_epochs=config.epochs)
        val_loss, _, _, _ = val(val_loader, model, config)

        # ReduceLROnPlateau 需要手动 step
        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        epoch_bar.set_postfix(
            val=f"{val_loss:.4f}",
            best=f"{best_val_loss:.4f}",
            pat=f"{patience_counter}/{config.early_stop}",
        )

        if patience_counter >= config.early_stop:
            break

    train_time_sec = time.time() - t_start

    # 加载最佳模型并测试
    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_pred, test_label, _ = test(test_loader, model, config)
    rmse, mae, r2, bias = get_metric(test_pred, test_label)

    # 第1天指标
    step1_metrics = get_metrics_per_step(test_pred, test_label)[0]

    # 释放显存
    del model, optimizer, scheduler, best_state
    del train_loader, val_loader, test_loader
    torch.cuda.empty_cache()

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "best_epoch": best_epoch,
        "train_time_sec": train_time_sec,
        "step1_rmse": step1_metrics["rmse"],
        "step1_mae": step1_metrics["mae"],
        "step1_r2": step1_metrics["r2"],
        "step1_bias": step1_metrics["bias"],
    }


def plot_results(df, save_path):
    """
    绘制 RMSE 折线图

    Args:
        df: 结果 DataFrame
        save_path: 图片保存路径
    """
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    models = df["model"].unique()

    # 颜色和标记
    colors = ["#E74C3C", "#2E86C1", "#27AE60", "#F39C12"]
    markers = ["o", "s", "D", "^"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- 子图1: RMSE ----
    ax1 = axes[0]
    for i, model_name in enumerate(models):
        sub = df[df["model"] == model_name].sort_values("hist_len")
        ax1.plot(
            sub["hist_len"],
            sub["rmse"],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=1.5,
            label=model_name,
            alpha=0.85,
        )

    ax1.set_xlabel("Historical Window Length (days)")
    ax1.set_ylabel("Test RMSE (°C)")
    ax1.set_title("RMSE vs. Historical Window Length")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(5, 31, 5))

    # ---- 子图2: MAE ----
    ax2 = axes[1]
    for i, model_name in enumerate(models):
        sub = df[df["model"] == model_name].sort_values("hist_len")
        ax2.plot(
            sub["hist_len"],
            sub["mae"],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            markersize=4,
            linewidth=1.5,
            label=model_name,
            alpha=0.85,
        )

    ax2.set_xlabel("Historical Window Length (days)")
    ax2.set_ylabel("Test MAE (°C)")
    ax2.set_title("MAE vs. Historical Window Length")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(5, 31, 5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 折线图已保存: {save_path}")


def main():
    """主实验入口"""
    # 读取 config 默认值用于显示
    _ref = Config()

    print("=" * 80)
    print("历史窗口长度对比实验")
    print(f"  窗口范围: {HIST_LEN_LIST[0]} ~ {HIST_LEN_LIST[-1]} 天")
    print(f"  预测长度: {_ref.pred_len} 天 (读自 config.py)")
    print(f"  模型: {[m[0] for m in MODEL_LIST]}")
    total = len(HIST_LEN_LIST) * len(MODEL_LIST)
    print(f"  总实验数: {total}")
    print(
        f"  训练参数: epochs={_ref.epochs}, early_stop={_ref.early_stop}, "
        f"lr={_ref.lr}, batch_size={_ref.batch_size}"
    )
    print("=" * 80)

    # 创建输出目录
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 构建图结构（所有实验共享，只需一次）
    print("\n[1/3] 构建图结构...")
    base_config = Config()
    graph = create_graph_from_config(base_config, feature_data=None)
    print("✓ 图结构构建完成")

    # 检查是否有断点续传的 CSV
    csv_path = os.path.join(RESULTS_DIR, "hist_len_comparison.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_keys = set(zip(existing_df["model"], existing_df["hist_len"]))
        results = existing_df.to_dict("records")
        print(f"\n检测到已有结果 ({len(done_keys)} 条)，将断点续传...")
    else:
        done_keys = set()
        results = []

    # 运行实验
    print("\n[2/3] 开始实验...")
    exp_idx = 0

    exp_tasks = [
        (model_name, use_sep, hist_len)
        for model_name, use_sep in MODEL_LIST
        for hist_len in HIST_LEN_LIST
    ]
    outer_bar = tqdm(exp_tasks, desc="实验进度", unit="exp", dynamic_ncols=True)

    for model_name, use_sep, hist_len in outer_bar:
        exp_idx += 1
        outer_bar.set_description(f"[{exp_idx}/{total}] {model_name}|hist={hist_len}")

        # 跳过已完成的
        if (model_name, hist_len) in done_keys:
            outer_bar.write(
                f"  [{exp_idx}/{total}] {model_name} | hist_len={hist_len} → 已完成，跳过"
            )
            continue

        try:
            result = run_single_experiment(model_name, use_sep, hist_len, graph)
            row = {
                "model": model_name,
                "hist_len": hist_len,
                "pred_len": _ref.pred_len,
                "rmse": round(result["rmse"], 4),
                "mae": round(result["mae"], 4),
                "r2": round(result["r2"], 4),
                "bias": round(result["bias"], 4),
                "best_epoch": result["best_epoch"],
                "train_time_sec": round(result["train_time_sec"], 1),
                "step1_rmse": round(result["step1_rmse"], 4),
                "step1_mae": round(result["step1_mae"], 4),
                "step1_r2": round(result["step1_r2"], 4),
                "step1_bias": round(result["step1_bias"], 4),
            }
            results.append(row)
            done_keys.add((model_name, hist_len))

            outer_bar.write(
                f"  ✓ {model_name} hist={hist_len:2d} | "
                f"RMSE={row['rmse']:.4f}  MAE={row['mae']:.4f}  "
                f"R²={row['r2']:.4f}  ep={row['best_epoch']}  "
                f"{row['train_time_sec']:.0f}s"
            )

            # 每次实验后保存 CSV（断点续传保障）
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        except Exception as e:
            outer_bar.write(f"  ✗ {model_name} hist={hist_len} 失败: {e}")
            traceback.print_exc()
            continue

    # 最终保存
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✓ 结果CSV已保存: {csv_path}")

    # 绘图
    print("\n[3/3] 绘制折线图...")
    fig_path = os.path.join(RESULTS_DIR, "hist_len_rmse.png")
    plot_results(df, fig_path)

    # 打印汇总
    print("\n" + "=" * 80)
    print("实验汇总 — 每个模型的最优 hist_len")
    print("=" * 80)

    for model_name, _ in MODEL_LIST:
        sub = df[df["model"] == model_name]
        if sub.empty:
            continue
        best_row = sub.loc[sub["rmse"].idxmin()]
        print(
            f"  {model_name:25s} → "
            f"hist_len={int(best_row['hist_len']):2d}  "
            f"RMSE={best_row['rmse']:.4f}  "
            f"MAE={best_row['mae']:.4f}  "
            f"R²={best_row['r2']:.4f}"
        )

    print("=" * 80)
    print("实验完成！")


if __name__ == "__main__":
    main()
