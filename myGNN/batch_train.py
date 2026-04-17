"""
批量串行训练脚本

功能：
- 批量训练多个模型和损失函数组合
- 串行执行每个任务，共享数据加载（优化性能）
- 自动汇总所有实验结果

训练任务配置：
| # | 模型 | 损失函数 |
|---|------|----------|
| 1 | GAT_LSTM | MSE |
| 2 | LSTM | MSE |
| 3 | GAT_Pure | MSE |
| 4 | GAT_SeparateEncoder | MSE |
| 5 | GAT_SeparateEncoder | WeightedTrend |

使用方法：
    python myGNN/batch_train.py

作者: GNN气温预测项目
日期: 2025
"""

import copy
import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入项目模块
from myGNN.config import (
    create_config,
    print_config,
    get_feature_indices_for_graph,
    Config,
    ArchConfig,
)
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


# ==================== 训练任务配置 ====================
TRAIN_TASKS = [
    {"model": "GAT_LSTM", "loss": "MSE", "desc": "GAT + LSTM"},
    {"model": "LSTM", "loss": "MSE", "desc": "纯LSTM基线"},
    {"model": "GAT_Pure", "loss": "MSE", "desc": "纯GAT（无LSTM）"},
    {"model": "GAT_SeparateEncoder", "loss": "MSE", "desc": "GAT + 特征分离（MSE）"},
    {
        "model": "GAT_SeparateEncoder",
        "loss": "WeightedTrend",
        "desc": "GAT + 特征分离（WeightedTrend）",
    },
]


def setup_seed(seed):
    """设置随机种子，保证结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_save_dir(config, task_suffix):
    """创建保存目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = Path(config.save_path) / f"{task_suffix}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _build_threshold_array_from_time(threshold_map, time_indices, pred_len, config):
    """根据时间索引和阈值表，构建逐样本逐预测步的阈值数组"""
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
                threshold_array[i, :, step] = np.median(threshold_map)

    return threshold_array


def run_single_training(task, base_config, base_arch_config, graph, dataloaders, stats):
    """
    执行单个训练任务

    Args:
        task: 任务配置字典
        base_config: 基础配置对象
        base_arch_config: 基础架构配置对象
        graph: 图结构对象
        dataloaders: (train_loader, val_loader, test_loader)
        stats: 数据统计信息

    Returns:
        results: 训练结果字典
    """
    task_desc = task["desc"]
    model_name = task["model"]
    loss_type = task["loss"]

    print(f"\n{'=' * 80}")
    print(f"开始训练: {task_desc} (模型: {model_name}, 损失: {loss_type})")
    print(f"{'=' * 80}")

    # 1. 复制配置并修改模型和损失函数
    config = copy.deepcopy(base_config)
    arch_config = copy.deepcopy(base_arch_config)

    config.exp_model = model_name
    config.loss_config.loss_type = loss_type

    # 设置随机种子（每个任务独立）
    setup_seed(config.seed)

    # 2. 创建保存目录
    task_suffix = f"{model_name}_{loss_type}"
    save_dir = create_save_dir(config, task_suffix)
    print(f"保存目录: {save_dir}")

    # 3. 获取数据加载器
    train_loader, val_loader, test_loader = dataloaders

    # 4. 更新配置中的标准化参数
    config.ta_mean = stats["ta_mean"]
    config.ta_std = stats["ta_std"]

    # 设置阈值
    loss_cfg = config.loss_config
    if getattr(loss_cfg, "use_station_day_threshold", False):
        threshold_map = stats.get("threshold_map")
        if threshold_map is not None:
            print(f"  [阈值] 站点-日内动态阈值: shape={threshold_map.shape}")
        else:
            loss_cfg.use_station_day_threshold = False
            config.loss_config.alert_temp = stats["ta_p90"]
            print(f"  [阈值] 全局动态阈值: {config.loss_config.alert_temp:.3f}°C")
    elif loss_cfg.use_dynamic_threshold:
        config.loss_config.alert_temp = stats["ta_p90"]
        print(f"  [阈值] 全局动态阈值: {config.loss_config.alert_temp:.3f}°C")
    else:
        print(f"  [阈值] 固定高温阈值: {config.loss_config.alert_temp:.1f}°C")

    # 5. 创建模型
    model = get_model(config, arch_config)
    model = model.to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型: {config.exp_model}, 参数: {trainable_params:,}")

    # 6. 设置优化器和调度器
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    # 7. 设置损失函数
    use_enhanced = loss_type != "MSE"

    if use_enhanced:
        from train_enhanced import (
            get_loss_function,
            train_epoch as train_enhanced,
            validate_epoch,
        )

        threshold_map_for_loss = None
        if getattr(config.loss_config, "use_station_day_threshold", False):
            threshold_map_for_loss = stats.get("threshold_map")

        criterion = get_loss_function(config, threshold_map=threshold_map_for_loss)
    else:
        criterion = None

    # 8. 训练模型
    print(f"\n开始训练 ({config.epochs} epochs)...")
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    best_val_results = None

    train_losses = []
    val_losses = []

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()

        # 训练
        if use_enhanced:
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
            train_loss = train(train_loader, model, optimizer, scheduler, config)
        train_losses.append(train_loss)

        # 验证
        if use_enhanced:
            val_loss, val_pred, val_label, val_time = validate_epoch(
                model, val_loader, criterion, config, config.device
            )
        else:
            val_loss, val_pred, val_label, val_time = val(val_loader, model, config)
        val_losses.append(val_loss)

        # 调度器
        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time

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

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "arch_config": arch_config,
                "graph": graph,
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, save_dir / "best_model.pth")

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

    # 9. 加载最佳模型并测试
    if best_val_results is None:
        val_loss, val_pred, val_label, val_time = val(val_loader, model, config)
        best_val_results = {
            "predict": val_pred,
            "label": val_label,
            "time": val_time,
            "loss": val_loss,
        }
        best_epoch = 0
    else:
        checkpoint = torch.load(save_dir / "best_model.pth", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    # 测试
    test_loss, test_pred, test_label, test_time = test(test_loader, model, config)

    # 评估训练集
    if use_enhanced:
        from train_enhanced import validate_epoch

        train_eval_loss, train_pred, train_label, train_time = validate_epoch(
            model, train_loader, criterion, config, config.device
        )
    else:
        train_eval_loss, train_pred, train_label, train_time = val(
            train_loader, model, config
        )

    # 计算指标
    train_rmse, train_mae, train_r2, train_bias = get_metric(train_pred, train_label)
    val_rmse, val_mae, val_r2, val_bias = get_metric(
        best_val_results["predict"], best_val_results["label"]
    )
    test_rmse, test_mae, test_r2, test_bias = get_metric(test_pred, test_label)

    # 计算按步长分解的指标
    test_metrics_per_step = get_metrics_per_step(test_pred, test_label)

    print(f"\n测试集结果:")
    print(
        f"  RMSE: {test_rmse:.4f} °C, MAE: {test_mae:.4f} °C, "
        f"R²: {test_r2:.4f}, Bias: {test_bias:+.4f} °C"
    )

    # 10. 保存结果
    np.save(save_dir / "train_predict.npy", train_pred)
    np.save(save_dir / "train_label.npy", train_label)
    np.save(save_dir / "train_time.npy", train_time)

    np.save(save_dir / "val_predict.npy", best_val_results["predict"])
    np.save(save_dir / "val_label.npy", best_val_results["label"])
    np.save(save_dir / "val_time.npy", best_val_results["time"])

    np.save(save_dir / "test_predict.npy", test_pred)
    np.save(save_dir / "test_label.npy", test_label)
    np.save(save_dir / "test_time.npy", test_time)

    np.save(save_dir / "train_losses.npy", np.array(train_losses))
    np.save(save_dir / "val_losses.npy", np.array(val_losses))

    # 保存config
    config_dict = {
        "model": config.exp_model,
        "loss_type": config.loss_config.loss_type,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "hist_len": config.hist_len,
        "pred_len": config.pred_len,
        "graph_type": config.graph_type,
    }
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # 保存metrics
    metrics_dict = {
        "task_desc": task_desc,
        "best_epoch": best_epoch,
        "train": {
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2),
            "bias": float(train_bias),
        },
        "val": {
            "rmse": float(val_rmse),
            "mae": float(val_mae),
            "r2": float(val_r2),
            "bias": float(val_bias),
        },
        "test": {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2),
            "bias": float(test_bias),
        },
    }

    # 添加按步长分解的测试指标
    metrics_dict["test_per_step"] = [
        {
            "step": m["step"],
            "rmse": float(m["rmse"]),
            "mae": float(m["mae"]),
            "r2": float(m["r2"]),
            "bias": float(m["bias"]),
        }
        for m in test_metrics_per_step
    ]

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 结果已保存到: {save_dir}")

    return {
        "task": task,
        "save_dir": str(save_dir),
        "best_epoch": best_epoch,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "test_bias": test_bias,
    }


def plot_mse_models_comparison(mse_results, output_dir):
    """
    为所有MSE模型生成Lead time对比图

    Args:
        mse_results: MSE模型结果列表，每个元素包含 {'task': {...}, 'save_dir': '...'}
        output_dir: 输出目录（通常是汇总目录）
    """
    if len(mse_results) < 2:
        print(f"  ⚠ MSE模型数量不足 ({len(mse_results)})，跳过对比图生成")
        return

    print(f"\n  正在为 {len(mse_results)} 个MSE模型生成对比图...")

    # 构建checkpoint路径字典
    checkpoint_paths = {}
    model_name_map = {
        "GAT_LSTM": "GAT-LSTM",
        "LSTM": "LSTM",
        "GAT_Pure": "GAT",
        "GAT_SeparateEncoder": "IGNN",
    }

    for r in mse_results:
        model = r["task"]["model"]
        save_dir = r["save_dir"]
        # 使用映射后的名称
        plot_name = model_name_map.get(model, model)
        checkpoint_paths[plot_name] = save_dir

    print(f"    模型映射: {checkpoint_paths}")

    # 调用plot_lead_time_comparison.py
    import importlib.util

    # 确保figdraw目录在路径中
    figdraw_dir = Path(__file__).parent.parent / "figdraw"
    if str(figdraw_dir) not in sys.path:
        sys.path.insert(0, str(figdraw_dir))

    try:
        # 动态导入并调用main函数
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "plot_lead_time_comparison", figdraw_dir / "plot_lead_time_comparison.py"
        )
        plot_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot_module)

        # 设置保存路径到output_dir
        save_path = Path(output_dir) / "mse_models_comparison.png"

        # 调用函数（绕过main，直接调用核心逻辑）
        lead_times, errors_dict, lower_dict, upper_dict, all_metrics_dict = (
            plot_module.generate_virtual_data(checkpoint_paths, "RMSE", "test")
        )

        plot_module.plot_lead_time_comparison(
            lead_times, errors_dict, str(save_path), lower_dict, upper_dict, "RMSE"
        )

        print(f"    ✓ 对比图已保存: {save_path}")

    except Exception as e:
        print(f"    ⚠ 对比图生成失败: {e}")
        import traceback

        traceback.print_exc()


def generate_summary(results_list):
    """生成汇总表格"""
    print("\n" + "=" * 80)
    print("批量训练结果汇总")
    print("=" * 80)

    # 按RMSE排序
    sorted_results = sorted(results_list, key=lambda x: x["test_rmse"])

    print(
        f"\n{'任务':<35} | {'测试RMSE':>10} | {'测试MAE':>10} | {'测试R²':>10} | {'测试Bias':>10}"
    )
    print("-" * 80)

    for r in sorted_results:
        task_desc = r["task"]["desc"]
        print(
            f"{task_desc:<35} | {r['test_rmse']:>10.4f} | "
            f"{r['test_mae']:>10.4f} | {r['test_r2']:>10.4f} | "
            f"{r['test_bias']:>+10.4f}"
        )

    print("-" * 80)

    # 最佳模型
    best = sorted_results[0]
    print(f"\n🏆 最佳模型: {best['task']['desc']}")
    print(f"   测试RMSE: {best['test_rmse']:.4f} °C")
    print(f"   测试MAE: {best['test_mae']:.4f} °C")
    print(f"   测试R²: {best['test_r2']:.4f}")

    return sorted_results


def main():
    """主函数"""
    print("=" * 80)
    print("批量串行训练 - GNN气温预测模型")
    print("=" * 80)
    print(f"任务数量: {len(TRAIN_TASKS)}")
    for i, task in enumerate(TRAIN_TASKS, 1):
        print(f"  {i}. {task['desc']} ({task['model']}, {task['loss']})")

    # ==================== 1. 加载基础配置 ====================
    print("\n[1/5] 加载基础配置...")

    # 创建基础配置（使用config.py中的默认值）
    base_config, base_arch_config = create_config()

    # 打印配置信息
    print(f"基础配置:")
    print(f"  训练轮数: {base_config.epochs}")
    print(f"  批次大小: {base_config.batch_size}")
    print(f"  学习率: {base_config.lr}")
    print(f"  历史窗口: {base_config.hist_len}天")
    print(f"  预测长度: {base_config.pred_len}天")

    # ==================== 2. 构建图结构 ====================
    print("\n[2/5] 构建图结构...")

    feature_data = None
    if base_config.graph_type == "spatial_similarity":
        print("  空间相似性图需要特征数据...")
        MetData_temp = np.load(base_config.MetData_fp)
        feature_indices = get_feature_indices_for_graph(base_config)
        train_data_temp = MetData_temp[
            base_config.train_start : base_config.train_end, :, :
        ]
        train_data_temp = train_data_temp[:, :, feature_indices]
        feature_data = train_data_temp.mean(axis=0)

    graph = create_graph_from_config(base_config, feature_data=feature_data)

    if hasattr(graph, "edge_form"):
        print(f"✓ 图类型: {graph.edge_form}, 边数: {graph.edge_index.shape[1]}")
    else:
        print(f"✓ 图类型: {base_config.graph_type}, 边数: {graph.num_edges}")

    # ==================== 3. 加载数据 ====================
    print("\n[3/5] 加载数据...")

    train_loader, val_loader, test_loader, stats = create_dataloaders(
        base_config, graph
    )

    print(f"✓ 数据加载完成")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print(f"  测试批次数: {len(test_loader)}")

    dataloaders = (train_loader, val_loader, test_loader)

    # ==================== 4. 串行训练每个任务 ====================
    print("\n[4/5] 开始串行训练...")

    results_list = []
    total_start_time = time.time()

    for i, task in enumerate(TRAIN_TASKS, 1):
        task_start_time = time.time()

        print(f"\n\n{'#' * 80}")
        print(f"# 任务 {i}/{len(TRAIN_TASKS)}: {task['desc']}")
        print(f"{'#' * 80}")

        try:
            result = run_single_training(
                task, base_config, base_arch_config, graph, dataloaders, stats
            )
            results_list.append(result)

            task_time = time.time() - task_start_time
            print(f"\n✓ 任务完成! 耗时: {task_time / 60:.1f}分钟")

        except Exception as e:
            print(f"\n❌ 任务失败: {task['desc']}")
            print(f"错误: {e}")
            import traceback

            traceback.print_exc()
            continue

    total_time = time.time() - total_start_time
    print(f"\n\n{'=' * 80}")
    print(f"所有任务完成! 总耗时: {total_time / 60:.1f}分钟")
    print(f"{'=' * 80}")

    # ==================== 5. 生成汇总 ====================
    print("\n[5/5] 生成汇总报告...")

    if results_list:
        sorted_results = generate_summary(results_list)

        # 筛选MSE模型用于对比图
        mse_results = [r for r in results_list if r["task"]["loss"] == "MSE"]
        if mse_results:
            # 创建汇总目录（用于保存对比图）
            summary_dir = Path(base_config.save_path)
            plot_mse_models_comparison(mse_results, summary_dir)

        # 保存汇总结果到文件
        summary_path = Path(base_config.save_path) / "batch_train_summary.json"
        summary_data = {
            "total_time_minutes": total_time / 60,
            "task_count": len(TRAIN_TASKS),
            "success_count": len(results_list),
            "results": [
                {
                    "task": r["task"],
                    "test_rmse": float(r["test_rmse"]),
                    "test_mae": float(r["test_mae"]),
                    "test_r2": float(r["test_r2"]),
                    "test_bias": float(r["test_bias"]),
                    "best_epoch": r["best_epoch"],
                    "save_dir": r["save_dir"],
                }
                for r in sorted_results
            ],
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 汇总报告已保存: {summary_path}")

    else:
        print("⚠ 没有成功完成的任务")

    print("\n" + "=" * 80)
    print("批量训练完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
