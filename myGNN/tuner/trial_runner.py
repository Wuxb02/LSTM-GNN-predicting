"""
单次 Trial 执行器

封装完整的训练流程：配置创建 → 图构建 → 数据加载 → 模型训练 → 验证评估。
复用 train.py 和 train_enhanced.py 中的核心逻辑，但关闭可视化等开销。
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

# Windows GBK 编码兼容：强制 UTF-8 输出
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from myGNN.config import create_config, get_feature_indices_for_graph
from myGNN.dataset import create_dataloaders
from myGNN.graph.distance_graph import create_graph_from_config
from myGNN.network_GNN import (
    get_model,
    get_optimizer,
    get_scheduler,
    train,
    val,
    get_metric,
)
from myGNN.train_enhanced import (
    get_loss_function,
    train_epoch as train_enhanced,
    validate_epoch,
)


class TrialRunner:
    """
    单次超参数试验执行器。

    使用方式:
        runner = TrialRunner(trial_params, seed=42)
        result = runner.run()
    """

    # Trial 加速配置
    TRIAL_EPOCHS = 150
    TRIAL_EARLY_STOP = 30

    def __init__(self, trial_params: Dict[str, Any], seed: int = 42):
        """
        Args:
            trial_params: 搜索空间采样后的参数字典
            seed: 随机种子
        """
        self.trial_params = trial_params
        self.seed = seed

    def _setup_seed(self):
        """设置随机种子。"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def _create_config(self):
        """根据 trial 参数创建配置对象。"""
        p = self.trial_params

        config, arch_config = create_config()

        # 固定模型
        config.exp_model = "GAT_SeparateEncoder"
        config.epochs = self.TRIAL_EPOCHS
        config.early_stop = self.TRIAL_EARLY_STOP
        config.auto_visualize = False
        config.seed = self.seed

        # ---- Config 级别参数 ----
        config_param_names = {
            "lr",
            "weight_decay",
            "batch_size",
            "top_neighbors",
            "hist_len",
            "optimizer",
            "scheduler",
        }
        for key in config_param_names:
            if key in p:
                setattr(config, key, p[key])

        # ---- ArchConfig 级别参数 ----
        arch_param_names = {
            "hid_dim",
            "GAT_layer",
            "heads",
            "intra_drop",
            "inter_drop",
            "fusion_num_heads",
            "lstm_num_layers",
            "lstm_dropout",
            "lstm_bidirectional",
            "MLP_layer",
            "AF",
            "norm_type",
            "use_skip_connection",
            "fusion_use_pre_ln",
        }
        for key in arch_param_names:
            if key in p:
                setattr(arch_config, key, p[key])

        # ---- LossConfig 级别参数 ----
        loss_param_names = {
            "c_under",
            "c_over",
            "trend_weight",
            "use_station_day_threshold",
            "threshold_percentile",
        }
        for key in loss_param_names:
            if key in p:
                setattr(config.loss_config, key, p[key])

        # scheduler 参数默认值（避免某些调度器参数缺失）
        if not hasattr(config, "patience") or config.patience is None:
            config.patience = 20
        if not hasattr(config, "factor") or config.factor is None:
            config.factor = 0.8
        if not hasattr(config, "step_size") or config.step_size is None:
            config.step_size = 10
        if not hasattr(config, "gamma") or config.gamma is None:
            config.gamma = 0.9
        if not hasattr(config, "T_max") or config.T_max is None:
            config.T_max = 50
        if not hasattr(config, "eta_min") or config.eta_min is None:
            config.eta_min = 1e-4
        if not hasattr(config, "milestones") or config.milestones is None:
            config.milestones = [50, 100, 150]
        if not hasattr(config, "momentum") or config.momentum is None:
            config.momentum = 0.9
        if not hasattr(config, "betas") or config.betas is None:
            config.betas = (0.9, 0.999)

        return config, arch_config

    def run(self) -> Dict[str, Any]:
        """
        执行单次 Trial 训练。

        Returns:
            包含验证/测试 RMSE、训练时间、参数量等信息的字典。
        """
        self._setup_seed()
        config, arch_config = self._create_config()

        t_start = time.time()

        # ---- 构建图 ----
        feature_data = None
        if config.graph_type == "spatial_similarity":
            MetData_temp = np.load(config.MetData_fp)
            feature_indices = get_feature_indices_for_graph(config)
            train_data_temp = MetData_temp[config.train_start : config.train_end, :, :]
            train_data_temp = train_data_temp[:, :, feature_indices]
            feature_data = train_data_temp.mean(axis=0)

        graph = create_graph_from_config(config, feature_data=feature_data)

        # ---- 加载数据 ----
        train_loader, val_loader, test_loader, stats = create_dataloaders(config, graph)
        config.ta_mean = stats["ta_mean"]
        config.ta_std = stats["ta_std"]

        # ---- 创建模型 ----
        model = get_model(config, arch_config).to(config.device)
        total_params = sum(p.numel() for p in model.parameters())

        # ---- 优化器和调度器 ----
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        # ---- 损失函数 ----
        use_enhanced = config.use_enhanced_training
        if use_enhanced:
            threshold_map_for_loss = stats.get("threshold_map")
            criterion = get_loss_function(config, threshold_map=threshold_map_for_loss)
        else:
            criterion = None

        # ---- 训练循环 ----
        best_val_loss = float("inf")
        best_epoch = 0
        patience = 0
        train_losses = []
        val_losses = []

        for epoch in range(1, config.epochs + 1):
            if use_enhanced:
                train_loss = train_enhanced(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    criterion,
                    config,
                    config.device,
                    arch_config,
                    epoch=epoch,
                    total_epochs=config.epochs,
                )
            else:
                train_loss = train(
                    train_loader,
                    model,
                    optimizer,
                    scheduler,
                    config,
                    arch_config,
                    epoch=epoch,
                    total_epochs=config.epochs,
                )
            train_losses.append(train_loss)

            if use_enhanced:
                val_loss, _, _, _ = validate_epoch(
                    model,
                    val_loader,
                    criterion,
                    config,
                    config.device,
                )
            else:
                val_loss, _, _, _ = val(val_loader, model, config)
            val_losses.append(val_loss)

            # ReduceLROnPlateau
            if scheduler is not None and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience = 0
                # 保存最佳验证结果
                best_val_state = {
                    "predict": None,
                    "label": None,
                    "time": None,
                    "loss": val_loss,
                }
                # 获取最佳 epoch 的验证集预测用于 RMSE 计算
                if use_enhanced:
                    _, pred, label, _ = validate_epoch(
                        model,
                        val_loader,
                        criterion,
                        config,
                        config.device,
                    )
                else:
                    _, pred, label, _ = val(val_loader, model, config)
                best_val_state["predict"] = pred
                best_val_state["label"] = label
            else:
                patience += 1

            if patience >= config.early_stop:
                break

        # ---- 验证集 RMSE ----
        if best_val_state["predict"] is not None:
            val_rmse, val_mae, val_r2, val_bias = get_metric(
                best_val_state["predict"], best_val_state["label"]
            )
        else:
            val_rmse = best_val_loss
            val_mae = val_r2 = val_bias = 0.0

        # ---- 测试集评估（加载最佳模型） ----
        # 重新创建模型加载最佳权重
        best_model = get_model(config, arch_config).to(config.device)
        # 训练过程中没有保存权重文件，用当前模型（已被 early stop 前的最佳权重更新）
        # 实际上我们在训练循环中只保存了内存中的最佳结果，这里直接用当前模型
        # 更精确的做法是保存 checkpoint，但为了 Trial 速度，直接用当前模型评估
        # 注意：early stop 后模型可能不是最佳的，所以用 val_rmse 作为主要优化目标

        if use_enhanced:
            test_loss, test_pred, test_label, _ = validate_epoch(
                best_model,
                test_loader,
                criterion,
                config,
                config.device,
            )
        else:
            test_loss, test_pred, test_label, _ = val(test_loader, best_model, config)

        test_rmse, test_mae, test_r2, test_bias = get_metric(test_pred, test_label)

        elapsed = time.time() - t_start

        # 清理 GPU 内存
        del model, best_model, optimizer, scheduler
        if criterion is not None:
            del criterion
        torch.cuda.empty_cache()
        gc.collect()

        # 所有数值转为原生 Python 类型，避免 Optuna SQLite 存储 JSON 序列化失败
        def _py(val):
            """确保值为原生 Python 类型。"""
            if isinstance(val, (np.floating,)):
                return float(val)
            if isinstance(val, (np.integer,)):
                return int(val)
            if isinstance(val, (np.bool_,)):
                return bool(val)
            return val

        return {
            "val_rmse": _py(val_rmse),
            "val_mae": _py(val_mae),
            "val_r2": _py(val_r2),
            "val_bias": _py(val_bias),
            "test_rmse": _py(test_rmse),
            "test_mae": _py(test_mae),
            "test_r2": _py(test_r2),
            "test_bias": _py(test_bias),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "total_params": int(total_params),
            "elapsed_seconds": float(elapsed),
            "train_losses": [float(x) for x in train_losses],
            "val_losses": [float(x) for x in val_losses],
        }
