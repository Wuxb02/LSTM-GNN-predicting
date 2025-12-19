"""
试验运行器模块

负责执行单次超参数试验，与现有训练代码集成。
"""

import time
import copy
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, ArchConfig, create_config
from dataset import create_dataloaders
from graph.distance_graph import create_graph_from_config
from network_GNN import (
    get_model, get_optimizer, get_scheduler,
    train, val, get_metric
)


class TrialRunner:
    """
    单次试验运行器

    负责将超参数应用到配置，执行训练，返回评估结果。
    """

    # Config 类参数映射
    CONFIG_PARAMS = {
        'lr', 'batch_size', 'weight_decay', 'epochs', 'early_stop',
        'optimizer', 'scheduler', 'exp_model', 'graph_type',
        'top_neighbors', 'use_edge_attr', 'hist_len', 'pred_len',
        'spatial_sim_top_k', 'spatial_sim_alpha',
        'step_size', 'gamma', 'T_max', 'eta_min', 'patience', 'factor'
    }

    # ArchConfig 类参数映射
    ARCH_PARAMS = {
        'hid_dim', 'MLP_layer', 'AF', 'norm_type', 'dropout',
        'GAT_layer', 'heads', 'intra_drop', 'inter_drop',
        'SAGE_layer', 'aggr',
        'lstm_num_layers', 'lstm_dropout', 'lstm_bidirectional',
        'use_recurrent_decoder', 'decoder_type', 'decoder_num_layers',
        'decoder_dropout', 'decoder_use_context', 'decoder_mlp_layers',
        'fusion_num_heads', 'fusion_use_pre_ln',
        'use_node_embedding', 'node_emb_dim', 'use_skip_connection'
    }

    # LossConfig 类参数映射
    LOSS_PARAMS = {
        'loss_type', 'temp_threshold', 'weight_coef', 'trend_weight',
        'weighted_beta', 'trend_gamma', 'alert_temp', 'c_under',
        'c_over', 'c_default_high'
    }

    def __init__(self, fixed_params: Optional[Dict] = None):
        """
        初始化运行器

        Args:
            fixed_params: 固定参数字典（不被调优覆盖）
        """
        self.fixed_params = fixed_params or {}

        # 图结构缓存
        self._cached_graph = None
        self._graph_params_hash = None

    def _apply_params(self, params: Dict[str, Any]) -> Tuple[Config, ArchConfig]:
        """
        将超参数应用到配置对象

        Args:
            params: 超参数字典

        Returns:
            Tuple[Config, ArchConfig]: 更新后的配置对象
        """
        # 创建基础配置
        config, arch_config = create_config()

        # 合并参数（固定参数优先级高于调优参数）
        merged_params = {**params, **self.fixed_params}

        # 应用参数
        for name, value in merged_params.items():
            if name in self.CONFIG_PARAMS:
                setattr(config, name, value)
            elif name in self.ARCH_PARAMS:
                setattr(arch_config, name, value)
            elif name in self.LOSS_PARAMS:
                setattr(config.loss_config, name, value)

        return config, arch_config

    def _get_graph_hash(self, config: Config) -> str:
        """计算图结构参数的哈希值（用于缓存）"""
        graph_params = (
            config.graph_type,
            config.top_neighbors,
            config.use_edge_attr,
            getattr(config, 'spatial_sim_top_k', None),
            getattr(config, 'spatial_sim_alpha', None)
        )
        return str(hash(graph_params))

    def _get_or_build_graph(self, config: Config):
        """
        获取或构建图结构

        图结构参数变化时重新构建，否则使用缓存
        """
        current_hash = self._get_graph_hash(config)

        if self._graph_params_hash != current_hash or self._cached_graph is None:
            # 准备特征数据（spatial_similarity 图需要）
            feature_data = None
            if config.graph_type == 'spatial_similarity':
                MetData = np.load(config.MetData_fp)
                num_features = 24
                train_data = MetData[config.train_start:config.train_end, :, :num_features]
                feature_data = train_data.mean(axis=0)

            self._cached_graph = create_graph_from_config(config, feature_data=feature_data)
            self._graph_params_hash = current_hash

        return self._cached_graph

    def run_trial(self,
                  params: Dict[str, Any],
                  trial=None,
                  verbose: bool = False) -> Dict[str, Any]:
        """
        执行单次试验

        Args:
            params: 超参数字典
            trial: Optuna Trial 对象（用于剪枝）
            verbose: 是否打印详细信息

        Returns:
            Dict: 结果字典
        """
        start_time = time.time()

        try:
            # 1. 应用参数到配置
            config, arch_config = self._apply_params(params)

            if verbose:
                print(f"\n试验参数: {params}")

            # 2. 设置随机种子
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

            # 3. 获取图结构（使用缓存）
            graph = self._get_or_build_graph(config)

            # 4. 创建数据加载器
            train_loader, val_loader, test_loader, stats = create_dataloaders(
                config, graph
            )
            config.ta_mean = stats['ta_mean']
            config.ta_std = stats['ta_std']

            # 5. 创建模型
            model = get_model(config, arch_config)
            model = model.to(config.device)

            # 6. 设置优化器和调度器
            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)

            # 7. 设置损失函数
            if config.use_enhanced_training:
                from train_enhanced import (
                    get_loss_function,
                    train_epoch as train_enhanced,
                    validate_epoch
                )
                criterion = get_loss_function(config)
                use_enhanced = True
            else:
                criterion = None
                use_enhanced = False

            # 8. 训练循环
            best_val_loss = float('inf')
            best_epoch = 0
            patience_counter = 0

            for epoch in range(1, config.epochs + 1):
                # 训练
                if use_enhanced:
                    train_loss = train_enhanced(
                        model, train_loader, optimizer, scheduler,
                        criterion, config, config.device
                    )
                else:
                    train_loss = train(train_loader, model, optimizer, scheduler, config)

                # 验证
                if use_enhanced:
                    val_loss, val_pred, val_label, _ = validate_epoch(
                        model, val_loader, criterion, config, config.device
                    )
                else:
                    val_loss, val_pred, val_label, _ = val(val_loader, model, config)

                # ReduceLROnPlateau 调度
                if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

                # Optuna 剪枝
                if trial is not None:
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        return {
                            'val_rmse': val_loss,
                            'status': 'pruned',
                            'best_epoch': epoch,
                            'training_time': time.time() - start_time
                        }

                # 保存最佳
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_val_pred = val_pred
                    best_val_label = val_label
                else:
                    patience_counter += 1

                # 早停
                if patience_counter >= config.early_stop:
                    break

            # 9. 计算最终指标
            val_rmse, val_mae, val_r2, val_bias = get_metric(best_val_pred, best_val_label)

            training_time = time.time() - start_time

            return {
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'val_bias': val_bias,
                'best_epoch': best_epoch,
                'training_time': training_time,
                'status': 'completed'
            }

        except Exception as e:
            return {
                'val_rmse': float('inf'),
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time
            }
