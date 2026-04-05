"""
Optuna 目标函数

将 TrialRunner 包装为 Optuna 可优化的目标函数，支持剪枝和多目标。
"""

from __future__ import annotations

from typing import Optional

import optuna

from myGNN.tuner.search_space import SearchSpaceFactory
from myGNN.tuner.trial_runner import TrialRunner


def create_objective(
    mode: str = "default",
    base_seed: int = 42,
    pruner: Optional[optuna.pruners.BasePruner] = None,
):
    """
    工厂函数：创建 Optuna 目标函数。

    Args:
        mode: 搜索模式 ('quick' / 'default' / 'comprehensive')
        base_seed: 基础随机种子（每个 trial 会加上 trial.number）
        pruner: 剪枝器实例（可选）

    Returns:
        可在 study.optimize() 中使用的目标函数。
    """

    def objective(trial: optuna.Trial) -> float:
        """
        单次 trial 的目标函数。

        Returns:
            验证集 RMSE（越小越好）。
        """
        # 1. 采样超参数
        params = SearchSpaceFactory.suggest_all(trial, mode)

        # 2. 设置 trial 用户属性（便于后续分析）
        trial.set_user_attr("mode", mode)
        trial.set_user_attr("hid_dim", params["hid_dim"])
        trial.set_user_attr("GAT_layer", params["GAT_layer"])
        trial.set_user_attr("heads", params["heads"])
        trial.set_user_attr("batch_size", params.get("batch_size", 32))
        trial.set_user_attr("c_under", params["c_under"])
        trial.set_user_attr("c_over", params["c_over"])
        trial.set_user_attr("trend_weight", params.get("trend_weight", 0.0))

        # 3. 执行 Trial
        seed = base_seed + trial.number
        runner = TrialRunner(params, seed=seed)
        result = runner.run()

        # 4. 报告中间结果（支持剪枝）
        val_losses = result["val_losses"]
        for epoch_idx, val_loss in enumerate(val_losses):
            trial.report(val_loss, step=epoch_idx)
            if pruner is not None and trial.should_prune():
                raise optuna.TrialPruned()

        # 5. 存储额外指标
        trial.set_user_attr("val_rmse", result["val_rmse"])
        trial.set_user_attr("test_rmse", result["test_rmse"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("total_params", result["total_params"])
        trial.set_user_attr("elapsed_seconds", result["elapsed_seconds"])
        trial.set_user_attr("val_mae", result["val_mae"])
        trial.set_user_attr("val_r2", result["val_r2"])
        trial.set_user_attr("test_mae", result["test_mae"])
        trial.set_user_attr("test_r2", result["test_r2"])

        # 6. 返回优化目标：验证集 RMSE
        return result["val_rmse"]

    return objective
