"""
myGNN 超参数调优入口

基于 Optuna 的贝叶斯超参数优化框架。

使用方法:
    1. 修改下方 TuneConfig 中的配置
    2. 运行: python myGNN/tune.py

依赖:
    pip install optuna

作者: GNN气温预测项目
日期: 2025
"""

import time
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from tuner import (
    get_search_space,
    TrialRunner,
    TuningVisualizer,
    print_search_space
)


# ==================== 调优配置 ====================
class TuneConfig:
    """
    调优配置（直接在这里修改）

    配置说明:
    - exp_name: 实验名称，用于保存目录命名
    - n_trials: 总试验次数（推荐: quick=20, default=50-100, comprehensive=100-200）
    - timeout: 超时时间(秒)，None 表示不限制
    - preset_space: 预设搜索空间
        - 'quick': 4个参数，快速验证
        - 'default': 12个参数，标准调优
        - 'comprehensive': 20+参数，深入调优
        - 'custom': 使用下方 custom_space 自定义
    - custom_space: 自定义搜索空间（preset_space='custom' 时生效）
    - fixed_params: 固定参数（不参与调优）
    - pruning_enabled: 是否启用早停剪枝（推荐开启）
    - pruning_warmup: 剪枝热身轮数（前N个试验不剪枝）
    """

    # ========== 基础配置 ==========
    exp_name = "tuning"                    # 实验名称
    n_trials = 100                          # 试验次数
    timeout = None                         # 超时时间(秒)，None 表示不限制

    # ========== 搜索空间配置 ==========
    preset_space = "comprehensive"               # 预设空间: quick/default/comprehensive/custom

    # 自定义搜索空间示例（preset_space='custom' 时生效）
    # 支持的类型: categorical, int, float
    custom_space = {
        # 'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
        # 'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
        # 'hid_dim': {'type': 'int', 'low': 16, 'high': 128, 'step': 16},
        # 'GAT_layer': {'type': 'int', 'low': 1, 'high': 3},
        # 'heads': {'type': 'categorical', 'choices': [2, 4, 8]},
        # 'intra_drop': {'type': 'float', 'low': 0.1, 'high': 0.5},
    }

    # 固定参数（不参与调优，将覆盖调优采样的值）
    fixed_params = {
        # 'exp_model': 'GAT_LSTM',         # 固定模型类型
        # 'graph_type': 'inv_dis',         # 固定图类型
        # 'loss_type': 'MSE',              # 固定损失函数
    }

    # ========== 优化目标配置 ==========
    optimize_metric = "val_rmse"           # 优化指标（目前仅支持 val_rmse）
    direction = "minimize"                 # 优化方向: minimize/maximize

    # ========== 保存配置 ==========
    save_dir = "checkpoints/tuning"        # 结果保存目录
    save_top_k = 3                         # 保存 Top-K 最优模型

    # ========== Optuna 配置 ==========
    pruning_enabled = True                 # 是否启用早停剪枝
    pruning_warmup = 10                    # 剪枝热身轮数


# ==================== 调优器 ====================
class Tuner:
    """
    超参数调优器

    基于 Optuna 的贝叶斯优化
    """

    def __init__(self, config):
        """
        初始化调优器

        Args:
            config: TuneConfig 类
        """
        self.config = config

        # 创建保存目录
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.save_dir = Path(config.save_dir) / f"{config.exp_name}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 获取搜索空间
        self.search_space = get_search_space(
            config.preset_space,
            config.custom_space if config.preset_space == 'custom' else None
        )

        # 创建试验运行器
        self.runner = TrialRunner(fixed_params=config.fixed_params)

        # 创建 Optuna Study
        storage = f"sqlite:///{self.save_dir / 'study.db'}"
        self.study = optuna.create_study(
            study_name=config.exp_name,
            direction=config.direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=config.pruning_warmup,
                n_warmup_steps=5
            ) if config.pruning_enabled else None,
            storage=storage,
            load_if_exists=True
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna 目标函数

        Args:
            trial: Optuna Trial 对象

        Returns:
            float: 优化目标值（验证 RMSE）
        """
        # 从搜索空间采样参数
        params = self.search_space.suggest(trial)

        print(f"\n[试验 {trial.number + 1}/{self.config.n_trials}]")
        print(f"参数: {params}")

        # 运行试验
        result = self.runner.run_trial(
            params=params,
            trial=trial,
            verbose=False
        )

        if result['status'] == 'completed':
            print(f"✓ 完成 | Val RMSE: {result['val_rmse']:.4f}°C | "
                  f"耗时: {result['training_time']:.1f}s")
            return result['val_rmse']
        elif result['status'] == 'pruned':
            print(f"✂ 剪枝 | Epoch {result['best_epoch']} | "
                  f"Val RMSE: {result['val_rmse']:.4f}°C")
            raise optuna.TrialPruned()
        else:
            print(f"✗ 失败 | 错误: {result.get('error', 'Unknown')}")
            return float('inf')

    def tune(self) -> dict:
        """
        执行超参数调优

        Returns:
            dict: 调优结果
        """
        print("=" * 80)
        print("myGNN 超参数调优")
        print("=" * 80)

        print(f"\n【配置信息】")
        print(f"  实验名称: {self.config.exp_name}")
        print(f"  试验次数: {self.config.n_trials}")
        print(f"  搜索空间: {self.config.preset_space}")
        print(f"  剪枝: {'启用' if self.config.pruning_enabled else '禁用'}")
        print(f"  保存目录: {self.save_dir}")

        if self.config.fixed_params:
            print(f"\n【固定参数】")
            for k, v in self.config.fixed_params.items():
                print(f"  {k}: {v}")

        print(f"\n【搜索空间】")
        print_search_space(self.search_space)

        print("\n" + "=" * 80)
        print("开始调优...")
        print("=" * 80)

        start_time = time.time()

        # 运行优化
        self.study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=False
        )

        total_time = time.time() - start_time

        # 生成可视化和报告
        print("\n" + "=" * 80)
        print("生成报告和可视化...")
        print("=" * 80)

        visualizer = TuningVisualizer(self.study, self.save_dir)

        # 生成报告
        report = visualizer.generate_report(self.save_dir / 'tuning_report.txt')
        print(report)

        # 保存最优参数
        best_params = visualizer.save_best_params()

        # 生成可视化
        visualizer.generate_all_plots()

        print(f"\n总耗时: {total_time / 3600:.2f} 小时")
        print(f"结果保存在: {self.save_dir}")
        print("=" * 80)

        return {
            'best_params': best_params.get('params', {}),
            'best_value': best_params.get('value', float('inf')),
            'study': self.study,
            'save_dir': str(self.save_dir)
        }


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 检查 optuna 是否安装
    try:
        import optuna
    except ImportError:
        print("错误: 需要安装 optuna")
        print("请运行: pip install optuna")
        exit(1)

    # 创建调优器并运行
    tuner = Tuner(TuneConfig)
    results = tuner.tune()

    print(f"\n最优参数: {results['best_params']}")
    print(f"最优 RMSE: {results['best_value']:.4f}°C")
