"""
GAT_SeparateEncoder 超参数自动调优入口

基于 Optuna 的贝叶斯优化框架，自动搜索最优超参数组合。

使用方法:
    # 快速模式 (10-20 trials, 约45-90分钟)
    python tune.py --mode quick --n_trials 15

    # 标准模式 (40-60 trials, 约4-8小时)
    python tune.py --mode default --n_trials 50

    # 全面模式 (80-120 trials, 约8-15小时)
    python tune.py --mode comprehensive --n_trials 100

    # 恢复之前的 study 继续优化
    python tune.py --mode default --n_trials 50 --resume

    # 多进程并行
    python tune.py --mode default --n_trials 50 --n_jobs 4

    # 查看最佳结果
    python tune.py --show-best

    # 生成可视化图表
    python tune.py --visualize

作者: GNN气温预测项目
日期: 2026
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from myGNN.tuner.search_space import SearchSpaceFactory
from myGNN.tuner.objective import create_objective
from myGNN.tuner.visualize_tuning import (
    visualize_results,
    save_best_config,
    print_best_summary,
)

# 输出目录
RESULTS_DIR = project_root / "myGNN" / "tuning_results"
DB_PATH = RESULTS_DIR / "optuna_study.db"


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="GAT_SeparateEncoder 超参数自动调优",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速模式
  python tune.py --mode quick --n_trials 15

  # 标准模式 + 4进程并行
  python tune.py --mode default --n_trials 50 --n_jobs 4

  # 恢复之前的优化
  python tune.py --mode default --n_trials 50 --resume

  # 查看最佳结果
  python tune.py --show-best

  # 生成可视化
  python tune.py --visualize
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "default", "comprehensive"],
        default="default",
        help="搜索模式 (default: default)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="试验次数 (default: quick=15, default=50, comprehensive=100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="恢复之前的 study 继续优化",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="并行进程数 (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="最大运行时间（秒）(default: 无限制)",
    )
    parser.add_argument(
        "--show-best",
        action="store_true",
        help="显示最佳配置并退出",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成可视化图表并退出",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Study 名称 (default: gat_tuning_{mode})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子 (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备 (default: auto)",
    )

    return parser.parse_args()


def get_default_n_trials(mode: str) -> int:
    """根据模式返回默认试验次数。"""
    defaults = {"quick": 15, "default": 50, "comprehensive": 100}
    return defaults[mode]


def create_study(mode: str, study_name: str, resume: bool) -> optuna.Study:
    """创建或恢复 Optuna Study。"""
    # 剪枝器
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=10,
    )

    # 采样器
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,
        seed=42,
    )

    # 数据库存储
    storage = f"sqlite:///{DB_PATH}"

    if resume:
        # 恢复已有 study
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
            )
            print(f"✓ 恢复已有 Study: {study_name}")
            print(f"  已完成 trials: {len(study.trials)}")
            return study
        except KeyError:
            print(f"⚠ 未找到已有 Study '{study_name}'，将创建新的")

    # 创建新 study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    print(f"✓ 创建新 Study: {study_name}")
    return study


def show_best():
    """显示已有的最佳配置。"""
    if not DB_PATH.exists():
        print(f"✗ 数据库文件不存在: {DB_PATH}")
        print("  请先运行超参数优化。")
        return

    storage = f"sqlite:///{DB_PATH}"
    studies = optuna.get_all_study_names(storage)
    if not studies:
        print("✗ 数据库中没有找到任何 study。")
        return

    print(f"可用的 studies: {studies}")

    # 加载最后一个 study
    study_name = studies[-1]
    study = optuna.load_study(study_name=study_name, storage=storage)

    if len(study.trials) == 0:
        print("✗ 该 study 没有完成的 trials。")
        return

    print_best_summary(study)

    # 也尝试加载 JSON
    best_json = RESULTS_DIR / "best_config.json"
    if best_json.exists():
        print(f"\n从 {best_json} 加载:")
        with open(best_json, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(json.dumps(config, indent=2, ensure_ascii=False))


def run_visualize():
    """生成可视化图表。"""
    if not DB_PATH.exists():
        print(f"✗ 数据库文件不存在: {DB_PATH}")
        return

    storage = f"sqlite:///{DB_PATH}"
    studies = optuna.get_all_study_names(storage)
    if not studies:
        print("✗ 数据库中没有找到任何 study。")
        return

    study_name = studies[-1]
    study = optuna.load_study(study_name=study_name, storage=storage)

    if len(study.trials) == 0:
        print("✗ 该 study 没有完成的 trials。")
        return

    print(f"正在为 Study '{study_name}' 生成可视化...")
    visualize_results(study, output_dir=str(RESULTS_DIR / "visualizations"))
    save_best_config(study, output_dir=str(RESULTS_DIR))


def main():
    """主函数。"""
    args = parse_args()

    # 处理 --show-best
    if args.show_best:
        show_best()
        return

    # 处理 --visualize
    if args.visualize:
        run_visualize()
        return

    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 确定参数
    mode = args.mode
    n_trials = (
        args.n_trials if args.n_trials is not None else get_default_n_trials(mode)
    )
    if n_trials <= 0:
        print(f"✗ 试验次数必须 > 0，当前值: {n_trials}")
        sys.exit(1)
    study_name = args.study_name or f"gat_tuning_{mode}"

    print("=" * 70)
    print("GAT_SeparateEncoder 超参数自动调优")
    print("=" * 70)
    print(f"  搜索模式: {mode}")
    print(f"  试验次数: {n_trials}")
    print(f"  Study 名称: {study_name}")
    print(f"  并行进程: {args.n_jobs}")
    print(f"  基础种子: {args.seed}")
    if args.timeout:
        print(f"  超时限制: {args.timeout} 秒")
    print(f"  结果目录: {RESULTS_DIR}")
    print(f"  搜索参数: {SearchSpaceFactory.get_param_names(mode)}")
    print("=" * 70)

    # 确保输出目录存在
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 创建 Study
    study = create_study(mode, study_name, args.resume)

    # 创建目标函数
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=10,
    )
    objective = create_objective(
        mode=mode,
        base_seed=args.seed,
        pruner=pruner,
    )

    # 运行优化
    print(f"\n开始优化...")
    print(f"  目标: 最小化验证集 RMSE")
    print(f"  采样器: TPE (Tree-structured Parzen Estimator)")
    print(f"  剪枝器: MedianPruner (n_startup=5, warmup=20)")
    print()

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # 输出结果
    print("\n" + "=" * 70)
    print("优化完成!")
    print("=" * 70)
    print(f"  总 trials: {len(study.trials)}")
    print(
        f"  完成: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )
    print(
        f"  剪枝: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    print(
        f"  失败: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}"
    )

    # 打印最佳配置
    print_best_summary(study)

    # 保存结果
    save_best_config(study, output_dir=str(RESULTS_DIR))

    # 生成可视化
    print("\n正在生成可视化图表...")
    visualize_results(study, output_dir=str(RESULTS_DIR / "visualizations"))

    print("\n" + "=" * 70)
    print("调优完成! 结果已保存到:")
    print(f"  数据库: {DB_PATH}")
    print(f"  最佳配置: {RESULTS_DIR / 'best_config.json'}")
    print(f"  Top-10 配置: {RESULTS_DIR / 'top10_configs.json'}")
    print(f"  试验记录: {RESULTS_DIR / 'trials_dataframe.csv'}")
    print(f"  可视化图表: {RESULTS_DIR / 'visualizations/'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
