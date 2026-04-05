"""
超参数调优模块 (tuner)

基于Optuna的贝叶斯优化框架，自动搜索GAT_SeparateEncoder最优超参数组合。

模块结构:
    - search_space.py:    搜索空间定义（3种预设模式）
    - trial_runner.py:    单次Trial执行器（封装训练流程）
    - objective.py:       Optuna目标函数
    - visualize_tuning.py: 调优结果可视化

使用方式:
    从入口脚本 tune.py 启动，无需直接调用本模块。
"""

from myGNN.tuner.search_space import SearchSpaceFactory
from myGNN.tuner.trial_runner import TrialRunner
from myGNN.tuner.objective import create_objective
from myGNN.tuner.visualize_tuning import visualize_results

__all__ = [
    "SearchSpaceFactory",
    "TrialRunner",
    "create_objective",
    "visualize_results",
]
