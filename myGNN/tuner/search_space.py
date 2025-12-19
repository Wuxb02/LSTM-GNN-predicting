"""
搜索空间定义模块

定义超参数搜索空间，支持预设和自定义配置。
基于 Optuna 的参数采样接口。
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class ParamType(Enum):
    """参数类型枚举"""
    CATEGORICAL = "categorical"
    INT = "int"
    FLOAT = "float"


@dataclass
class ParamConfig:
    """单个参数的配置"""
    name: str
    param_type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[int] = None
    log: bool = False


class SearchSpace:
    """
    搜索空间管理类

    管理所有超参数的搜索空间定义，并提供 Optuna 采样接口。
    """

    def __init__(self):
        self.params: Dict[str, ParamConfig] = {}

    def add_categorical(self, name: str, choices: List[Any]):
        """
        添加分类参数

        Args:
            name: 参数名称
            choices: 候选值列表
        """
        self.params[name] = ParamConfig(
            name=name,
            param_type=ParamType.CATEGORICAL,
            choices=choices
        )

    def add_int(self, name: str, low: int, high: int, step: int = 1):
        """
        添加整数参数

        Args:
            name: 参数名称
            low: 最小值
            high: 最大值
            step: 步长
        """
        self.params[name] = ParamConfig(
            name=name,
            param_type=ParamType.INT,
            low=low,
            high=high,
            step=step
        )

    def add_float(self, name: str, low: float, high: float, log: bool = False):
        """
        添加浮点参数

        Args:
            name: 参数名称
            low: 最小值
            high: 最大值
            log: 是否使用对数尺度
        """
        self.params[name] = ParamConfig(
            name=name,
            param_type=ParamType.FLOAT,
            low=low,
            high=high,
            log=log
        )

    def suggest(self, trial) -> Dict[str, Any]:
        """
        从 Optuna trial 采样参数

        Args:
            trial: Optuna Trial 对象

        Returns:
            Dict[str, Any]: 采样的参数字典
        """
        params = {}

        for name, config in self.params.items():
            if config.param_type == ParamType.CATEGORICAL:
                params[name] = trial.suggest_categorical(name, config.choices)
            elif config.param_type == ParamType.INT:
                params[name] = trial.suggest_int(
                    name, int(config.low), int(config.high),
                    step=config.step or 1
                )
            elif config.param_type == ParamType.FLOAT:
                params[name] = trial.suggest_float(
                    name, config.low, config.high,
                    log=config.log
                )

        return params

    def __len__(self) -> int:
        return len(self.params)

    def __repr__(self) -> str:
        return f"SearchSpace({len(self.params)} params: {list(self.params.keys())})"


class PresetSpaces:
    """
    预设搜索空间

    提供三种预设配置: quick, default, comprehensive
    """

    @staticmethod
    def quick() -> SearchSpace:
        """
        快速搜索空间

        适用于快速验证，参数范围小
        推荐试验次数: 10-20
        """
        space = SearchSpace()

        # 训练参数
        space.add_float('lr', 1e-4, 1e-2, log=True)
        space.add_categorical('batch_size', [16, 32, 64])

        # 模型架构
        space.add_categorical('hid_dim', [16, 32, 64])
        space.add_int('GAT_layer', 1, 2)

        return space

    @staticmethod
    def default() -> SearchSpace:
        """
        默认搜索空间

        适用于标准调优，参数范围适中
        推荐试验次数: 50-100
        """
        space = SearchSpace()

        # ========== 训练参数 ==========
        space.add_float('lr', 1e-5, 1e-2, log=True)
        space.add_categorical('batch_size', [16, 32, 64])
        space.add_float('weight_decay', 1e-5, 1e-2, log=True)
        space.add_categorical('optimizer', ['Adam', 'AdamW'])

        # ========== 模型架构 ==========
        space.add_categorical('hid_dim', [16, 32, 64, 128])
        space.add_int('lstm_num_layers', 1, 2)
        space.add_float('lstm_dropout', 0.0, 0.5)

        # ========== GAT参数 ==========
        space.add_int('GAT_layer', 1, 3)
        space.add_categorical('heads', [2, 4, 8])
        space.add_float('intra_drop', 0.1, 0.5)
        space.add_float('inter_drop', 0.1, 0.5)

        # ========== 图结构参数 ==========
        space.add_int('top_neighbors', 5, 15)

        return space

    @staticmethod
    def comprehensive() -> SearchSpace:
        """
        全面搜索空间

        适用于深入调优，参数范围大
        推荐试验次数: 100-200
        """
        space = PresetSpaces.default()

        # ========== 模型选择 ==========
        space.add_categorical('exp_model', [
            # 'GAT_LSTM',
            # 'GSAGE_LSTM',
            # 'LSTM',
            'GAT_SeparateEncoder',
            # 'GSAGE_SeparateEncoder'
        ])

        # ========== 损失函数 ==========
        space.add_categorical('loss_type', ['MSE', 'WeightedTrend'])

        # ========== 图类型 ==========
        space.add_categorical('graph_type', [
            'inv_dis', 'spatial_similarity', 'knn'
        ])

        # ========== 高级架构参数 ==========
        space.add_categorical('use_recurrent_decoder', [True, False])
        space.add_categorical('AF', ['ReLU', 'LeakyReLU', 'GELU'])
        space.add_categorical('norm_type', ['BatchNorm', 'LayerNorm'])

        # ========== 学习率调度器 ==========
        space.add_categorical('scheduler', [
            'CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR'
        ])

        return space


def get_search_space(preset: str = 'default',
                     custom_space: Optional[Dict] = None) -> SearchSpace:
    """
    获取搜索空间

    Args:
        preset: 预设名称 ('quick', 'default', 'comprehensive', 'custom')
        custom_space: 自定义空间字典（当 preset='custom' 时使用）

    Returns:
        SearchSpace: 搜索空间对象

    自定义空间格式示例:
        custom_space = {
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1e-2, 'log': True},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'hid_dim': {'type': 'int', 'low': 16, 'high': 128, 'step': 16}
        }
    """
    if preset == 'quick':
        return PresetSpaces.quick()
    elif preset == 'default':
        return PresetSpaces.default()
    elif preset == 'comprehensive':
        return PresetSpaces.comprehensive()
    elif preset == 'custom':
        return _build_custom_space(custom_space)
    else:
        raise ValueError(f"未知的预设空间: {preset}，"
                         f"支持: quick, default, comprehensive, custom")


def _build_custom_space(space_dict: Dict) -> SearchSpace:
    """
    从字典构建自定义搜索空间

    Args:
        space_dict: 搜索空间字典

    Returns:
        SearchSpace: 搜索空间对象
    """
    if not space_dict:
        raise ValueError("自定义搜索空间不能为空")

    space = SearchSpace()

    for name, config in space_dict.items():
        param_type = config.get('type')

        if param_type == 'categorical':
            space.add_categorical(name, config['choices'])
        elif param_type == 'int':
            space.add_int(
                name,
                config['low'],
                config['high'],
                config.get('step', 1)
            )
        elif param_type == 'float':
            space.add_float(
                name,
                config['low'],
                config['high'],
                config.get('log', False)
            )
        else:
            raise ValueError(f"未知的参数类型: {param_type}，"
                             f"支持: categorical, int, float")

    return space


def print_search_space(space: SearchSpace):
    """
    打印搜索空间信息

    Args:
        space: SearchSpace 对象
    """
    print("=" * 60)
    print("搜索空间配置")
    print("=" * 60)
    print(f"参数数量: {len(space)}")
    print("-" * 60)

    for name, config in space.params.items():
        if config.param_type == ParamType.CATEGORICAL:
            print(f"  {name}: {config.choices}")
        elif config.param_type == ParamType.INT:
            step_str = f", step={config.step}" if config.step != 1 else ""
            print(f"  {name}: [{config.low}, {config.high}]{step_str} (int)")
        elif config.param_type == ParamType.FLOAT:
            log_str = " (log)" if config.log else ""
            print(f"  {name}: [{config.low}, {config.high}]{log_str} (float)")

    print("=" * 60)
