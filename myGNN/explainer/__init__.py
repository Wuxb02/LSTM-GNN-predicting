"""
GNN模型可解释性分析模块

提供时序和空间两个维度的可解释性分析:
- TemporalAnalyzer: 使用Integrated Gradients分析时序特征重要性
- SpatialExplainer: 使用GNNExplainer分析空间关系重要性
- HybridExplainer: 统一接口,整合两种分析方法

支持模型:
- GAT_LSTM
- GSAGE_LSTM
- GAT_SeparateEncoder (v3.0 特征级Cross-Attention + 节点嵌入)

作者: GNN气温预测项目
日期: 2025
"""

from .explainer_config import ExplainerConfig
from .gnn_wrapper import (
    create_gnn_wrapper,
    GATWrapper,
    GSAGEWrapper,
    GATSeparateEncoderWrapper
)
from .temporal_analyzer import TemporalAnalyzer
from .spatial_explainer import SpatialExplainer
from .hybrid_explainer import HybridExplainer
from .utils import filter_by_season, detect_model_type, get_original_dataset

__all__ = [
    'ExplainerConfig',
    'create_gnn_wrapper',
    'GATWrapper',
    'GSAGEWrapper',
    'GATSeparateEncoderWrapper',
    'TemporalAnalyzer',
    'SpatialExplainer',
    'HybridExplainer',
    'filter_by_season',
    'detect_model_type',
    'get_original_dataset',
]
