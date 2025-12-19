"""
myGNN图结构模块

包含所有图构建和管理功能:
- distance_graph.py: 基于距离的图构建（K近邻、逆距离权重、空间相似性）
- dynamic_graph.py: 动态图学习模块
"""

from .distance_graph import (
    Graph_full,
    Graph_knn,
    Graph_inv_dis,
    Graph_spatial_similarity,
    load_graph_from_station_info,
    create_spatial_similarity_graph,
    create_graph_from_config,
)


__all__ = [
    # 静态图类
    'Graph_full',
    'Graph_knn',
    'Graph_inv_dis',
    'Graph_spatial_similarity',
    # 图构建函数
    'load_graph_from_station_info',
    'create_spatial_similarity_graph',
    'create_graph_from_config'
]
