"""
myGNN图结构构建模块

功能：
1. 基于气象站经纬度构建空间图结构
2. 支持多种图类型（全连接、K近邻、带权重边）
3. 提供便捷的加载函数

图类型：
- Graph_knn: K近邻图（无边权重）
- Graph_inv_dis: K近邻图 + 逆距离权重
- Graph_spatial_similarity: 空间相似性

作者: GNN气温预测项目
日期: 2025
"""

import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch_geometric.data import Data

class Graph_knn:
    """
    K近邻图（无边权重）

    每个节点连接到最近的K个邻居（包括自环）
    """

    def __init__(self, lon, lat, top_neighbors=5):
        """
        初始化K近邻图

        Args:
            lon: 经度数组 [num_stations]
            lat: 纬度数组 [num_stations]
            top_neighbors: K近邻数量（不包括自己）
        """
        self.node_num = len(lon)
        self.edge_form = f'knn_{top_neighbors}'
        self.use_edge_attr = False

        # 计算距离矩阵
        coordinates = np.column_stack([lon, lat])
        distances = distance_matrix(coordinates, coordinates)

        # 构建K近邻边
        edges = []
        for i in range(self.node_num):
            # 找到最近的K+1个节点（包括自己）
            nearest_indices = np.argsort(distances[i, :])[:top_neighbors + 1]
            for idx in nearest_indices:
                edges.append((i, idx))

        edge_index_np = np.array(edges).T
        self.edge_index = torch.tensor(edge_index_np, dtype=torch.long)


class Graph_inv_dis:
    """
    K近邻图 + 逆距离边权重

    每个节点连接到最近的K个邻居，边权重为归一化的逆距离
    """

    def __init__(self, lon, lat, top_neighbors=5):
        """
        初始化带权重的K近邻图

        Args:
            lon: 经度数组 [num_stations]
            lat: 纬度数组 [num_stations]
            top_neighbors: K近邻数量（不包括自己）
        """
        self.node_num = len(lon)
        self.edge_form = f'inv_dis_{top_neighbors}'
        self.use_edge_attr = True

        # 计算距离矩阵
        coordinates = np.column_stack([lon, lat])
        self.distances = distance_matrix(coordinates, coordinates)

        # 计算边索引和权重
        self._compute_edge_weights_and_indices(top_neighbors)

    def _compute_edge_weights_and_indices(self, k):
        """
        计算K近邻的边索引和逆距离权重

        Args:
            k: 邻居数量（不包括自己）
        """
        # 获取每个节点最近的K个邻居（排除自己）
        sorted_indices = np.argsort(self.distances, axis=1)[:, 1:k + 1]
        sorted_distances = np.sort(self.distances, axis=1)[:, 1:k + 1]

        # 计算逆距离权重
        # 添加小常数避免除零
        weights = 1.0 / (sorted_distances + 1e-10)

        # 归一化权重（每个节点的所有出边权重和为1）
        weights_sum = weights.sum(axis=1, keepdims=True)
        # 确保分母不为0
        weights_sum = np.maximum(weights_sum, 1e-10)
        weights = weights / weights_sum

        # 构建边索引
        # source: [0,0,...,0, 1,1,...,1, ..., n-1,n-1,...,n-1]
        # target: [邻居1, 邻居2, ..., 邻居k] for each source node
        edge_index_source = np.repeat(np.arange(self.node_num), k)
        edge_index_target = sorted_indices.flatten()
        edge_index_np = np.vstack([edge_index_source, edge_index_target])

        # 转换为PyTorch张量
        self.edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        self.edge_attr = torch.tensor(weights.flatten(), dtype=torch.float32)


class Graph_spatial_similarity:
    """
    基于空间相似性的图构建（GeoGAT方法）

    参考文献：
    Jiao, Z., & Tao, R. (2025). Geographical Graph Attention Networks:
    Spatial Deep Learning Models for Spatial Prediction and Exploratory
    Spatial Data Analysis. Transactions in GIS, 29:e70029.

    实现论文中的空间相似性计算方法:
    1. 局部相似性 S_l(i,j): 基于节点自身特征的相似性
    2. 邻域相似性 S_n(N(i),N(j)): 基于节点邻域特征的相似性
    3. 总体相似性 S(i,j) = S_l(i,j) + α × S_n(N(i),N(j))

    使用高斯核函数计算特征相似性:
    C_v(i,j) = exp(-(x_v(i) - x_v(j))^2 / (2σ^2))
    """

    def __init__(self, feature_data, lon, lat, top_k=10, alpha=1.0,
                 use_neighborhood=True, initial_neighbors=5):
        """
        初始化基于空间相似性的图

        Args:
            feature_data: 特征数据 [num_stations, num_features]
            lon: 经度数组 [num_stations]
            lat: 纬度数组 [num_stations]
            top_k: 选择最相似的K个邻居（不包括自己）
            alpha: 邻域相似性权重系数（论文默认1.0）
            use_neighborhood: 是否使用邻域相似性
            initial_neighbors: 用于计算邻域相似性的初始空间邻居数
        """
        self.node_num = len(lon)
        self.edge_form = f'spatial_sim_k{top_k}_alpha{alpha}'
        self.use_edge_attr = True

        self.feature_data = feature_data
        self.lon = lon
        self.lat = lat
        self.top_k = top_k
        self.alpha = alpha
        self.use_neighborhood = use_neighborhood
        self.initial_neighbors = initial_neighbors

        # 计算距离矩阵（用于定义初始邻域）
        coordinates = np.column_stack([lon, lat])
        self.distances = distance_matrix(coordinates, coordinates)

        # 计算空间相似性矩阵和构建图
        self._compute_spatial_similarity()
        self._build_graph()

    def _compute_local_similarity(self):
        """
        计算局部空间相似性 S_l(i,j)

        对每个特征使用高斯核: C_v(i,j) = exp(-(x_v(i)-x_v(j))^2 / (2σ^2))
        然后对所有特征求平均

        Returns:
            local_sim: [num_stations, num_stations] 局部相似性矩阵
        """
        num_stations, num_features = self.feature_data.shape

        # 计算每个特征的标准差作为σ
        feature_std = np.std(self.feature_data, axis=0)
        feature_std = np.maximum(feature_std, 1e-8)  # 避免除零

        # 初始化相似性矩阵
        local_sim = np.zeros((num_stations, num_stations))

        # 对每个特征计算相似性
        for v in range(num_features):
            x_v = self.feature_data[:, v][:, np.newaxis]  # [N, 1]
            x_v_T = x_v.T  # [1, N]

            # 计算差值矩阵
            diff = x_v - x_v_T  # [N, N]

            # 高斯核
            sigma_v = feature_std[v]
            C_v = np.exp(-(diff ** 2) / (2 * sigma_v ** 2))

            # 累加
            local_sim += C_v

        # 对所有特征求平均（论文公式2使用P函数，这里使用平均）
        local_sim = local_sim / num_features

        return local_sim

    def _compute_neighborhood_similarity(self, local_sim):
        """
        计算邻域相似性 S_n(N(i), N(j))

        使用空间距离定义初始邻域，然后计算邻域特征均值的相似性

        Args:
            local_sim: [num_stations, num_stations] 局部相似性矩阵

        Returns:
            neighborhood_sim: [num_stations, num_stations] 邻域相似性矩阵
        """
        if not self.use_neighborhood:
            return np.zeros_like(local_sim)

        num_stations = self.node_num

        # 使用空间距离定义初始邻域（K近邻）
        # 对每个站点找到最近的initial_neighbors个邻居（不包括自己）
        neighbor_indices = np.argsort(self.distances, axis=1)[:, 1:self.initial_neighbors + 1]

        # 计算每个站点的邻域特征均值
        neighborhood_means = np.zeros_like(self.feature_data)
        for i in range(num_stations):
            neighbors = neighbor_indices[i]
            neighborhood_means[i] = self.feature_data[neighbors].mean(axis=0)

        # 计算邻域均值之间的相似性（使用相同的高斯核方法）
        num_features = self.feature_data.shape[1]
        feature_std = np.std(self.feature_data, axis=0)
        feature_std = np.maximum(feature_std, 1e-8)

        neighborhood_sim = np.zeros((num_stations, num_stations))

        for v in range(num_features):
            x_v_mean = neighborhood_means[:, v][:, np.newaxis]  # [N, 1]
            x_v_mean_T = x_v_mean.T  # [1, N]

            diff = x_v_mean - x_v_mean_T  # [N, N]
            sigma_v = feature_std[v]
            C_v = np.exp(-(diff ** 2) / (2 * sigma_v ** 2))

            neighborhood_sim += C_v

        neighborhood_sim = neighborhood_sim / num_features

        return neighborhood_sim

    def _compute_spatial_similarity(self):
        """
        计算总体空间相似性

        S(i,j) = S_l(i,j) + α × S_n(N(i), N(j))
        """
        print(f"\n计算空间相似性矩阵...")
        print(f"  特征数量: {self.feature_data.shape[1]}")
        print(f"  站点数量: {self.node_num}")

        # 计算局部相似性
        local_sim = self._compute_local_similarity()
        print(f"  局部相似性范围: [{local_sim.min():.4f}, {local_sim.max():.4f}]")

        # 计算邻域相似性
        neighborhood_sim = self._compute_neighborhood_similarity(local_sim)
        if self.use_neighborhood:
            print(f"  邻域相似性范围: [{neighborhood_sim.min():.4f}, {neighborhood_sim.max():.4f}]")

        # 组合得到总相似性
        self.similarity_matrix = local_sim + self.alpha * neighborhood_sim
        print(f"  总相似性范围: [{self.similarity_matrix.min():.4f}, {self.similarity_matrix.max():.4f}]")

    def _build_graph(self):
        """
        基于相似性矩阵构建图

        对每个节点选择top-K最相似的节点作为邻居
        边权重为相似性值（归一化）
        """
        edges = []
        weights = []

        for i in range(self.node_num):
            # 排除自己，选择top-K最相似的节点
            sim_scores = self.similarity_matrix[i].copy()
            sim_scores[i] = -np.inf  # 排除自己

            # 找到top-K
            top_indices = np.argsort(sim_scores)[::-1][:self.top_k]
            top_similarities = sim_scores[top_indices]

            # 归一化权重（每个节点的出边权重和为1）
            weights_sum = top_similarities.sum()
            if weights_sum > 1e-10:
                normalized_weights = top_similarities / weights_sum
            else:
                normalized_weights = np.ones(self.top_k) / self.top_k

            # 添加边
            for j, weight in zip(top_indices, normalized_weights):
                edges.append((i, j))
                weights.append(weight)

        # 转换为PyTorch张量
        edge_index_np = np.array(edges).T
        self.edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        self.edge_attr = torch.tensor(weights, dtype=torch.float32)

        print(f"✓ 图构建完成:")
        print(f"  边数: {self.edge_index.shape[1]}")
        print(f"  边权重范围: [{self.edge_attr.min():.4f}, {self.edge_attr.max():.4f}]")
        print(f"  平均度数: {self.edge_index.shape[1] / self.node_num:.2f}")



# ==================== 便捷加载函数 ====================


def load_graph_from_station_info(station_info_fp, top_neighbors=5, use_edge_attr=True):
    """
    从气象站信息文件加载图结构

    Args:
        station_info_fp: 气象站信息文件路径 (.npy)
        top_neighbors: K近邻数量
        use_edge_attr: 是否使用边权重

    Returns:
        graph: 图对象（Graph_inv_dis 或 Graph_knn）

    气象站信息格式:
        shape: [num_stations, 4]
        列0: 站点ID
        列1: 经度
        列2: 纬度
        列3: 海拔高度
    """
    # 加载气象站信息
    station_info = np.load(station_info_fp)
    lon = station_info[:, 1]  # 经度
    lat = station_info[:, 2]  # 纬度

    # 根据配置选择图类型
    if use_edge_attr:
        graph = Graph_inv_dis(lon, lat, top_neighbors)
    else:
        graph = Graph_knn(lon, lat, top_neighbors)

    return graph


def create_spatial_similarity_graph(feature_data, station_info_fp, top_k=10,
                                   alpha=1.0, use_neighborhood=True,
                                   initial_neighbors=5):
    """
    创建基于空间相似性的图（GeoGAT方法）

    Args:
        feature_data: 特征数据 [num_stations, num_features]
                     通常使用气象数据的某个时间切片或时间平均
        station_info_fp: 气象站信息文件路径 (.npy)
        top_k: 选择最相似的K个邻居（论文中使用10）
        alpha: 邻域相似性权重系数（论文默认1.0）
        use_neighborhood: 是否使用邻域相似性
        initial_neighbors: 用于计算邻域相似性的初始空间邻居数

    Returns:
        graph: Graph_spatial_similarity对象

    示例:
        >>> # 使用训练数据的时间平均作为特征
        >>> MetData = np.load('data.npy')  # [time, stations, features]
        >>> train_data = MetData[0:2000]
        >>> feature_data = train_data.mean(axis=0)  # [stations, features]
        >>> graph = create_spatial_similarity_graph(
        ...     feature_data, 'station_info.npy', top_k=10
        ... )
    """
    # 加载气象站信息
    station_info = np.load(station_info_fp)
    lon = station_info[:, 1]  # 经度
    lat = station_info[:, 2]  # 纬度

    # 创建空间相似性图
    graph = Graph_spatial_similarity(
        feature_data=feature_data,
        lon=lon,
        lat=lat,
        top_k=top_k,
        alpha=alpha,
        use_neighborhood=use_neighborhood,
        initial_neighbors=initial_neighbors
    )

    return graph


def create_graph_from_config(config, feature_data=None):
    """
    从配置对象创建图

    Args:
        config: 配置对象（需包含graph_type及相关参数）
        feature_data: 特征数据
            - spatial_similarity类型: [num_stations, num_features]
            - correlation_climate类型: [total_len, num_stations, num_features]

    Returns:
        graph: 图对象

    支持的图类型:
        - 'inv_dis': K近邻图 + 逆距离权重
        - 'spatial_similarity': 基于空间相似性的图（需要feature_data）
        - 'correlation_climate': 基于气温相关性拓扑和气候统计量的图（需要feature_data）
        - 'knn': K近邻图（无权重）
        - 'full': 全连接图
    """
    # 加载气象站信息
    station_info = np.load(config.station_info_fp)
    lon = station_info[:, 1]
    lat = station_info[:, 2]

    # 根据配置的图类型创建图
    if config.graph_type == 'inv_dis':
        # K近邻 + 逆距离权重
        graph = Graph_inv_dis(lon, lat, config.top_neighbors)
        print(f"✓ 创建逆距离权重图: K={config.top_neighbors}")

    elif config.graph_type == 'spatial_similarity':
        # 空间相似性图
        if feature_data is None:
            raise ValueError(
                "空间相似性图需要feature_data参数!\n"
                "请提供特征数据，例如:\n"
                "  feature_data = MetData[train_start:train_end, :, :24].mean(axis=0)"
            )
        graph = Graph_spatial_similarity(
            feature_data=feature_data,
            lon=lon,
            lat=lat,
            top_k=config.spatial_sim_top_k,
            alpha=config.spatial_sim_alpha,
            use_neighborhood=config.spatial_sim_use_neighborhood,
            initial_neighbors=config.spatial_sim_initial_neighbors
        )
        print(f"✓ 创建空间相似性图: K={config.spatial_sim_top_k}, α={config.spatial_sim_alpha}")

    elif config.graph_type == 'knn':
        # K近邻图（无权重）
        graph = Graph_knn(lon, lat, config.top_neighbors)
        print(f"✓ 创建K近邻图: K={config.top_neighbors}")


    else:
        raise ValueError(
            f"未知的图类型: {config.graph_type}\n"
            f"支持的类型: 'inv_dis', 'spatial_similarity', 'correlation_climate', 'knn', 'full'"
        )

    return graph


# ==================== 测试代码 ====================


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append('..')

    from config import create_config

    # 创建配置
    config, arch_config = create_config()

    # 测试1: 使用配置加载图
    print("=" * 80)
    print("测试1: 从配置加载图（逆距离权重）")
    print("=" * 80)
    graph = create_graph_from_config(config)

    print(f"✓ 图类型: {graph.edge_form}")
    print(f"  节点数: {graph.node_num}")
    print(f"  边索引形状: {graph.edge_index.shape}")
    if hasattr(graph, 'edge_attr'):
        print(f"  边权重形状: {graph.edge_attr.shape}")
        print(f"  边权重统计: min={graph.edge_attr.min():.4f}, "
              f"max={graph.edge_attr.max():.4f}, "
              f"mean={graph.edge_attr.mean():.4f}")

    # 测试2: 直接加载
    print("\n" + "=" * 80)
    print("测试2: 直接从文件加载图（带权重）")
    print("=" * 80)
    graph2 = load_graph_from_station_info(
        config.station_info_fp,
        top_neighbors=5,
        use_edge_attr=True
    )
    print(f"✓ 图类型: {graph2.edge_form}")
    print(f"  使用边属性: {graph2.use_edge_attr}")

    # 测试3: 无权重图
    print("\n" + "=" * 80)
    print("测试3: K近邻图（无权重）")
    print("=" * 80)
    graph3 = load_graph_from_station_info(
        config.station_info_fp,
        top_neighbors=5,
        use_edge_attr=False
    )
    print(f"✓ 图类型: {graph3.edge_form}")
    print(f"  使用边属性: {graph3.use_edge_attr}")

    # 测试5: 空间相似性图（GeoGAT方法）
    print("\n" + "=" * 80)
    print("测试5: 空间相似性图（GeoGAT方法）")
    print("=" * 80)
    try:
        # 加载数据
        MetData = np.load(config.MetData_fp)
        print(f"✓ 数据形状: {MetData.shape}")

        # 使用训练集的时间平均作为特征
        # 移除时间特征（doy和month），使用前24个特征
        train_data = MetData[config.train_start:config.train_end, :, :24]
        feature_data = train_data.mean(axis=0)  # [stations, 24]
        print(f"✓ 特征数据形状: {feature_data.shape}")

        # 创建空间相似性图
        graph5 = create_spatial_similarity_graph(
            feature_data=feature_data,
            station_info_fp=config.station_info_fp,
            top_k=10,
            alpha=1.0,
            use_neighborhood=True,
            initial_neighbors=5
        )

        print(f"\n✓ 图创建成功!")
        print(f"  图类型: {graph5.edge_form}")
        print(f"  节点数: {graph5.node_num}")
        print(f"  边索引形状: {graph5.edge_index.shape}")
        print(f"  边权重形状: {graph5.edge_attr.shape}")

        # 对比不同图的边权重分布
        print(f"\n【图类型对比】")
        print(f"  逆距离图边权重: min={graph.edge_attr.min():.4f}, "
              f"max={graph.edge_attr.max():.4f}, "
              f"mean={graph.edge_attr.mean():.4f}")
        print(f"  空间相似性图边权重: min={graph5.edge_attr.min():.4f}, "
              f"max={graph5.edge_attr.max():.4f}, "
              f"mean={graph5.edge_attr.mean():.4f}")

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print(f"  (如果数据文件不存在，这是正常的)")

    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
