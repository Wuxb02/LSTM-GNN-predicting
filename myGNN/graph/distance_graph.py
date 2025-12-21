"""
myGNN图结构构建模块

功能：
1. 基于气象站经纬度构建空间图结构
2. 支持多种图类型（全连接、K近邻、带权重边）
3. 提供便捷的加载函数

图类型：
- Graph_full: 全连接图
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


class Graph_full:
    """
    全连接图

    每个节点与所有其他节点相连（包括自环）
    """

    def __init__(self, num_nodes):
        """
        初始化全连接图

        Args:
            num_nodes: 节点数量
        """
        self.node_num = num_nodes
        self.edge_form = 'full'
        self.use_edge_attr = False

        # 创建全连接边
        # meshgrid生成所有节点对
        edge_index_np = np.array(
            np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))
        ).reshape(2, -1)

        self.edge_index = torch.tensor(edge_index_np, dtype=torch.long)


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


class Graph_correlation_climate:
    """
    基于气温相关性拓扑和气候统计量的图构建

    核心理念: 动态拓扑 + 静态气质
    - 拓扑: 使用去季节化气温相关性定义邻居
    - 权重: 使用气候统计量(均值/标准差/最大/最小)计算GeoGAT权重

    参考:
    - 动态拓扑: 皮尔逊相关系数 + 去季节化处理
    - 静态气质: 4维统计量(均值/标准差/最大值/最小值)
    - 权重计算: GeoGAT框架(局部相似性 + 邻域相似性)
    """

    def __init__(self, MetData, station_info, config):
        """
        初始化correlation_climate图构建器

        Args:
            MetData: [total_len, num_stations, num_features] 气象数据
            station_info: [num_stations, 4] 站点信息
            config: Config对象
        """
        self.MetData = MetData
        self.config = config
        self.num_stations = MetData.shape[1]

        # 提取站点信息
        self.lon = station_info[:, 0]
        self.lat = station_info[:, 1]

        # 图参数
        self.top_k = min(config.correlation_top_k, self.num_stations - 1)
        self.alpha = config.correlation_climate_alpha
        self.edge_form = f'corr_climate_k{self.top_k}_alpha{self.alpha}'
        self.use_edge_attr = True

    def _deseasonalize(self, tmax_series):
        """
        去季节化处理: 按月份减去多年平均值

        Args:
            tmax_series: [train_len, num_stations]
        Returns:
            residuals: [train_len, num_stations] 残差序列
        """
        train_len, num_stations = tmax_series.shape

        # 验证数据量
        if train_len < 30:
            raise ValueError(
                f"训练集数据不足: {train_len}天 < 30天最低要求"
            )

        # 提取月份信息(从原始数据的索引27)
        train_start = self.config.train_start
        train_end = self.config.train_end
        month_feature = self.MetData[train_start:train_end, :, 27]

        # 初始化残差数组
        residuals = np.zeros_like(tmax_series)

        # 按月份去季节化
        for month in range(1, 13):
            month_mask = (month_feature[:, 0] == month)
            month_count = np.sum(month_mask)

            if month_count < 3:
                # 样本不足,保持原值
                residuals[month_mask] = tmax_series[month_mask]
                print(f"  警告: {month}月样本数{month_count}<3,跳过去季节化")
            else:
                # 减去月均值
                monthly_mean = np.mean(tmax_series[month_mask], axis=0)
                residuals[month_mask] = tmax_series[month_mask] - monthly_mean

        return residuals

    def _compute_correlation_neighbors(self, tmax_train):
        """
        计算相关性邻居集合

        Args:
            tmax_train: [train_len, num_stations]
        Returns:
            corr_matrix: [num_stations, num_stations] 相关系数矩阵
            neighbor_indices: [num_stations, K] Top-K邻居索引
            neighbor_correlations: [num_stations, K] 对应相关系数
        """
        num_stations = tmax_train.shape[1]
        K = self.top_k

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(tmax_train.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, -1)  # 避免自环

        # 选择Top-K (基于绝对值)
        neighbor_indices = np.zeros((num_stations, K), dtype=int)
        neighbor_correlations = np.zeros((num_stations, K))

        for i in range(num_stations):
            abs_corr = np.abs(corr_matrix[i])
            top_k_idx = np.argsort(abs_corr)[::-1][:K]
            neighbor_indices[i] = top_k_idx
            neighbor_correlations[i] = corr_matrix[i, top_k_idx]

        return corr_matrix, neighbor_indices, neighbor_correlations

    def _compute_climate_statistics(self, train_data):
        """
        计算静态气候特征均值

        从静态特征中排除x(0)和y(1)坐标,仅使用其他静态特征计算均值

        Args:
            train_data: [train_len, num_stations, num_features]
        Returns:
            stats: [num_stations, num_static_features] 静态特征均值 (排除x和y后)
        """
        # 获取静态特征索引
        static_indices = self.config.static_feature_indices

        # 动态过滤: 排除x(0)和y(1)
        filtered_indices = [idx for idx in static_indices if idx not in [0, 1]]

        # 提取过滤后的静态特征
        static_features = train_data[:, :, filtered_indices]

        # 仅计算均值
        mean_vals = np.mean(static_features, axis=0)  # [num_stations, num_filtered_static]

        print(f"    原始静态特征索引: {static_indices}")
        print(f"    过滤后特征索引 (排除x和y): {filtered_indices}")
        print(f"    使用 {len(filtered_indices)} 个静态特征计算均值")
        print(f"    特征维度: {mean_vals.shape}")

        return mean_vals

    def _compute_local_similarity(self, stats):
        """
        计算局部相似性 S_l(i,j)

        使用标准化特征的高斯核: S_l(i,j) = exp(-Σ_v [(z_v(i) - z_v(j))]² / num_features)

        Args:
            stats: [num_stations, num_features]
        Returns:
            S_l: [num_stations, num_stations]
        """
        num_stations, num_features = stats.shape

        # 标准化特征 (Z-score标准化)
        feature_mean = np.mean(stats, axis=0)
        feature_std = np.std(stats, axis=0)
        feature_std[feature_std == 0] = 1.0  # 避免除零

        standardized_stats = (stats - feature_mean) / feature_std

        # 计算欧氏距离平方
        diff = standardized_stats[:, None, :] - standardized_stats[None, :, :]
        squared_dist = np.sum(diff ** 2, axis=2) / num_features

        # 高斯核
        S_l = np.exp(-squared_dist)

        return S_l

    def _compute_neighborhood_similarity(self, stats, neighbor_indices):
        """
        计算邻域相似性 S_n(N(i), N(j))

        基于标准化特征的相关性邻居邻域均值计算相似性

        Args:
            stats: [num_stations, num_features]
            neighbor_indices: [num_stations, K]
        Returns:
            S_n: [num_stations, num_stations]
        """
        num_stations, K = neighbor_indices.shape
        num_features = stats.shape[1]

        if K == 0:
            return np.zeros((num_stations, num_stations))

        # 先标准化特征 (与局部相似性保持一致)
        feature_mean = np.mean(stats, axis=0)
        feature_std = np.std(stats, axis=0)
        feature_std[feature_std == 0] = 1.0

        standardized_stats = (stats - feature_mean) / feature_std

        # 计算每站邻域的均值 (在标准化空间)
        neighborhood_means = np.zeros((num_stations, num_features))
        for i in range(num_stations):
            neighbors = neighbor_indices[i]
            neighborhood_means[i] = np.mean(standardized_stats[neighbors], axis=0)

        # 计算邻域间的相似性 (已在标准化空间,无需再标准化)
        diff = neighborhood_means[:, None, :] - neighborhood_means[None, :, :]
        squared_dist = np.sum(diff ** 2, axis=2) / num_features
        S_n = np.exp(-squared_dist)

        return S_n

    def _build_graph(self, neighbor_indices, weights):
        """
        构建最终图结构

        Args:
            neighbor_indices: [num_stations, K]
            weights: [num_stations, K]
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1]
        """
        edge_list = []
        weight_list = []

        num_stations, K = neighbor_indices.shape

        for src in range(num_stations):
            for k in range(K):
                dst = neighbor_indices[src, k]
                weight = weights[src, k]
                edge_list.append([src, dst])
                weight_list.append(weight)

        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        edge_attr = torch.tensor(weight_list, dtype=torch.float32).unsqueeze(1)

        # 权重归一化到[0,1]
        if edge_attr.max() > edge_attr.min():
            edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())
        else:
            edge_attr = torch.ones_like(edge_attr)

        return edge_index, edge_attr

    def build_graph(self):
        """
        主构建流程

        Returns:
            graph: PyG Data对象
        """
        print(f"\n=== Building correlation_climate graph ===")

        # 1. 提取训练集数据
        train_data = self.MetData[self.config.train_start:self.config.train_end]
        tmax_train = train_data[:, :, self.config.target_feature_idx]
        print(f"  Training range: [{self.config.train_start}, {self.config.train_end})")
        print(f"  Target feature: index {self.config.target_feature_idx} (tmax)")

        # 2. 去季节化
        print(f"  Step 1: Deseasonalization...")
        tmax_deseasonalized = self._deseasonalize(tmax_train)

        # 3. 计算相关性邻居
        print(f"  Step 2: Computing temperature correlation neighbors (K={self.top_k})...")
        corr_matrix, neighbor_indices, neighbor_corrs = \
            self._compute_correlation_neighbors(tmax_deseasonalized)

        mean_abs_corr = np.mean(np.abs(neighbor_corrs))
        print(f"    Average |correlation|: {mean_abs_corr:.3f}")

        # 4. 计算静态特征均值
        print(f"  Step 3: Computing static feature means (excluding x and y)...")
        climate_stats = self._compute_climate_statistics(train_data)

        # 5. 计算局部相似性
        print(f"  Step 4: Computing local similarity...")
        S_l = self._compute_local_similarity(climate_stats)

        # 6. 计算邻域相似性
        print(f"  Step 5: Computing neighborhood similarity (alpha={self.alpha})...")
        S_n = self._compute_neighborhood_similarity(climate_stats, neighbor_indices)

        # 7. 融合权重
        combined_similarity = S_l + self.alpha * S_n

        # 8. 提取邻居对应的权重
        num_stations, K = neighbor_indices.shape
        weights = np.zeros((num_stations, K))
        for i in range(num_stations):
            for k in range(K):
                j = neighbor_indices[i, k]
                weights[i, k] = combined_similarity[i, j]

        # 9. 构建图
        print(f"  Step 6: Building graph structure...")
        edge_index, edge_attr = self._build_graph(neighbor_indices, weights)

        # 10. 创建PyG Data对象
        graph = Data(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.num_stations
        )

        print(f"  [OK] Graph construction completed:")
        print(f"    - Number of nodes: {graph.num_nodes}")
        print(f"    - Number of edges: {graph.num_edges}")
        print(f"    - Average degree: {graph.num_edges / graph.num_nodes:.2f}")
        print(f"    - Edge weight range: [{edge_attr.min():.3f}, {edge_attr.max():.3f}]")

        return graph


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

    elif config.graph_type == 'full':
        # 全连接图
        graph = Graph_full(config.node_num)
        print(f"✓ 创建全连接图: {config.node_num}个节点")

    elif config.graph_type == 'correlation_climate':
        # 气温相关性拓扑 + 气候统计量图
        if feature_data is None:
            raise ValueError(
                "correlation_climate图需要feature_data参数!\n"
                "请提供气象数据，例如:\n"
                "  MetData = np.load(config.MetData_fp)\n"
                "  graph = create_graph_from_config(config, MetData)"
            )
        graph_builder = Graph_correlation_climate(
            MetData=feature_data,
            station_info=station_info,
            config=config
        )
        graph = graph_builder.build_graph()

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

    # 测试4: 全连接图
    print("\n" + "=" * 80)
    print("测试4: 全连接图")
    print("=" * 80)
    graph4 = Graph_full(config.node_num)
    print(f"✓ 图类型: {graph4.edge_form}")
    print(f"  节点数: {graph4.node_num}")
    print(f"  边数: {graph4.edge_index.shape[1]}")
    print(f"  预期边数: {config.node_num ** 2}")

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
