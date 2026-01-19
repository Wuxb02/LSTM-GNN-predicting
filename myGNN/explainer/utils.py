"""
可解释性分析工具函数

作者: GNN气温预测项目
日期: 2025
"""

import torch
import numpy as np
from datetime import datetime
from torch.utils.data import Subset, ConcatDataset


def get_original_dataset(dataset):
    """
    获取原始数据集（处理 Subset 和 ConcatDataset 情况）

    当使用季节筛选时，数据集会被包装为 Subset 对象。
    当合并 val+test 时，数据集会被包装为 ConcatDataset 对象。
    此函数递归获取原始的 WeatherGraphDataset。

    Args:
        dataset: 数据集对象（可能是 Subset、ConcatDataset 或 WeatherGraphDataset）

    Returns:
        WeatherGraphDataset: 原始数据集对象
    """
    if isinstance(dataset, Subset):
        return get_original_dataset(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        # ConcatDataset 包含多个数据集，取第一个的原始数据集
        return get_original_dataset(dataset.datasets[0])
    return dataset


def detect_model_type(model):
    """
    自动检测模型类型

    Args:
        model: PyTorch模型实例

    Returns:
        str: 模型类型 ('GAT_LSTM', 'GSAGE_LSTM', 'GAT_SeparateEncoder')

    Raises:
        ValueError: 如果模型类型不支持
    """
    # 通过类名判断(最可靠的方法)
    model_class_name = type(model).__name__

    if model_class_name == 'GAT_SeparateEncoder':
        return 'GAT_SeparateEncoder'
    elif model_class_name == 'GAT_LSTM':
        return 'GAT_LSTM'
    elif model_class_name == 'GSAGE_LSTM':
        return 'GSAGE_LSTM'
    # 备用方案:通过属性判断
    elif hasattr(model, 'static_encoder') and hasattr(model, 'dynamic_encoder'):
        # GAT_SeparateEncoder特有的属性
        return 'GAT_SeparateEncoder'
    elif hasattr(model, 'GAT_layers'):
        return 'GAT_LSTM'
    elif hasattr(model, 'SAGE_layers'):
        return 'GSAGE_LSTM'
    else:
        raise ValueError(
            f"不支持的模型类型: {model_class_name}. "
            f"支持的类型: GAT_LSTM, GSAGE_LSTM, GAT_SeparateEncoder"
        )


def filter_by_season(dataset, season, config):
    """
    根据季节筛选数据集样本

    Args:
        dataset: WeatherGraphDataset实例
        season: 季节 ('spring', 'summer', 'autumn', 'winter', None)
        config: Config配置对象(需要包含数据时间信息)

    Returns:
        list: 筛选后的样本索引列表

    季节定义:
        - spring: 3-5月
        - summer: 6-8月
        - autumn: 9-11月
        - winter: 12-2月
    """
    if season is None:
        return list(range(len(dataset)))

    season_months = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }

    if season not in season_months:
        raise ValueError(
            f"无效的季节: {season}. "
            f"可选值: {list(season_months.keys())}"
        )

    target_months = season_months[season]
    filtered_indices = []

    # 尝试从数据集提取时间信息
    for idx in range(len(dataset)):
        try:
            data, time_idx = dataset[idx]
            # 根据时间索引计算月份
            # 假设time_idx是从数据集开始的索引,需要转换为实际月份
            month = extract_month_from_index(time_idx.item(), config)
            if month in target_months:
                filtered_indices.append(idx)
        except Exception as e:
            # 如果无法提取时间信息,警告并跳过筛选
            print(f"警告: 无法提取时间信息,将分析所有样本. 错误: {e}")
            return list(range(len(dataset)))

    print(f"季节筛选({season}): 从{len(dataset)}个样本中筛选出{len(filtered_indices)}个样本")
    return filtered_indices


def extract_month_from_index(time_idx, config):
    """
    从时间索引提取月份

    根据数据集的时间索引计算对应的月份。
    默认数据集从2010年1月1日开始,每天一个样本。

    Args:
        time_idx (int): 时间索引(从数据集开始计算)
        config: Config配置对象
            - 可选属性 data_start_date: 数据集起始日期 (datetime 或 str 格式 "YYYY-MM-DD")

    Returns:
        int: 月份(1-12)

    数据集时间范围 (默认):
        - 2010-2015年(训练): 索引0-2190
        - 2016年(验证): 索引2191-2556
        - 2017年(测试): 索引2557-2921
    """
    from datetime import datetime, timedelta

    # 从配置获取数据集起始日期，默认为 2010-01-01
    if hasattr(config, 'data_start_date'):
        base_date = config.data_start_date
        if isinstance(base_date, str):
            base_date = datetime.strptime(base_date, '%Y-%m-%d')
    else:
        base_date = datetime(2010, 1, 1)

    # 每个样本对应一天
    actual_date = base_date + timedelta(days=int(time_idx))

    return actual_date.month


def normalize_importance(importance_scores):
    """
    归一化重要性得分到[0, 1]

    Args:
        importance_scores: torch.Tensor或numpy.ndarray

    Returns:
        归一化后的重要性得分
    """
    if isinstance(importance_scores, torch.Tensor):
        min_val = importance_scores.min()
        max_val = importance_scores.max()
        if max_val - min_val < 1e-10:
            return torch.zeros_like(importance_scores)
        return (importance_scores - min_val) / (max_val - min_val)
    else:
        min_val = np.min(importance_scores)
        max_val = np.max(importance_scores)
        if max_val - min_val < 1e-10:
            return np.zeros_like(importance_scores)
        return (importance_scores - min_val) / (max_val - min_val)


def get_top_k_edges(edge_importance, edge_index, k=20):
    """
    获取Top-K重要边

    Args:
        edge_importance: torch.Tensor [num_edges] 边重要性得分
        edge_index: torch.Tensor [2, num_edges] 边索引
        k: int 返回的边数量

    Returns:
        list: [(src, dst, importance), ...] Top-K重要边列表
    """
    k = min(k, len(edge_importance))
    top_values, top_indices = torch.topk(edge_importance, k=k)

    top_edges = []
    for idx, value in zip(top_indices, top_values):
        src, dst = edge_index[:, idx]
        top_edges.append((int(src), int(dst), float(value)))

    return top_edges


# ============ GAT注意力权重分析工具函数 ============

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两点间Haversine球面距离

    使用Haversine公式计算地球表面两点之间的大圆距离。

    Args:
        lon1 (float): 点1的经度 (度)
        lat1 (float): 点1的纬度 (度)
        lon2 (float): 点2的经度 (度)
        lat2 (float): 点2的纬度 (度)

    Returns:
        float: 两点之间的距离 (公里)

    公式:
        a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
        c = 2 × atan2(√a, √(1−a))
        d = R × c  (R=6371km 地球平均半径)

    References:
        https://en.wikipedia.org/wiki/Haversine_formula
    """
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # 地球平均半径 (公里)
    radius = 6371.0

    return radius * c


def compute_edge_distances(edge_index, station_coords):
    """
    批量计算所有边的物理距离

    Args:
        edge_index (np.ndarray): [2, num_edges] 边索引
        station_coords (np.ndarray): [num_stations, 2] 气象站坐标 [lon, lat]

    Returns:
        np.ndarray: [num_edges] 每条边的距离 (公里)

    Example:
        >>> edge_index = np.array([[0, 1], [1, 2]])
        >>> coords = np.array([[113.0, 23.0], [113.5, 23.5], [114.0, 24.0]])
        >>> distances = compute_edge_distances(edge_index, coords)
        >>> print(distances.shape)
        (2,)
    """
    num_edges = edge_index.shape[1]
    distances = np.zeros(num_edges)

    for i in range(num_edges):
        src, dst = edge_index[:, i]
        lon1, lat1 = station_coords[src]
        lon2, lat2 = station_coords[dst]
        distances[i] = haversine_distance(lon1, lat1, lon2, lat2)

    return distances


def compute_temperature_correlation(weather_data, train_indices,
                                   target_feature_idx=4):
    """
    计算训练集所有气象站对的温度相关性矩阵

    使用皮尔逊相关系数衡量不同气象站温度时间序列的相似性。
    仅使用训练集数据以避免数据泄露。

    Args:
        weather_data (np.ndarray): [time_steps, num_stations, features] 气象数据
        train_indices (tuple): (start, end) 训练集索引范围
        target_feature_idx (int): 目标特征索引 (默认4=tmax最高温度)

    Returns:
        np.ndarray: [num_stations, num_stations] 皮尔逊相关系数矩阵
            - 对角线为1 (自相关)
            - 矩阵对称
            - 取值范围: [-1, 1]

    Note:
        - 相关系数接近1: 温度模式高度一致
        - 相关系数接近0: 温度模式无相关
        - 相关系数接近-1: 温度模式负相关(罕见)

    Example:
        >>> weather_data = np.load('data/real_weather_data_2010_2017.npy')
        >>> train_indices = (0, 2191)  # 2010-2015年
        >>> corr_matrix = compute_temperature_correlation(
        ...     weather_data, train_indices, target_feature_idx=4
        ... )
        >>> print(corr_matrix.shape)
        (28, 28)
    """
    start, end = train_indices

    # 提取训练集目标特征: [num_days, num_stations]
    train_temp = weather_data[start:end, :, target_feature_idx]

    # 转置为 [num_stations, num_days]
    train_temp = train_temp.T

    # 计算相关系数矩阵 [num_stations, num_stations]
    corr_matrix = np.corrcoef(train_temp)

    return corr_matrix


def extract_edge_correlations(edge_index, corr_matrix):
    """
    从相关系数矩阵提取边级相关系数

    Args:
        edge_index (np.ndarray): [2, num_edges] 边索引
        corr_matrix (np.ndarray): [num_stations, num_stations] 相关系数矩阵

    Returns:
        np.ndarray: [num_edges] 每条边的相关系数

    Example:
        >>> edge_index = np.array([[0, 1], [1, 2]])
        >>> corr_matrix = np.array([[1.0, 0.8, 0.6],
        ...                          [0.8, 1.0, 0.9],
        ...                          [0.6, 0.9, 1.0]])
        >>> edge_corrs = extract_edge_correlations(edge_index, corr_matrix)
        >>> print(edge_corrs)
        [0.8 0.9]
    """
    num_edges = edge_index.shape[1]
    edge_corrs = np.zeros(num_edges)

    for i in range(num_edges):
        src, dst = edge_index[:, i]
        edge_corrs[i] = corr_matrix[src, dst]

    return edge_corrs


def edge_attention_to_matrix(edge_index, attention_weights, num_nodes,
                            aggregation='mean'):
    """
    将边级注意力转换为节点级注意力矩阵

    Args:
        edge_index (np.ndarray): [2, num_edges] 边索引
        attention_weights (np.ndarray): [num_edges] 边级注意力权重
        num_nodes (int): 节点数量
        aggregation (str): 多边聚合策略
            - 'mean': 平均 (默认, 适用于大多数情况)
            - 'sum': 求和 (放大重要性)
            - 'max': 最大值 (保留最强注意力)

    Returns:
        np.ndarray: [num_nodes, num_nodes] 注意力矩阵
            - matrix[i, j]: 节点i对节点j的注意力强度
            - 无连接位置为0
            - 稀疏矩阵 (大部分元素为0)

    Note:
        - 如果同一节点对(i, j)有多条边，按aggregation策略聚合
        - K近邻图通常无重复边，但此函数提供健壮性保证

    Example:
        >>> edge_index = np.array([[0, 1, 0], [1, 0, 2]])
        >>> attention_weights = np.array([0.8, 0.6, 0.7])
        >>> matrix = edge_attention_to_matrix(edge_index, attention_weights,
        ...                                   num_nodes=3, aggregation='mean')
        >>> print(matrix)
        [[0.  0.8 0.7]
         [0.6 0.  0. ]
         [0.  0.  0. ]]
    """
    attention_matrix = np.zeros((num_nodes, num_nodes))

    if aggregation == 'mean':
        count_matrix = np.zeros((num_nodes, num_nodes))

    # 填充注意力矩阵
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        weight = attention_weights[i]

        if aggregation == 'mean':
            attention_matrix[src, dst] += weight
            count_matrix[src, dst] += 1
        elif aggregation == 'sum':
            attention_matrix[src, dst] += weight
        elif aggregation == 'max':
            attention_matrix[src, dst] = max(
                attention_matrix[src, dst], weight
            )

    # 平均聚合
    if aggregation == 'mean':
        nonzero_mask = count_matrix > 0
        attention_matrix[nonzero_mask] /= count_matrix[nonzero_mask]

    return attention_matrix



def tsne_reduce_embeddings(embeddings, random_state=42, perplexity=10,
                            n_iter=1000, learning_rate='auto'):
    """
    使用t-SNE算法将高维节点嵌入降维到2D平面

    t-SNE (t-Distributed Stochastic Neighbor Embedding) 是一种非线性
    降维算法，擅长保留数据的局部结构，适合可视化高维数据的聚类模式。

    Args:
        embeddings: [num_nodes, embedding_dim] 高维节点嵌入
        random_state: int, 随机种子，用于结果可重复性 (默认: 42)
        perplexity: float, t-SNE超参数 (默认: 10)
            - 平衡局部和全局结构的重要性
            - 推荐范围: 5 ~ 50
            - 对小数据集 (num_nodes < 50): perplexity ≈ num_nodes / 3
            - 对28个节点: 建议 perplexity=9
        n_iter: int, 最大迭代次数 (默认: 1000)
        learning_rate: float or 'auto', 学习率 (默认: 'auto')
            - 'auto': 自动设置为 max(200, num_samples / 12)

    Returns:
        np.ndarray: [num_nodes, 2] 2D投影坐标
            - 列0: t-SNE维度1
            - 列1: t-SNE维度2

    注意:
        - t-SNE是非凸优化，不同运行结果可能略有差异（通过random_state保证可重复性）
        - 对于小数据集（< 50个样本），降维效果可能不稳定
        - perplexity 不能大于 num_samples - 1

    示例:
        >>> embeddings = np.random.randn(28, 4)
        >>> tsne_2d = tsne_reduce_embeddings(embeddings, perplexity=9)
        >>> print(tsne_2d.shape)  # (28, 2)
    """
    from sklearn.manifold import TSNE

    # 参数验证
    num_nodes = embeddings.shape[0]
    if perplexity >= num_nodes:
        perplexity = max(5, num_nodes // 3)
        print(f"  ⚠ perplexity过大，自动调整为 {perplexity}")

    # 创建t-SNE对象（兼容不同版本的scikit-learn）
    try:
        # 新版本sklearn使用n_iter
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=n_iter,
            learning_rate=learning_rate,
            verbose=0
        )
    except TypeError:
        # 旧版本sklearn使用max_iter
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=n_iter,
            learning_rate=learning_rate,
            verbose=0
        )

    # 执行降维
    tsne_2d = tsne.fit_transform(embeddings)

    return tsne_2d


# ==============================================================================
# Cross-Attention分析工具函数 (v3.0 新增)
# ==============================================================================

def extract_cross_attention_weights(model, data_loader, device, num_samples=100):
    """
    批量提取GAT_SeparateEncoder模型的Cross-Attention权重

    遍历数据加载器，收集每个样本的特征级注意力权重。
    注意力权重表示模型对各静态特征的关注程度。

    Args:
        model: GAT_SeparateEncoder模型实例
        data_loader: PyTorch DataLoader
        device: 计算设备 ('cpu' or 'cuda')
        num_samples: 分析样本数量上限

    Returns:
        dict: {
            'attention_weights': np.ndarray [num_samples, num_heads, num_static_features]
                每个样本的注意力权重（已在节点维度上求平均）
            'dynamic_features': np.ndarray [num_samples, hist_len, dynamic_dim]
                每个样本的动态特征（已在节点维度上求平均）
            'static_features': np.ndarray [num_samples, static_dim]
                每个样本的静态特征（已在节点维度上求平均）
            'sample_indices': list 样本索引
            'num_heads': int 注意力头数
            'num_static_features': int 静态特征数量
        }

    Raises:
        ValueError: 如果模型不是GAT_SeparateEncoder或不支持return_cross_attention
    """
    # 验证模型类型（detect_model_type 在同一文件中已定义）
    model_type = detect_model_type(model)
    if model_type != 'GAT_SeparateEncoder':
        raise ValueError(
            f"extract_cross_attention_weights仅支持GAT_SeparateEncoder模型，"
            f"当前模型类型: {model_type}"
        )

    model.eval()
    all_attention_weights = []
    all_dynamic_features = []
    all_static_features = []
    sample_indices = []
    samples_collected = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if samples_collected >= num_samples:
                break

            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            data = data.to(device)

            # 提取边索引
            edge_index = data.edge_index

            # 获取输入特征
            x = data.x  # [num_nodes, hist_len, in_dim]

            # 前向传播并获取注意力权重
            _, attn_weights = model(
                x, edge_index, return_cross_attention=True
            )
            # attn_weights: [num_nodes, num_heads, num_static_features]

            # 提取静态和动态特征
            static_dim = model.static_dim
            static_feats = x[:, 0, :static_dim].cpu().numpy()
            dynamic_feats = x[:, :, static_dim:].cpu().numpy()

            # 在节点维度上求平均，确保每个样本形状一致
            # attn_weights: [num_nodes, num_heads, num_static_features]
            # → 求平均: [num_heads, num_static_features]
            attn_weights_mean = attn_weights.mean(dim=0).cpu().numpy()
            all_attention_weights.append(attn_weights_mean)

            # dynamic_feats: [num_nodes, hist_len, dynamic_dim]
            # → 求平均: [hist_len, dynamic_dim]
            dynamic_feats_mean = dynamic_feats.mean(axis=0)
            all_dynamic_features.append(dynamic_feats_mean)

            # static_feats: [num_nodes, static_dim]
            # → 求平均: [static_dim]
            static_feats_mean = static_feats.mean(axis=0)
            all_static_features.append(static_feats_mean)

            sample_indices.append(batch_idx)
            samples_collected += 1

    # 使用 np.stack 堆叠固定形状的数组
    return {
        'attention_weights': np.stack(all_attention_weights, axis=0),
        'dynamic_features': np.stack(all_dynamic_features, axis=0),
        'static_features': np.stack(all_static_features, axis=0),
        'sample_indices': sample_indices,
        'num_heads': model.fusion.num_heads,
        'num_static_features': model.num_static_features
    }


def aggregate_attention_by_group(attention_weights, group_mapping):
    """
    按语义分组聚合注意力权重

    将12个静态特征的注意力权重按预定义的语义分组聚合。

    Args:
        attention_weights: np.ndarray [num_samples, num_nodes, num_heads, num_features]
        group_mapping: dict 分组映射
            {
                'Geographic': [0, 1, 2],         # x, y, height
                'Building': [3, 4, 5, 6, 7],     # BH, BHstd, SCD, lambda_p, lambda_b
                'LandCover': [8, 9, 10, 11]      # PLA, POI, POW, POV
            }

    Returns:
        dict: {group_name: np.ndarray [num_samples, num_nodes, num_heads]}
            每个分组的聚合注意力权重（组内平均）
    """
    grouped_attention = {}

    for group_name, feature_indices in group_mapping.items():
        # 提取该组的特征注意力: [..., len(feature_indices)]
        group_attn = attention_weights[..., feature_indices]
        # 在特征维度上求平均: [...] → 得到该组的聚合注意力
        grouped_attention[group_name] = np.mean(group_attn, axis=-1)

    return grouped_attention


def save_node_embedding_analysis(node_embeddings, tsne_2d, station_info,
                                  static_features, feature_names, save_path):
    """
    保存节点嵌入分析结果到CSV文件

    Args:
        node_embeddings: np.ndarray [num_nodes, embedding_dim] 节点嵌入
        tsne_2d: np.ndarray [num_nodes, 2] t-SNE降维结果
        station_info: np.ndarray [num_nodes, 4] 站点信息 [id, lon, lat, height]
        static_features: np.ndarray [num_nodes, num_static_features] 静态特征
        feature_names: list 静态特征名称列表
        save_path: str 保存路径

    Returns:
        str: 保存的文件路径
    """
    import pandas as pd

    num_nodes = node_embeddings.shape[0]
    embedding_dim = node_embeddings.shape[1]

    # 构建数据字典
    data = {
        'station_id': station_info[:, 0].astype(int),
        'lon': station_info[:, 1],
        'lat': station_info[:, 2],
        'height': station_info[:, 3],
    }

    # 添加嵌入维度
    for i in range(embedding_dim):
        data[f'emb_{i}'] = node_embeddings[:, i]

    # 添加t-SNE坐标
    data['tsne_x'] = tsne_2d[:, 0]
    data['tsne_y'] = tsne_2d[:, 1]

    # 添加静态特征
    num_static_to_add = min(len(feature_names), static_features.shape[1])
    for i in range(num_static_to_add):
        data[feature_names[i]] = static_features[:, i]

    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False, encoding='utf-8')

    return str(save_path)


# 静态特征分组定义（用于可视化）
STATIC_FEATURE_GROUPS = {
    'Geographic': {
        'indices': [0, 1, 2],
        'names': ['x', 'y', 'height'],
        'label': 'Geographic Location'
    },
    'Building': {
        'indices': [3, 4, 5, 6, 7],
        'names': ['BH', 'BHstd', 'SCD', 'lambda_p', 'lambda_b'],
        'label': 'Building Morphology'
    },
    'LandCover': {
        'indices': [8, 9, 10, 11],
        'names': ['PLA', 'POI', 'POW', 'POV'],
        'label': 'Land Cover & Population'
    }
}

# 如果有节点嵌入，添加额外的分组
STATIC_FEATURE_GROUPS_WITH_EMBEDDING = {
    'Geographic': {
        'indices': [0, 1, 2],
        'names': ['x', 'y', 'height'],
        'label': 'Geographic Location'
    },
    'Building': {
        'indices': [3, 4, 5, 6, 7],
        'names': ['BH', 'BHstd', 'SCD', 'lambda_p', 'lambda_b'],
        'label': 'Building Morphology'
    },
    'LandCover': {
        'indices': [8, 9, 10, 11],
        'names': ['PLA', 'POI', 'POW', 'POV'],
        'label': 'Land Cover & Population'
    },
    'NodeEmbedding': {
        'indices': [12, 13, 14, 15],
        'names': ['emb_0', 'emb_1', 'emb_2', 'emb_3'],
        'label': 'Learned Node Embedding'
    }
}
