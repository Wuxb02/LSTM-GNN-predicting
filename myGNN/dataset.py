"""
myGNN数据加载模块

功能：
1. 滑动窗口切分
2. 4维时间周期编码（年周期sin/cos + 月周期sin/cos）
3. 特征选择
4. 数据标准化
5. PyG Data格式转换
6. 静态/动态特征分离编码（新增）

作者: GNN气温预测项目
日期: 2025
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


# 年份边界定义（2010-2017年数据）
# 数据索引从0开始，每年的[start, end)
YEAR_BOUNDARIES = {
    2010: (0, 365),       # 365天
    2011: (365, 730),     # 365天
    2012: (730, 1096),    # 366天（闰年）
    2013: (1096, 1461),   # 365天
    2014: (1461, 1826),   # 365天
    2015: (1826, 2191),   # 365天
    2016: (2191, 2557),   # 366天（闰年）
    2017: (2557, 2922),   # 365天
}


def _get_year_boundaries():
    """
    获取年份边界字典

    Returns:
        dict: {year: (start_idx, end_idx)} 每年的索引边界
        例如: {2010: (0, 365), 2011: (365, 730), ...}
    """
    return YEAR_BOUNDARIES.copy()


def _get_year_from_idx(time_idx):
    """
    根据时间索引获取对应的年份

    Args:
        time_idx: 时间索引

    Returns:
        int: 年份
    """
    for year, (start, end) in YEAR_BOUNDARIES.items():
        if start <= time_idx < end:
            return year
    # 边界情况处理
    if time_idx < 0:
        return 2010
    return 2017


def _get_years_in_range(start_idx, end_idx):
    """
    根据索引范围判断跨越哪些年份

    Args:
        start_idx: 窗口起始索引
        end_idx: 窗口结束索引

    Returns:
        list: 年份列表，如 [2015] 或 [2015, 2016]
    """
    years = set()
    for idx in [start_idx, end_idx - 1]:  # 检查起始和结束点
        years.add(_get_year_from_idx(idx))
    return sorted(list(years))


class WeatherGraphDataset(Dataset):
    """
    气象图数据集

    支持：
    - 滑动窗口切分（可配置hist_len, pred_len）
    - 时间周期编码（doy, month → 4维sin/cos）
    - 特征选择（feature_indices）
    - 任意预测目标（target_feature_idx）
    - 静态/动态特征分离编码（use_feature_separation）
    """

    def __init__(self, MetData, graph, start_idx, end_idx, config,
                 static_encoded=None, static_encoded_by_year=None):
        """
        初始化数据集

        Args:
            MetData: 气象数据 [time_steps, num_stations, features]
            graph: 图结构对象
            start_idx: 数据集起始索引
            end_idx: 数据集结束索引
            config: 配置对象
            static_encoded: 预编码的静态特征 [num_nodes, encoded_dim]（可选，向后兼容）
            static_encoded_by_year: 按年份的静态编码字典（可选）
                {year: [num_nodes, encoded_dim], ...}
        """
        self.MetData = MetData
        self.graph = graph
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.config = config

        # 特征分离相关
        self.use_feature_separation = getattr(
            config, 'use_feature_separation', False
        )
        self.static_encoded = static_encoded
        self.static_encoded_by_year = static_encoded_by_year

        # 计算有效样本数量
        # 需要hist_len个历史数据和pred_len个未来数据
        self.valid_start = start_idx + config.hist_len
        self.valid_end = end_idx - config.pred_len
        self.num_samples = max(0, self.valid_end - self.valid_start)

    def __len__(self):
        """返回数据集样本数量"""
        return self.num_samples

    def _encode_temporal_features(self, time_idx):
        """
        将时间特征转换为4维sin/cos编码

        Args:
            time_idx: 当前时间索引

        Returns:
            temporal_features: [hist_len, 4] 数组
                - 列0: doy_sin (年周期正弦)
                - 列1: doy_cos (年周期余弦)
                - 列2: month_sin (月周期正弦)
                - 列3: month_cos (月周期余弦)
        """
        temporal_list = []

        for t in range(self.config.hist_len):
            current_idx = time_idx - self.config.hist_len + t

            # 获取原始时间特征（所有气象站的时间特征相同，取第一个）
            doy = self.MetData[current_idx, 0, 26]      # 年内日序数 (1-366)
            month = self.MetData[current_idx, 0, 27]    # 月份 (1-12)

            # 年周期编码
            days_in_year = 366 if doy > 365 else 365
            year_phase = 2 * np.pi * (doy - 1) / days_in_year
            doy_sin = np.sin(year_phase)
            doy_cos = np.cos(year_phase)

            # 月周期编码 (12个月一个周期)
            # month从1开始，需要减1使其从0开始
            month_phase = 2 * np.pi * (month - 1) / 12
            month_sin = np.sin(month_phase)
            month_cos = np.cos(month_phase)

            # 组合为4维时间特征
            temporal_feat = np.array([
                doy_sin, doy_cos,      # 年周期
                month_sin, month_cos   # 月周期
            ])

            temporal_list.append(temporal_feat)

        return np.array(temporal_list)  # [hist_len, 4]

    def __getitem__(self, idx):
        """
        获取一个样本

        Args:
            idx: 样本索引

        Returns:
            data: PyG Data对象
            time_idx: 时间索引（用于追踪）
        """
        time_idx = self.valid_start + idx

        # 根据是否启用特征分离选择不同的处理方式
        if self.use_feature_separation:
            return self._getitem_separated(time_idx)
        else:
            return self._getitem_original(time_idx)

    def _getitem_original(self, time_idx):
        """
        原模式获取样本（保持向后兼容）

        Args:
            time_idx: 时间索引

        Returns:
            data: PyG Data对象
            time_idx: 时间索引张量
        """
        # 1. 提取历史窗口 [hist_len, num_stations, features]
        hist_window = self.MetData[time_idx - self.config.hist_len:time_idx]

        # 2. 特征选择
        if self.config.feature_indices is not None:
            # 使用指定特征
            features = hist_window[:, :, self.config.feature_indices]
        else:
            # 使用所有特征，但移除doy和month（索引26-27）
            # 保留索引0-25
            features = hist_window[:, :, :26]

        # features shape: [hist_len, num_stations, base_features]

        # 3. 添加时间周期编码
        if self.config.add_temporal_encoding:
            temporal_features = self._encode_temporal_features(time_idx)

            # 为每个气象站复制时间特征
            num_stations = features.shape[1]
            temporal_expanded = np.tile(
                temporal_features[:, np.newaxis, :],
                (1, num_stations, 1)
            )

            # 拼接原始特征和时间特征
            features = np.concatenate([features, temporal_expanded], axis=2)

        # 4. 转换为 [num_stations, hist_len, features]
        x = features.transpose(1, 0, 2).copy()

        # 5. 提取标签
        y = self.MetData[
            time_idx:time_idx + self.config.pred_len,
            :,
            self.config.target_feature_idx
        ]
        y = y.T.copy()

        # 6. 构建PyG Data对象
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)

        if self.config.use_edge_attr and hasattr(self.graph, 'edge_attr'):
            data = Data(
                x=x_tensor,
                y=y_tensor,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr
            )
        else:
            data = Data(
                x=x_tensor,
                y=y_tensor,
                edge_index=self.graph.edge_index
            )

        return data, torch.LongTensor([time_idx])

    def _get_static_embedding_for_window(self, time_idx):
        """
        根据历史窗口的时间范围获取对应的静态特征嵌入

        如果窗口在单一年份内，返回该年的静态编码；
        如果窗口跨越两年，返回两年静态编码的平均值。

        Args:
            time_idx: 当前时间索引（窗口结束位置）

        Returns:
            static_embedding: [num_nodes, encoded_dim] 静态特征嵌入
        """
        # 如果使用按年份的静态编码
        if self.static_encoded_by_year is not None:
            window_start = time_idx - self.config.hist_len
            window_end = time_idx

            # 获取窗口跨越的年份
            years_in_window = _get_years_in_range(window_start, window_end)

            if len(years_in_window) == 1:
                # 单年：直接使用该年的静态编码
                year = years_in_window[0]
                return self.static_encoded_by_year[year]
            else:
                # 跨年：取多年平均
                embeddings = [
                    self.static_encoded_by_year[y] for y in years_in_window
                ]
                return np.mean(embeddings, axis=0)

        # 向后兼容：使用单一静态编码
        elif self.static_encoded is not None:
            return self.static_encoded

        # 都没有预编码，使用原始静态特征
        else:
            hist_window = self.MetData[
                time_idx - self.config.hist_len:time_idx
            ]
            static_indices = self.config.static_feature_indices
            return hist_window[0, :, static_indices]

    def _getitem_separated(self, time_idx):
        """
        特征分离模式获取样本

        数据流：
        1. 提取动态特征窗口 [hist_len, nodes, dynamic_dim]
        2. 添加时间编码 [hist_len, nodes, dynamic_dim + 4]
        3. 获取静态嵌入 [nodes, static_encoded_dim]（支持按年份获取和跨年平均）
        4. 广播静态嵌入 [hist_len, nodes, static_encoded_dim]
        5. 拼接 [hist_len, nodes, total_dim]
        6. 转置 [nodes, hist_len, total_dim]

        Args:
            time_idx: 时间索引

        Returns:
            data: PyG Data对象
            time_idx: 时间索引张量
        """
        # 1. 提取动态特征历史窗口
        hist_window = self.MetData[time_idx - self.config.hist_len:time_idx]
        dynamic_indices = self.config.dynamic_feature_indices
        dynamic_features = hist_window[:, :, dynamic_indices]
        # shape: [hist_len, num_nodes, dynamic_dim]

        # 2. 添加时间周期编码
        if self.config.add_temporal_encoding:
            temporal_features = self._encode_temporal_features(time_idx)
            # shape: [hist_len, 4]

            num_nodes = dynamic_features.shape[1]
            temporal_expanded = np.tile(
                temporal_features[:, np.newaxis, :],
                (1, num_nodes, 1)
            )
            # shape: [hist_len, num_nodes, 4]

            dynamic_features = np.concatenate(
                [dynamic_features, temporal_expanded], axis=2
            )
            # shape: [hist_len, num_nodes, dynamic_dim + 4]

        # 3. 获取静态特征嵌入（支持按年份和跨年平均）
        static_embedding = self._get_static_embedding_for_window(time_idx)
        # shape: [num_nodes, encoded_dim]

        # 4. 广播静态嵌入到每个时间步
        hist_len = dynamic_features.shape[0]
        static_broadcast = np.tile(
            static_embedding[np.newaxis, :, :],  # [1, num_nodes, static_dim]
            (hist_len, 1, 1)                      # [hist_len, num_nodes, static_dim]
        )

        # 5. 拼接静态和动态特征
        # 顺序：[静态编码, 动态特征, 时间编码]
        features = np.concatenate([static_broadcast, dynamic_features], axis=2)
        # shape: [hist_len, num_nodes, static_dim + dynamic_dim + temporal_dim]

        # 6. 转置为模型期望的格式
        x = features.transpose(1, 0, 2).copy()
        # shape: [num_nodes, hist_len, total_dim]

        # 7. 提取标签
        y = self.MetData[
            time_idx:time_idx + self.config.pred_len,
            :,
            self.config.target_feature_idx
        ]
        y = y.T.copy()

        # 8. 构建PyG Data对象
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y)

        if self.config.use_edge_attr and hasattr(self.graph, 'edge_attr'):
            data = Data(
                x=x_tensor,
                y=y_tensor,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr
            )
        else:
            data = Data(
                x=x_tensor,
                y=y_tensor,
                edge_index=self.graph.edge_index
            )

        return data, torch.LongTensor([time_idx])


def create_dataloaders(config, graph):
    """
    创建数据加载器

    Args:
        config: 配置对象
        graph: 图结构对象

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        stats: 统计信息字典
    """
    # 1. 加载数据
    MetData = np.load(config.MetData_fp)
    print(f"✓ 加载数据: {config.MetData_fp}")
    print(f"  原始形状: {MetData.shape}")

    # 2. 判断是否使用特征分离模式
    use_separation = getattr(config, 'use_feature_separation', False)

    if use_separation:
        return _create_dataloaders_separated(config, graph, MetData)
    else:
        return _create_dataloaders_original(config, graph, MetData)


def _create_dataloaders_original(config, graph, MetData):
    """
    原模式创建数据加载器（保持向后兼容）
    """
    # 2. 计算标准化统计量（仅使用训练集）
    train_data = MetData[config.train_start:config.train_end]

    # 2.1 计算所有输入特征的统计量（0-25索引，移除时间特征26-27）
    feature_data = train_data[:, :, :26]

    feature_mean = feature_data.mean(axis=(0, 1))
    feature_std = feature_data.std(axis=(0, 1))

    # 安全检查
    if np.isnan(feature_mean).any() or np.isnan(feature_std).any():
        raise ValueError(
            f"标准化参数包含NaN - mean: {feature_mean}, std: {feature_std}"
        )

    if (feature_std < 1e-6).any():
        problematic_features = np.where(feature_std < 1e-6)[0]
        raise ValueError(
            f"部分特征标准差过小: 特征索引 {problematic_features}\n"
            f"标准差值: {feature_std[problematic_features]}"
        )

    # 2.2 提取目标特征的统计量
    ta_mean = float(feature_mean[config.target_feature_idx])
    ta_std = float(feature_std[config.target_feature_idx])

    # 🆕 计算90分位数（用于动态高温阈值）
    target_feature_data = train_data[:, :, config.target_feature_idx]  # shape: [train_len, 28]
    ta_p90 = float(np.percentile(target_feature_data, 90))  # 90分位数

    print(f"\n标准化参数计算完成:")
    print(f"  特征均值范围: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
    print(f"  特征标准差范围: [{feature_std.min():.4f}, {feature_std.max():.4f}]")
    print(f"  目标特征(索引{config.target_feature_idx}) - "
          f"mean: {ta_mean:.4f}, std: {ta_std:.4f}")
    print(f"  目标特征90分位数: {ta_p90:.4f}°C (可用于动态高温阈值)")

    # 3. 标准化
    MetData[:, :, :26] = (MetData[:, :, :26] - feature_mean) / (feature_std + 1e-8)

    print(f"✓ 已标准化所有26个输入特征")

    stats = {
        'ta_mean': ta_mean,
        'ta_std': ta_std,
        'ta_p90': ta_p90,          # 🆕 添加90分位数
        'feature_mean': feature_mean,
        'feature_std': feature_std
    }

    # 4. 创建数据集
    train_dataset = WeatherGraphDataset(
        MetData, graph, config.train_start, config.train_end, config
    )
    val_dataset = WeatherGraphDataset(
        MetData, graph, config.val_start, config.val_end, config
    )
    test_dataset = WeatherGraphDataset(
        MetData, graph, config.test_start, config.test_end, config
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 5. 创建DataLoader
    train_loader, val_loader, test_loader = _create_loaders(
        train_dataset, val_dataset, test_dataset, config
    )

    return train_loader, val_loader, test_loader, stats


def _create_dataloaders_separated(config, graph, MetData):
    """
    特征分离模式创建数据加载器

    流程：
    1. 分离提取静态和动态特征
    2. 分别计算统计量并标准化
    3. 按年份提取静态特征并使用MLP编码
    4. 创建数据集（支持跨年平均）
    """
    from myGNN.feature_encoder import StaticFeatureEncoder

    print(f"\n✓ 启用特征分离模式（按年份提取静态特征）")
    print(f"  静态特征索引: {config.static_feature_indices} "
          f"({len(config.static_feature_indices)}个)")
    print(f"  动态特征索引: {config.dynamic_feature_indices} "
          f"({len(config.dynamic_feature_indices)}个)")

    static_indices = config.static_feature_indices
    dynamic_indices = config.dynamic_feature_indices

    # 2. 分别计算静态和动态特征的标准化参数（仅使用训练集）
    train_data = MetData[config.train_start:config.train_end]

    # 2.1 静态特征统计量
    static_data = train_data[:, :, static_indices]
    static_mean = static_data.mean(axis=(0, 1))
    static_std = static_data.std(axis=(0, 1))

    # 安全检查静态特征
    if (static_std < 1e-6).any():
        problematic = np.where(static_std < 1e-6)[0]
        print(f"  ⚠ 部分静态特征标准差过小: 索引 {problematic}")
        static_std = np.maximum(static_std, 1e-6)

    # 2.2 动态特征统计量
    dynamic_data = train_data[:, :, dynamic_indices]
    dynamic_mean = dynamic_data.mean(axis=(0, 1))
    dynamic_std = dynamic_data.std(axis=(0, 1))

    # 安全检查动态特征
    if (dynamic_std < 1e-6).any():
        problematic = np.where(dynamic_std < 1e-6)[0]
        print(f"  ⚠ 部分动态特征标准差过小: 索引 {problematic}")
        dynamic_std = np.maximum(dynamic_std, 1e-6)

    print(f"\n静态特征标准化:")
    print(f"  均值范围: [{static_mean.min():.4f}, {static_mean.max():.4f}]")
    print(f"  标准差范围: [{static_std.min():.4f}, {static_std.max():.4f}]")
    print(f"\n动态特征标准化:")
    print(f"  均值范围: [{dynamic_mean.min():.4f}, {dynamic_mean.max():.4f}]")
    print(f"  标准差范围: [{dynamic_std.min():.4f}, {dynamic_std.max():.4f}]")

    # 🆕 在标准化之前计算目标特征的90分位数（用于动态高温阈值）
    target_idx = config.target_feature_idx
    target_feature_data = train_data[:, :, target_idx]  # 使用原始未标准化的数据
    ta_p90 = float(np.percentile(target_feature_data, 90))
    print(f"\n目标特征(索引{target_idx})90分位数: {ta_p90:.4f}°C (可用于动态高温阈值)")

    # 3. 标准化整个数据集
    MetData[:, :, static_indices] = (
        (MetData[:, :, static_indices] - static_mean) / (static_std + 1e-8)
    )
    MetData[:, :, dynamic_indices] = (
        (MetData[:, :, dynamic_indices] - dynamic_mean) / (dynamic_std + 1e-8)
    )

    print(f"✓ 已分别标准化静态和动态特征")

    # 4. 按年份提取并编码静态特征
    year_boundaries = _get_year_boundaries()
    print(f"\n按年份提取静态特征:")

    # 创建静态编码器（恒等映射）
    static_encoder = StaticFeatureEncoder(
        input_dim=len(static_indices),
        output_dim=config.static_encoded_dim
    )

    # 按年份提取和编码静态特征
    static_encoded_by_year = {}
    for year, (start, end) in year_boundaries.items():
        # 提取该年的静态特征平均值
        year_static = MetData[start:end, :, static_indices].mean(axis=0)
        # shape: [num_nodes, static_dim]

        # 编码
        with torch.no_grad():
            year_encoded = static_encoder(
                torch.FloatTensor(year_static)
            ).numpy()
        # shape: [num_nodes, encoded_dim]

        static_encoded_by_year[year] = year_encoded
        print(f"  {year}年: 索引[{start}, {end}), "
              f"编码形状: {year_encoded.shape}")

    print(f"\n静态编码器参数量: "
          f"{sum(p.numel() for p in static_encoder.parameters()):,}")

    # 5. 计算目标特征统计量（用于损失反标准化）
    # 注意：ta_p90 已在标准化之前计算（见上方第561-565行）
    if target_idx in dynamic_indices:
        target_in_dynamic = dynamic_indices.index(target_idx)
        ta_mean = float(dynamic_mean[target_in_dynamic])
        ta_std = float(dynamic_std[target_in_dynamic])
    else:
        # 如果目标在静态特征中（不常见）
        target_in_static = static_indices.index(target_idx)
        ta_mean = float(static_mean[target_in_static])
        ta_std = float(static_std[target_in_static])

    print(f"\n目标特征(索引{target_idx}) - mean: {ta_mean:.4f}, std: {ta_std:.4f}")

    # 统计跨年样本数量
    cross_year_count = 0
    for year, (start, end) in year_boundaries.items():
        # 检查该年初有多少样本会跨越到上一年
        cross_start = start + config.hist_len
        if year > 2010:
            # hist_len个样本的窗口会跨越年份边界
            cross_count = min(config.hist_len, end - start)
            cross_year_count += cross_count

    print(f"\n跨年样本统计:")
    print(f"  历史窗口长度: {config.hist_len} 天")
    print(f"  预计跨年样本数: ~{cross_year_count} (每年初约{config.hist_len}个)")

    stats = {
        'ta_mean': ta_mean,
        'ta_std': ta_std,
        'ta_p90': ta_p90,          # 🆕 添加90分位数
        'static_mean': static_mean,
        'static_std': static_std,
        'dynamic_mean': dynamic_mean,
        'dynamic_std': dynamic_std,
        'static_encoder': static_encoder,
        'static_encoded_by_year': static_encoded_by_year,
        # 向后兼容：保留一个全局静态编码（取所有年份平均）
        'static_encoded': np.mean(
            list(static_encoded_by_year.values()), axis=0
        )
    }

    # 6. 创建数据集（使用按年份的静态编码）
    train_dataset = WeatherGraphDataset(
        MetData, graph, config.train_start, config.train_end, config,
        static_encoded_by_year=static_encoded_by_year
    )
    val_dataset = WeatherGraphDataset(
        MetData, graph, config.val_start, config.val_end, config,
        static_encoded_by_year=static_encoded_by_year
    )
    test_dataset = WeatherGraphDataset(
        MetData, graph, config.test_start, config.test_end, config,
        static_encoded_by_year=static_encoded_by_year
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 维度信息
    print(f"\n特征维度信息:")
    print(f"  静态编码维度: {config.static_encoded_dim}")
    print(f"  动态特征维度: {len(dynamic_indices)}")
    print(f"  时间编码维度: "
          f"{config.temporal_features if config.add_temporal_encoding else 0}")
    print(f"  总输入维度: {config.in_dim}")

    # 7. 创建DataLoader
    train_loader, val_loader, test_loader = _create_loaders(
        train_dataset, val_dataset, test_dataset, config
    )

    return train_loader, val_loader, test_loader, stats


def _create_loaders(train_dataset, val_dataset, test_dataset, config):
    """
    创建DataLoader（共用函数）
    """
    def collate_fn(batch):
        """将batch中的样本组合为PyG Batch对象"""
        data_list = [item[0] for item in batch]
        time_indices = [item[1] for item in batch]
        batched_data = Batch.from_data_list(data_list)
        time_indices = torch.cat(time_indices, dim=0)
        return batched_data, time_indices

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载
    import sys
    sys.path.append('..')

    from config import create_config
    from graph import load_graph_from_station_info

    print("=" * 70)
    print("数据加载模块测试")
    print("=" * 70)

    # 测试1: 特征分离模式（默认）
    print("\n【测试1: 特征分离模式】")
    print("-" * 50)

    config, arch_config = create_config()
    config.use_feature_separation = True

    # 构建图
    graph = load_graph_from_station_info(
        config.station_info_fp,
        top_neighbors=config.top_neighbors,
        use_edge_attr=config.use_edge_attr
    )

    # 创建数据加载器
    train_loader, val_loader, test_loader, stats = create_dataloaders(
        config, graph
    )

    # 测试一个batch
    for batch_data, time_indices in train_loader:
        print(f"\n特征分离模式 - 测试batch:")
        print(f"  batch_data.x shape: {batch_data.x.shape}")
        print(f"  batch_data.y shape: {batch_data.y.shape}")
        print(f"  期望x形状: [batch*nodes, hist_len, {config.in_dim}]")

        # 验证维度
        expected_dim = config.in_dim
        actual_dim = batch_data.x.shape[-1]
        assert actual_dim == expected_dim, \
            f"维度不匹配: 实际{actual_dim} vs 期望{expected_dim}"
        print(f"  ✓ 维度验证通过: {actual_dim}")
        break

    # 测试2: 原模式（向后兼容）
    print("\n" + "=" * 70)
    print("【测试2: 原模式（向后兼容）】")
    print("-" * 50)

    config2, _ = create_config()
    config2.use_feature_separation = False

    train_loader2, val_loader2, test_loader2, stats2 = create_dataloaders(
        config2, graph
    )

    for batch_data2, time_indices2 in train_loader2:
        print(f"\n原模式 - 测试batch:")
        print(f"  batch_data.x shape: {batch_data2.x.shape}")
        print(f"  batch_data.y shape: {batch_data2.y.shape}")
        print(f"  期望x形状: [batch*nodes, hist_len, {config2.in_dim}]")

        expected_dim2 = config2.in_dim
        actual_dim2 = batch_data2.x.shape[-1]
        assert actual_dim2 == expected_dim2, \
            f"维度不匹配: 实际{actual_dim2} vs 期望{expected_dim2}"
        print(f"  ✓ 维度验证通过: {actual_dim2}")
        break

    # 对比两种模式
    print("\n" + "=" * 70)
    print("【模式对比】")
    print("-" * 50)
    print(f"特征分离模式 输入维度: {config.in_dim}")
    print(f"  - 静态编码: {config.static_encoded_dim}")
    print(f"  - 动态特征: {len(config.dynamic_feature_indices)}")
    print(f"  - 时间编码: {config.temporal_features}")
    print(f"\n原模式 输入维度: {config2.in_dim}")
    print(f"  - 全部特征: 26")
    print(f"  - 时间编码: {config2.temporal_features}")

    print("\n" + "=" * 70)
    print("所有测试通过!")
    print("=" * 70)

