"""
myGNN配置管理模块 (优化版)

统一管理所有训练参数，包括：
- 数据配置（路径、划分）
- 时间窗口配置
- 特征配置（选择、编码）
- 模型配置
- 训练配置
- 损失函数配置

优化说明：
1. 合并了 Config 和 ConfigWithEnhancements 类
2. 移除了冗余的配置项 (use_weighted_trend_loss, use_enhanced_training)
3. 统一使用 loss_config.loss_type 控制损失函数
5. 修正了所有注释与代码不一致的地方
6. 修正了命名错误（top_neighbors → top_neighbors）

作者: GNN气温预测项目
日期: 2025
版本: 2.0 (优化版)
"""

import torch
from pathlib import Path


class LossConfig:
    """
    损失函数配置

    支持的损失函数类型:
    - 'MSE': 标准均方误差（默认）
    - 'WeightedTrend': 加权趋势损失（论文方法，推荐用于夏季气温预测）

    高温阈值计算方式:
    1. **固定阈值模式** (use_dynamic_threshold=False, 默认):
       - 使用预设的alert_temp值（如35.0°C）
       - 适用于：已知高温标准的场景

    2. **动态阈值模式** (use_dynamic_threshold=True):
       - 自动计算训练集温度的90分位数作为阈值
       - 适用于：不同地区/季节的自适应预测
       - alert_temp会在训练开始时被动态更新

    示例:
        # 使用固定阈值
        loss_config = LossConfig()
        loss_config.use_dynamic_threshold = False
        loss_config.alert_temp = 35.0

        # 使用动态阈值（推荐用于探索性分析）
        loss_config = LossConfig()
        loss_config.use_dynamic_threshold = True
    """

    def __init__(self):
        # 损失函数类型选择（这是唯一需要修改的配置！）
        self.loss_type = "MSE"  # WeightedTrend

        # 🆕 高温阈值计算模式
        self.use_dynamic_threshold = True  # True=使用90分位数, False=使用固定值

        # 🔥 加权趋势MSE损失参数（改进版 - 温度加权 + 趋势约束）
        # 四个核心机制: 精确阈值定义 + 不对称惩罚 + 极端值加权 + 趋势约束
        self.alert_temp = 35.0  # 高温警戒阈值T_alert (°C) - 固定值或动态更新
        self.c_under = 4  # 漏报权重系数(低估高温的惩罚),应较大
        self.c_over = 2  # 误报权重系数(高估的惩罚),可较小
        self.c_default_high = 1.0  # 正确预报高温的权重
        self.delta = 0.1  # 小偏置,缓冲max(0,⋅)计算
        self.trend_weight = 0  # 趋势权重

        # 站点-日内动态阈值配置
        self.use_station_day_threshold = True  # True=启用365x28阈值表, False=使用旧模式
        self.threshold_percentile = 90  # 计算阈值的分位数
        self.threshold_window_radius = 7  # 前后窗口天数（共 2*7+1=15 天）


def get_feature_indices_for_graph(config):
    """
    获取用于图构建的特征索引列表（与数据加载保持一致）

    使用场景:
    - spatial_similarity 图构建需要与模型输入特征保持一致
    - 确保边权重计算基于模型实际使用的特征

    Args:
        config: Config对象

    Returns:
        list: 特征索引列表，例如 [0,1,2,...,26] 或 [0,1,2,3,4,10,11,21,22,23]

    逻辑:
        1. 如果启用特征分离 (use_feature_separation=True):
           合并静态和动态特征索引，去重并排序
        2. 如果指定了feature_indices:
           使用指定的特征索引
        3. 否则:
           使用默认的0-26（移除doy和month）

    注意:
        - 返回的索引用于从原始数据 [time, stations, 29] 中提取特征
        - 不包括时间编码（doy_sin/cos, month_sin/cos），因为时间编码在数据加载时动态生成
        - 索引27-28（doy, month）应被排除，因为它们会被转换为sin/cos
    """
    if config.use_feature_separation:
        # 分离模式：合并静态和动态特征索引
        static_indices = config.static_feature_indices
        dynamic_indices = config.dynamic_feature_indices
        combined = sorted(list(set(static_indices + dynamic_indices)))

        print(f"  [特征选择] 分离模式:")
        print(f"    静态特征索引: {static_indices} ({len(static_indices)}个)")
        print(f"    动态特征索引: {dynamic_indices} ({len(dynamic_indices)}个)")
        print(f"    合并后: {combined} (共{len(combined)}个)")

        return combined

    elif config.feature_indices is not None:
        # 使用指定的特征索引
        indices = list(config.feature_indices)
        print(f"  [特征选择] 使用指定特征: {indices} ({len(indices)}个)")
        return indices

    else:
        # 默认：使用0-26（移除doy和month）
        indices = list(range(27))
        print(f"  [特征选择] 使用默认特征: 0-26 (27个)")
        return indices


class Config:
    """
    统一配置类

    所有可配置参数的集中管理
    """

    def __init__(self):
        # ==================== 数据配置 ====================
        project_root = Path(__file__).parent.parent
        self.MetData_fp = str(project_root / "data" / "real_weather_data_2010_2019.npy")
        self.station_info_fp = str(project_root / "data" / "station_info.npy")

        # 数据集划分（按年份）
        # 2010-2017年(训练): 索引0-2921 (共2922天)
        # 2018年(验证): 索引2922-3286 (共365天)
        # 2019年(测试): 索引3287-3651 (共365天)
        self.train_start = 0
        self.train_end = 2922
        self.val_start = 2922
        self.val_end = 3287
        self.test_start = 3287
        self.test_end = 3652

        self.dataset_num = "real_data_2010_2019"

        # ==================== 时间窗口配置 ====================
        self.hist_len = 14  # 历史窗口长度（天）
        self.pred_len = 5  # 预测长度（天）

        # ==================== 特征配置 ====================
        # 原始特征索引（0-28共29个）:
        # 0-1: x, y (经纬度)
        # 2: height (海拔高度)
        # 3-5: tmin, tmax, tave (温度)
        # 6-9: pre, prs, rh, win (气象要素)
        # 10-11: BH, BHstd (建筑高度特征)
        # 12-13: SCD, PLA (地表覆盖)
        # 14-15: λp, λb (天空开阔度参数)
        # 16-18: POI, POW, POV (兴趣点/人口密度)
        # 19: NDVI (植被指数)
        # 20-21: surface_pressure, surface_solar_radiation (ERA5)
        # 22-23: u_component_of_wind_10m, v_component_of_wind_10m (风速分量)
        # 24: total_precipitation_sum (ERA5累计降水)
        # 25-26: VegHeight_mean, VegHeight_std (植被高度特征)
        # 27-28: doy, month (时间，将被转换为sin/cos)

        self.base_feature_dim = 29  # 原始特征维度（0-28）
        # 预测目标：可以是数字索引（如 5 = tave, 8 = rh）或字符串 "wb"（湿球温度）
        self.target_feature_idx =4  # 预测目标：索引5 = tave, "wb" = 湿球温度

        # 特征选择（None表示使用所有基础特征，即0-25，移除doy和month）
        # 可设置为列表选择部分特征
        self.feature_indices = None

        # 时间编码配置
        self.add_temporal_encoding = True  # 是否添加sin/cos时间编码
        self.temporal_features = 4  # 时间编码维度（年周期2 + 月周期2）

        # ==================== 特征分离配置 ====================
        # 是否启用静态/动态特征分离编码
        # 启用后：静态特征只编码一次，动态特征保留时序处理
        self.use_feature_separation = True

        # 静态特征索引（逐年数据，不随时间变化）
        # 0-2: x, y, height (地理位置)
        # 10-18: BH, BHstd, SCD, PLA, λp, λb, POI, POW, POV (城市形态)
        # 25-26: VegHeight_mean, VegHeight_std (植被高度)
        self.static_feature_indices = [0, 1, 2, 10, 11, 12, 16, 17, 18, 25]
        # self.static_feature_indices = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25, 26]

        # 动态特征索引（逐日数据，随时间变化）
        # 3-9: tmin, tmax, tave, pre, prs, rh, win (气象要素)
        # 19-24: NDVI, surface_pressure, surface_solar_radiation, u_wind, v_wind,
        #         total_precipitation_sum
        # 注意：doy(27)和month(28)将单独转换为sin/cos编码
        self.dynamic_feature_indices = [4, 8, 21, 22, 23, 24]

        # 配置验证
        if self.use_feature_separation:
            # 检查是否有重复索引
            combined = self.static_feature_indices + self.dynamic_feature_indices
            if len(combined) != len(set(combined)):
                raise ValueError(
                    f"静态和动态特征索引有重复！\n"
                    f"静态: {self.static_feature_indices}\n"
                    f"动态: {self.dynamic_feature_indices}"
                )

            # 检查是否包含时间特征（27-28应被排除）
            if 27 in combined or 28 in combined:
                raise ValueError(
                    f"特征索引不应包含时间特征27(doy)和28(month)！\n"
                    f"当前静态: {self.static_feature_indices}\n"
                    f"当前动态: {self.dynamic_feature_indices}\n"
                    f"时间特征将自动转换为sin/cos编码"
                )

            # 如果同时设置了feature_indices，发出警告
            if self.feature_indices is not None:
                import warnings

                warnings.warn(
                    f"检测到同时设置了 use_feature_separation=True 和 feature_indices！\n"
                    f"分离模式将忽略 feature_indices，使用 static + dynamic 索引。\n"
                    f"当前feature_indices: {self.feature_indices}",
                    UserWarning,
                )

        # 静态特征编码器配置
        # 静态特征不进行编码,直接使用原始特征 (恒等映射)
        self.static_encoded_dim = len(
            self.static_feature_indices
        )  # 静态特征编码后的维度

        # 标准化参数（训练时自动计算）
        self.ta_mean = 0.0
        self.ta_std = 1.0

        # ==================== 节点配置 ====================
        self.node_num = 28
        self.city_num = 28

        # ==================== 模型配置 ====================
        # 支持的模型:
        # 基础模型: 'GAT_LSTM', 'GSAGE_LSTM', 'LSTM', 'GAT_Pure' (纯GAT，无LSTM)
        # 分离式编码: 'GAT_SeparateEncoder', 'GSAGE_SeparateEncoder' (静态/动态分离)
        self.exp_model = "GAT_SeparateEncoder"

        # ==================== 图结构配置 ====================
        # 图类型选择：
        # - 'inv_dis': K近邻图 + 逆距离权重（默认，适合距离相关的空间预测）
        # - 'spatial_similarity': 基于空间相似性的图（GeoGAT方法，适合特征相似性建模）
        # - 'knn': K近邻图（无权重，简单快速）
        self.graph_type = "inv_dis"  # 默认使用逆距离权重图

        # K近邻图参数（用于 'inv_dis' 和 'knn' 类型）
        self.top_neighbors = 6
        self.use_edge_attr = False  # 是否使用边属性（逆距离权重）

        # 空间相似性图参数（用于 'spatial_similarity' 类型）
        self.spatial_sim_top_k = 5  # 选择最相似的K个邻居（论文推荐10），一共构建多少边
        self.spatial_sim_alpha = 1.0  # 邻域相似性权重系数（论文默认1.0）
        self.spatial_sim_use_neighborhood = True  # 是否使用邻域相似性
        self.spatial_sim_initial_neighbors = (
            3  # 用于计算邻域相似性的初始空间邻居数，判断地理背景
        )

        # ==================== 可视化配置 ====================
        # 训练后自动生成可视化图表（需要visualize_results.py）
        self.auto_visualize = True  # 训练完成后自动生成可视化
        self.viz_pred_steps = "all"  # 可视化的预测步长：'all'或列表[0,1,2]
        self.viz_plot_all_stations = True  # 是否绘制全部28个站点时间序列
        self.viz_dpi = 300  # 图表分辨率（150=快速预览，300=高质量）
        self.viz_use_basemap = True  # 是否使用地理底图（需要contextily库）

        # ==================== 训练配置 ====================
        self.batch_size = 32  # 批次大小（从128改为32以平衡内存和收敛速度）
        self.epochs = 500
        self.lr = 0.0002
        self.weight_decay = 0.0002  # 从1e-4增大到1e-3以增强正则化
        self.early_stop = 50  # 早停耐心值

        # 优化器配置
        self.optimizer = "Adam"  # 'Adam', 'AdamW', 'SGD', 'RMSprop'
        self.momentum = 0.9  # SGD动量参数
        self.betas = (0.9, 0.999)  # Adam/AdamW的beta参数

        # 学习率调度器配置
        # 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'None'
        self.scheduler = "CosineAnnealingLR"
        # StepLR参数
        self.step_size = 10  # 每隔多少epoch衰减一次
        self.gamma = 0.9  # 学习率衰减系数
        # CosineAnnealingLR参数
        self.T_max = 100  # 余弦退火周期
        self.eta_min = self.lr * 0.1  # 最小学习率
        # ReduceLROnPlateau参数
        self.patience = 20  # 性能不提升的耐心值
        self.factor = 0.8  # 学习率衰减因子
        # MultiStepLR参数
        self.milestones = [50, 100, 150]  # 学习率衰减的epoch列表

        # ==================== 设备配置 ====================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==================== 路径配置 ====================
        self.save_path = "./checkpoints"
        self.log_path = "logs"
        self.seed = 42

        # ==================== 损失函数配置 ====================
        self.loss_config = LossConfig()

    @property
    def in_dim(self):
        """
        自动计算输入特征维度

        Returns:
            输入维度 = 基础特征数 + 时间特征数
        """
        if self.use_feature_separation:
            # 特征分离模式：静态编码维度 + 动态特征数 + 时间编码
            static_dim = self.static_encoded_dim
            dynamic_dim = len(self.dynamic_feature_indices)
            temporal_dim = self.temporal_features if self.add_temporal_encoding else 0
            return static_dim + dynamic_dim + temporal_dim
        else:
            # 原模式：所有特征混合
            if self.feature_indices is not None:
                # 使用指定特征
                base_dim = len(self.feature_indices)
            else:
                # 使用所有特征（移除doy和month后剩余27个：0-26）
                base_dim = 27

            if self.add_temporal_encoding:
                return base_dim + self.temporal_features
            return base_dim

    @property
    def use_enhanced_training(self):
        """
        是否使用增强训练（根据损失函数类型自动判断）

        Returns:
            bool: loss_type != 'MSE' 时返回True
        """
        return self.loss_config.loss_type != "MSE"


class ArchConfig:
    """
    模型架构配置类
    """

    def __init__(self):
        # ==================== 通用架构参数 ====================
        self.hid_dim = 32  # 隐藏层维度（从32增加到64以提升模型容量）
        self.MLP_layer = 1
        self.AF = "LeakyReLU"  # 激活函数：'ReLU', 'LeakyReLU', 'PReLU','GELU'

        # 规范化层类型: 'BatchNorm', 'LayerNorm', 'None'
        # BatchNorm: 适合大batch (>16)，训练/推理有差异
        # LayerNorm: 适合小batch，更稳定
        self.norm_type = "LayerNorm"

        self.dropout = True

        # ==================== GAT特定参数 ====================
        self.GAT_layer = 1  # GAT层数（从2增加到3以增强图学习能力）
        self.heads = 8  # 注意力头数
        self.intra_drop = 0.2  # GAT层内Dropout
        self.inter_drop = 0.2  # GNN层间Dropout

        # ==================== SAGE特定参数 ====================
        self.SAGE_layer = 1  # SAGE层数（从2增加到3，保持一致）
        self.aggr = "mean"  # 聚合方式：'mean', 'max', 'add'
        # inter_drop已在GAT中定义，这里共用

        # ==================== LSTM特定参数 ====================
        self.lstm_num_layers = 1  # LSTM层数（默认1）
        self.lstm_dropout = 0.2  # LSTM层间Dropout（仅num_layers > 1时生效）
        self.lstm_bidirectional = False  # 是否使用双向LSTM

        # ==================== 分离式编码器参数 (v2.0优化版) ====================
        # 用于 GAT_SeparateEncoder 模型

        # 🔥 改进1: 交叉注意力融合参数
        # 废弃原fusion_type参数，现在统一使用CrossAttentionFusion
        self.fusion_num_heads = 2  # 交叉注意力头数（必须能整除hid_dim）
        self.fusion_use_pre_ln = False  # 是否使用Pre-LN（推荐True）

        # 🔥 改进3: GAT残差连接参数
        self.use_skip_connection = True  # 是否在GAT前后添加残差连接


def create_config(loss_type=None, **kwargs):
    """
    创建配置（统一工厂函数）

    Args:
        loss_type (str, optional): 损失函数类型
            - None: 使用 LossConfig.__init__() 中的默认值（推荐）
            - 'MSE': 标准均方误差
            - 'WeightedTrend': 加权趋势损失（推荐用于夏季气温预测）
        **kwargs: 其他配置参数

    Returns:
        config: Config对象
        arch_config: ArchConfig对象

    示例:
        # 使用 LossConfig 中的默认损失函数（在 config.py 中配置）
        config, arch = create_config()

        # 临时覆盖损失函数类型
        config, arch = create_config(loss_type='MSE')

        # 自定义参数
        config, arch = create_config(
            batch_size=64,
            lr=0.001
        )
    """
    config = Config()
    arch_config = ArchConfig()

    # 只有显式传递 loss_type 时才覆盖 LossConfig.__init__() 中的默认值
    if loss_type is not None:
        config.loss_config.loss_type = loss_type

    # 更新其他配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.loss_config, key):
            setattr(config.loss_config, key, value)
        else:
            print(f"警告: 未知配置参数 '{key}'，已忽略")

    return config, arch_config


def print_config(config, arch_config):
    """
    打印配置信息

    Args:
        config: Config对象
        arch_config: ArchConfig对象
    """
    print("=" * 80)
    print("myGNN配置信息")
    print("=" * 80)

    print("\n【数据配置】")
    print(f"  数据路径: {config.MetData_fp}")
    print(f"  数据集: {config.dataset_num}")
    print(f"  气象站数量: {config.node_num}")

    print("\n【数据集划分】")
    print(
        f"  训练集: 索引 {config.train_start}-{config.train_end - 1} ({config.train_end - config.train_start} 天, 2010-2017年)"
    )
    print(
        f"  验证集: 索引 {config.val_start}-{config.val_end - 1} ({config.val_end - config.val_start} 天, 2018年)"
    )
    print(
        f"  测试集: 索引 {config.test_start}-{config.test_end - 1} ({config.test_end - config.test_start} 天, 2019年)"
    )

    print("\n【时间窗口】")
    print(f"  历史窗口长度: {config.hist_len} 天")
    print(f"  预测长度: {config.pred_len} 天")

    print("\n【特征配置】")
    print(f"  原始特征维度: {config.base_feature_dim}")
    print(f"  预测目标: 特征索引 {config.target_feature_idx}")
    if config.feature_indices:
        print(f"  选择特征: {config.feature_indices}")
    else:
        print(f"  选择特征: 所有基础特征（0-26）")
    print(f"  时间编码: {'启用' if config.add_temporal_encoding else '禁用'}")
    if config.add_temporal_encoding:
        print(f"    - 年周期: doy_sin, doy_cos")
        print(f"    - 月周期: month_sin, month_cos")
        print(f"    - 时间特征维度: {config.temporal_features}")

    # 特征分离配置
    if config.use_feature_separation:
        print(f"\n【特征分离配置】")
        print(f"  特征分离: 启用")
        print(
            f"  静态特征索引: {config.static_feature_indices} ({len(config.static_feature_indices)}个)"
        )
        print(
            f"  动态特征索引: {config.dynamic_feature_indices} ({len(config.dynamic_feature_indices)}个)"
        )
        print(f"  静态编码维度: {config.static_encoded_dim}")
        print(f"  静态编码器: 恒等映射 (无编码)")
    else:
        print(f"  特征分离: 禁用（使用原混合模式）")

    print(f"  最终输入维度: {config.in_dim}")

    print("\n【模型配置】")
    print(f"  模型类型: {config.exp_model}")
    print(f"  隐藏层维度: {arch_config.hid_dim}")
    print(f"  激活函数: {arch_config.AF}")
    print(f"  BatchNorm: {arch_config.norm_type}")

    if "GAT" in config.exp_model:
        print(f"  GAT层数: {arch_config.GAT_layer}")
        print(f"  注意力头数: {arch_config.heads}")
        print(
            f"  Dropout: intra={arch_config.intra_drop}, inter={arch_config.inter_drop}"
        )
    elif "SAGE" in config.exp_model:
        print(f"  SAGE层数: {arch_config.SAGE_layer}")
        print(f"  聚合方式: {arch_config.aggr}")
        print(f"  Dropout: inter={arch_config.inter_drop}")

    # 分离式编码器配置 (v2.0优化版)
    if config.exp_model in ["GAT_SeparateEncoder", "GSAGE_SeparateEncoder"]:
        print(f"\n【分离式编码器配置 (v2.0优化版)】")
        print(f"  交叉注意力头数: {arch_config.fusion_num_heads}")
        print(f"  Pre-LN模式: {arch_config.fusion_use_pre_ln}")
        print(f"  残差连接: {'启用' if arch_config.use_skip_connection else '禁用'}")
        if "GSAGE" in config.exp_model:
            print(f"  SAGE聚合方式: {arch_config.aggr}")

    print("\n【图结构】")
    print(f"  图类型: {config.graph_type}")

    if config.graph_type in ["inv_dis", "knn"]:
        print(f"  K近邻数量: {config.top_neighbors}")
        print(f"  使用边属性: {config.use_edge_attr}")
    elif config.graph_type == "spatial_similarity":
        print(f"  选择邻居数: {config.spatial_sim_top_k}")
        print(f"  邻域相似性权重α: {config.spatial_sim_alpha}")
        print(f"  使用邻域相似性: {config.spatial_sim_use_neighborhood}")
        print(f"  初始空间邻居数: {config.spatial_sim_initial_neighbors}")

        # 新增：显示图构建将使用的特征（与数据加载一致性检查）
        graph_features = get_feature_indices_for_graph(config)
        print(f"  图构建特征索引: {graph_features} (共{len(graph_features)}个)")

    elif config.graph_type == "correlation_climate":
        print(f"  相关性邻居数量K: {config.correlation_top_k}")
        print(f"  邻域权重系数α: {config.correlation_climate_alpha}")
        print(f"  动态拓扑: 基于训练集tmax气温相关性")
        print(f"  静态气质: 26特征×4统计量(均值/标准差/最大/最小)")

    elif config.graph_type == "full":
        print(f"  全连接图: 所有节点互相连接")

    print("\n【训练配置】")
    print(f"  批次大小: {config.batch_size}")
    print(f"  最大训练轮数: {config.epochs}")
    print(f"  学习率: {config.lr}")
    print(f"  权重衰减: {config.weight_decay}")
    print(f"  早停耐心: {config.early_stop}")

    print("\n【优化器配置】")
    print(f"  优化器: {config.optimizer}")
    if config.optimizer == "SGD":
        print(f"    - 动量: {config.momentum}")
    elif config.optimizer in ["Adam", "AdamW"]:
        print(f"    - Betas: {config.betas}")

    print("\n【学习率调度器】")
    print(f"  调度器: {config.scheduler}")
    if config.scheduler == "StepLR":
        print(f"    - Step Size: {config.step_size}")
        print(f"    - Gamma: {config.gamma}")
    elif config.scheduler == "CosineAnnealingLR":
        print(f"    - T_max: {config.T_max}")
        print(f"    - Eta_min: {config.eta_min}")
    elif config.scheduler == "ReduceLROnPlateau":
        print(f"    - Patience: {config.patience}")
        print(f"    - Factor: {config.factor}")
    elif config.scheduler == "MultiStepLR":
        print(f"    - Milestones: {config.milestones}")
        print(f"    - Gamma: {config.gamma}")

    print("\n【损失函数配置】")
    print(f"  损失函数类型: {config.loss_config.loss_type}")
    if config.loss_config.loss_type == "WeightedTrend":
        if getattr(config.loss_config, "use_station_day_threshold", False):
            print(f"    - 阈值模式: 站点-日内动态阈值")
            print(f"    - 分位数: {config.loss_config.threshold_percentile}")
            print(f"    - 窗口半径: ±{config.loss_config.threshold_window_radius}天")
            print(f"    - 阈值表形状: (365, 28)")
        elif config.loss_config.use_dynamic_threshold:
            print(f"    - 阈值模式: 全局动态阈值（训练集90分位数）")
            print(f"    - alert_temp: {config.loss_config.alert_temp}°C")
        else:
            print(f"    - 阈值模式: 固定阈值")
            print(f"    - alert_temp: {config.loss_config.alert_temp}°C")

        print(f"    - 漏报权重c_under: {config.loss_config.c_under}")
        print(f"    - 误报权重c_over: {config.loss_config.c_over}")
        print(f"    - 正确预报高温权重: {config.loss_config.c_default_high}")
        print(f"    - 趋势权重α: {config.loss_config.trend_weight}")

    print(f"\n  设备: {config.device}")

    print("=" * 80)


if __name__ == "__main__":
    # 测试配置
    print("=" * 80)
    print("测试1: 标准MSE配置")
    print("=" * 80)
    config, arch_config = create_config()
    print_config(config, arch_config)

    print("\n\n")
    print("=" * 80)
    print("测试2: 加权趋势损失配置")
    print("=" * 80)
    config2, arch2 = create_config(loss_type="WeightedTrend")
    print_config(config2, arch2)

    print("\n\n")
    print("=" * 80)
    print("测试3: 自定义参数配置")
    print("=" * 80)
    config3, arch3 = create_config(
        loss_type="WeightedTrend", batch_size=64, lr=0.001, temp_threshold=29.0
    )
    print(f"批次大小: {config3.batch_size}")
    print(f"学习率: {config3.lr}")
    print(f"高温阈值: {config3.loss_config.temp_threshold}°C")
    print(f"使用增强训练: {config3.use_enhanced_training}")
