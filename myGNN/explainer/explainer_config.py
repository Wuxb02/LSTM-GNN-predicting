"""
可解释性分析配置类

作者: GNN气温预测项目
日期: 2025
"""


class ExplainerConfig:
    """
    可解释性分析配置

    Args:
        num_samples (int): 分析样本数量,默认100
        epochs (int): GNNExplainer训练轮数,默认200
        season (str): 季节筛选,可选值:
            - None: 不筛选,分析全年
            - 'spring': 春季(3-5月)
            - 'summer': 夏季(6-8月)
            - 'autumn': 秋季(9-11月)
            - 'winter': 冬季(12-2月)
        ig_steps (int): Integrated Gradients积分步数,默认50
        lr (float): GNNExplainer学习率,默认0.01
        top_k_edges (int): 保存的Top-K重要边数量,默认20
        extract_attention (bool): 是否提取GAT注意力权重,默认True
        all_edges_mode (str): 全边可视化模式,默认'both'
            - 'overlay': 只生成叠加模式图
            - 'separate': 只生成分离模式图
            - 'both': 生成两种模式图
            - None: 不生成全边可视化图

    示例:
        >>> # 全年分析
        >>> config = ExplainerConfig(num_samples=100, season=None)
        >>>
        >>> # 仅分析夏季,提取注意力权重
        >>> config = ExplainerConfig(
        >>>     num_samples=100,
        >>>     season='summer',
        >>>     extract_attention=True
        >>> )
    """

    def __init__(
        self,
        num_samples=100,
        epochs=200,
        season=None,
        ig_steps=50,
        lr=0.01,
        top_k_edges=20,
        extract_attention=True,
        all_edges_mode='both'
    ):
        self.num_samples = num_samples
        self.epochs = epochs
        self.season = season
        self.ig_steps = ig_steps
        self.lr = lr
        self.top_k_edges = top_k_edges
        self.extract_attention = extract_attention
        self.all_edges_mode = all_edges_mode

        # 验证季节参数
        valid_seasons = [None, 'spring', 'summer', 'autumn', 'winter']
        if self.season not in valid_seasons:
            raise ValueError(
                f"无效的季节参数: {season}. "
                f"可选值: {valid_seasons}"
            )

        # 验证全边可视化模式
        valid_modes = ['overlay', 'separate', 'both', None]
        if self.all_edges_mode not in valid_modes:
            raise ValueError(
                f"无效的all_edges_mode参数: {all_edges_mode}. "
                f"可选值: {valid_modes}"
            )

    def __repr__(self):
        return (
            f"ExplainerConfig("
            f"num_samples={self.num_samples}, "
            f"epochs={self.epochs}, "
            f"season={self.season}, "
            f"ig_steps={self.ig_steps}, "
            f"extract_attention={self.extract_attention}, "
            f"all_edges_mode={self.all_edges_mode})"
        )
