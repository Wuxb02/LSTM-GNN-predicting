"""
静态特征编码器模块

功能：
1. 将高维静态特征编码为低维嵌入
2. 支持多种编码方式（MLP、线性、恒等）
3. 可在训练中联合优化

设计思路：
- 静态特征（如经纬度、建筑高度）不随时间变化
- 通过MLP编码为低维表示，减少冗余信息
- 编码后的静态嵌入广播到每个时间步，与动态特征拼接

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn
import numpy as np


class StaticFeatureEncoder(nn.Module):
    """
    静态特征编码器

    将节点级静态特征（如地理位置、城市形态参数）编码为低维嵌入。

    Args:
        input_dim: 静态特征输入维度（默认12）
        output_dim: 编码输出维度（默认8）
        encoder_type: 编码器类型 ('mlp', 'linear', 'none')
        num_layers: MLP层数（仅当encoder_type='mlp'时有效）
        dropout: Dropout率

    输入输出:
        输入: [num_nodes, input_dim] 静态特征
        输出: [num_nodes, output_dim] 编码后的静态嵌入
    """

    def __init__(
        self,
        input_dim: int = 12,
        output_dim: int = 8,
        encoder_type: str = 'mlp',
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(StaticFeatureEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder_type = encoder_type

        if encoder_type == 'mlp':
            layers = []
            current_dim = input_dim

            for i in range(num_layers):
                # 渐进式降维
                next_dim = max(output_dim, (current_dim + output_dim) // 2)
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                current_dim = next_dim

            # 最终投影到目标维度
            layers.append(nn.Linear(current_dim, output_dim))
            self.encoder = nn.Sequential(*layers)

        elif encoder_type == 'linear':
            self.encoder = nn.Linear(input_dim, output_dim)

        elif encoder_type == 'none':
            # 恒等映射（需要 input_dim == output_dim）
            if input_dim != output_dim:
                raise ValueError(
                    f"'none' encoder requires input_dim == output_dim, "
                    f"got {input_dim} vs {output_dim}"
                )
            self.encoder = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [num_nodes, input_dim] 静态特征

        Returns:
            encoded: [num_nodes, output_dim] 编码后的静态嵌入
        """
        return self.encoder(x)


def create_static_encoder(config) -> StaticFeatureEncoder:
    """
    根据配置创建静态特征编码器

    Args:
        config: 配置对象，需包含以下属性:
            - static_feature_indices: 静态特征索引列表
            - static_encoded_dim: 编码输出维度
            - static_encoder_type: 编码器类型
            - static_encoder_layers: MLP层数
            - static_encoder_dropout: Dropout率

    Returns:
        StaticFeatureEncoder实例
    """
    input_dim = len(config.static_feature_indices)
    output_dim = config.static_encoded_dim
    encoder_type = getattr(config, 'static_encoder_type', 'mlp')
    num_layers = getattr(config, 'static_encoder_layers', 1)
    dropout = getattr(config, 'static_encoder_dropout', 0.1)

    return StaticFeatureEncoder(
        input_dim=input_dim,
        output_dim=output_dim,
        encoder_type=encoder_type,
        num_layers=num_layers,
        dropout=dropout
    )


def encode_static_features(
    static_features: np.ndarray,
    encoder: StaticFeatureEncoder,
    device: torch.device = None
) -> np.ndarray:
    """
    使用编码器编码静态特征

    Args:
        static_features: [num_nodes, static_dim] numpy数组
        encoder: StaticFeatureEncoder实例
        device: 计算设备

    Returns:
        encoded: [num_nodes, encoded_dim] numpy数组
    """
    if device is None:
        device = next(encoder.parameters()).device

    # 转换为tensor
    x = torch.FloatTensor(static_features).to(device)

    # 编码
    with torch.no_grad():
        encoder.eval()
        encoded = encoder(x)

    return encoded.cpu().numpy()


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("静态特征编码器测试")
    print("=" * 60)

    # 模拟静态特征：28个节点，12个特征
    num_nodes = 28
    input_dim = 12
    output_dim = 8

    static_features = torch.randn(num_nodes, input_dim)
    print(f"\n输入形状: {static_features.shape}")

    # 测试三种编码器类型
    encoder_types = ['mlp', 'linear']

    for enc_type in encoder_types:
        print(f"\n{'='*40}")
        print(f"测试编码器类型: {enc_type}")
        print(f"{'='*40}")

        encoder = StaticFeatureEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            encoder_type=enc_type,
            num_layers=1,
            dropout=0.1
        )

        # 前向传播
        encoded = encoder(static_features)
        print(f"输出形状: {encoded.shape}")
        print(f"期望形状: [{num_nodes}, {output_dim}]")

        # 统计参数量
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"参数量: {total_params:,}")

        # 验证形状
        assert encoded.shape == (num_nodes, output_dim), \
            f"形状不匹配: {encoded.shape} != ({num_nodes}, {output_dim})"
        print("✓ 形状验证通过")

    # 测试 none 类型（需要相同维度）
    print(f"\n{'='*40}")
    print("测试编码器类型: none (恒等映射)")
    print(f"{'='*40}")

    encoder_none = StaticFeatureEncoder(
        input_dim=8,
        output_dim=8,
        encoder_type='none'
    )
    x_same = torch.randn(num_nodes, 8)
    out_none = encoder_none(x_same)
    print(f"输入形状: {x_same.shape}")
    print(f"输出形状: {out_none.shape}")
    assert torch.equal(x_same, out_none), "恒等映射输出应与输入相同"
    print("✓ 恒等映射验证通过")

    # 测试广播到时间维度
    print(f"\n{'='*40}")
    print("测试静态嵌入广播")
    print(f"{'='*40}")

    encoder = StaticFeatureEncoder(input_dim=12, output_dim=8)
    static_embedding = encoder(static_features)  # [28, 8]

    hist_len = 7
    # 广播到时间维度
    static_broadcast = static_embedding.unsqueeze(1).expand(-1, hist_len, -1)
    print(f"编码后形状: {static_embedding.shape}")
    print(f"广播后形状: {static_broadcast.shape}")
    print(f"期望形状: [{num_nodes}, {hist_len}, {output_dim}]")
    assert static_broadcast.shape == (num_nodes, hist_len, output_dim)
    print("✓ 广播验证通过")

    # 模拟与动态特征拼接
    dynamic_dim = 12
    temporal_dim = 4
    dynamic_features = torch.randn(num_nodes, hist_len, dynamic_dim + temporal_dim)

    combined = torch.cat([static_broadcast, dynamic_features], dim=-1)
    print(f"\n动态特征形状: {dynamic_features.shape}")
    print(f"拼接后形状: {combined.shape}")
    print(f"期望形状: [{num_nodes}, {hist_len}, {output_dim + dynamic_dim + temporal_dim}]")
    assert combined.shape == (num_nodes, hist_len, output_dim + dynamic_dim + temporal_dim)
    print("✓ 拼接验证通过")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
