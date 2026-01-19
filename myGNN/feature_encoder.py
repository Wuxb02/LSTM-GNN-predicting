"""
静态特征编码器模块

功能：
1. 使用恒等映射保留静态特征原始信息
2. 要求输入输出维度相等

设计思路：
- 静态特征（如经纬度、建筑高度）不随时间变化
- 使用恒等映射（nn.Identity）直接传递特征，无需编码
- 编码后的静态嵌入广播到每个时间步，与动态特征拼接

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn
import numpy as np


class StaticFeatureEncoder(nn.Module):
    """
    静态特征编码器（恒等映射）

    将节点级静态特征（如地理位置、城市形态参数）直接传递，不做编码。

    Args:
        input_dim: 静态特征输入维度（默认12）
        output_dim: 编码输出维度（必须等于 input_dim）

    输入输出:
        输入: [num_nodes, input_dim] 静态特征
        输出: [num_nodes, output_dim] 静态特征（与输入相同）
    """

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10
    ):
        super(StaticFeatureEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 恒等映射，要求维度相等
        if input_dim != output_dim:
            raise ValueError(
                f"StaticFeatureEncoder (恒等映射) 要求 input_dim == output_dim, "
                f"当前: input_dim={input_dim}, output_dim={output_dim}"
            )

        self.encoder = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [num_nodes, input_dim] 静态特征

        Returns:
            encoded: [num_nodes, output_dim] 静态特征（与输入相同）
        """
        return self.encoder(x)


def create_static_encoder(config) -> StaticFeatureEncoder:
    """
    根据配置创建静态特征编码器（恒等映射）

    Args:
        config: 配置对象，需包含以下属性:
            - static_feature_indices: 静态特征索引列表
            - static_encoded_dim: 编码输出维度

    Returns:
        StaticFeatureEncoder实例（恒等映射）
    """
    input_dim = len(config.static_feature_indices)
    output_dim = config.static_encoded_dim

    return StaticFeatureEncoder(
        input_dim=input_dim,
        output_dim=output_dim
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
    print("静态特征编码器测试（恒等映射）")
    print("=" * 60)

    # 模拟静态特征：28个节点，10个特征
    num_nodes = 28
    input_dim = 10
    output_dim = 10  # 恒等映射要求相同维度

    static_features = torch.randn(num_nodes, input_dim)
    print(f"\n输入形状: {static_features.shape}")

    # 测试恒等映射编码器
    print(f"\n{'='*40}")
    print("测试恒等映射编码器")
    print(f"{'='*40}")

    encoder = StaticFeatureEncoder(
        input_dim=input_dim,
        output_dim=output_dim
    )

    # 前向传播
    encoded = encoder(static_features)
    print(f"输出形状: {encoded.shape}")
    print(f"期望形状: [{num_nodes}, {output_dim}]")

    # 统计参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"参数量: {total_params:,}")

    # 验证形状和恒等性
    assert encoded.shape == (num_nodes, output_dim), \
        f"形状不匹配: {encoded.shape} != ({num_nodes}, {output_dim})"
    assert torch.equal(static_features, encoded), \
        "恒等映射输出应与输入完全相同"
    print("✓ 形状验证通过")
    print("✓ 恒等映射验证通过")

    # 测试维度不匹配的错误情况
    print(f"\n{'='*40}")
    print("测试维度不匹配错误")
    print(f"{'='*40}")

    try:
        encoder_mismatch = StaticFeatureEncoder(
            input_dim=10,
            output_dim=8  # 不同维度
        )
        print("✗ 应该抛出ValueError")
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")

    # 测试广播到时间维度
    print(f"\n{'='*40}")
    print("测试静态嵌入广播")
    print(f"{'='*40}")

    static_embedding = encoder(static_features)  # [28, 10]

    hist_len = 7
    # 广播到时间维度
    static_broadcast = static_embedding.unsqueeze(1).expand(-1, hist_len, -1)
    print(f"编码后形状: {static_embedding.shape}")
    print(f"广播后形状: {static_broadcast.shape}")
    print(f"期望形状: [{num_nodes}, {hist_len}, {output_dim}]")
    assert static_broadcast.shape == (num_nodes, hist_len, output_dim)
    print("✓ 广播验证通过")

    # 模拟与动态特征拼接
    dynamic_dim = 10
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
