"""
GAT_Pure (纯图注意力网络) 模型

与GAT_LSTM的核心区别:
1. 移除LSTM模块 - 不进行时序建模
2. 展平历史窗口 - 将[hist_len, in_dim]展平为[hist_len*in_dim]
3. 专注空间建模 - 纯图卷积+注意力机制

适用场景:
- 评估纯空间关系建模的效果
- 对比时空融合 vs 纯空间的性能差异
- 作为消融实验的基线模型

作者: GNN气温预测项目
日期: 2025-12-20
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


def get_norm_layer(norm_type, dim):
    """
    规范化层选择

    Args:
        norm_type: 规范化类型 ('BatchNorm', 'LayerNorm', 'None')
        dim: 特征维度

    Returns:
        规范化层或 None
    """
    if norm_type == 'BatchNorm':
        return nn.BatchNorm1d(dim)
    elif norm_type == 'LayerNorm':
        return nn.LayerNorm(dim)
    elif norm_type == 'None' or norm_type is None:
        return None
    else:
        raise ValueError(f"未知的规范化类型: {norm_type}")


def whichAF(AF):
    """
    激活函数选择

    Args:
        AF: 激活函数类型字符串

    Returns:
        激活函数实例
    """
    if AF == 'ReLU':
        return nn.ReLU()
    elif AF == 'PReLU':
        return nn.PReLU()
    elif AF == 'LeakyReLU':
        return nn.LeakyReLU()
    elif AF == 'GELU':
        return nn.GELU()
    else:
        return nn.ReLU()


class GAT_Pure(torch.nn.Module):
    """
    纯GAT模型（无LSTM）

    结构:
    1. 特征展平: [num_nodes, hist_len, in_dim] → [num_nodes, hist_len*in_dim]
    2. MLP特征变换: [num_nodes, hist_len*in_dim] → [num_nodes, hid_dim]
    3. GAT图卷积层 x N: [num_nodes, hid_dim] → [num_nodes, hid_dim]
    4. MLP输出层: [num_nodes, hid_dim] → [num_nodes, pred_len]
    """

    def __init__(self, config, arch_arg):
        super(GAT_Pure, self).__init__()

        # 基础参数
        self.in_dim = config.in_dim
        self.hist_len = config.hist_len
        self.nMLP_layer = arch_arg.MLP_layer
        self.nGAT_layer = arch_arg.GAT_layer
        self.hid_dim = arch_arg.hid_dim
        self.heads = arch_arg.heads
        self.out_dim = config.pred_len

        # 激活函数
        AF = whichAF(arch_arg.AF)

        # 计算展平后的输入维度
        flatten_dim = self.hist_len * self.in_dim

        # 1. MLP特征变换层
        MLP_layers_in = []
        MLP_layers_in.append(nn.Linear(flatten_dim, self.hid_dim))
        MLP_layers_in.append(AF)

        for n in range(self.nMLP_layer - 1):
            MLP_layers_in.append(nn.Linear(self.hid_dim, self.hid_dim))
            MLP_layers_in.append(AF)

        self.MLP_layers_in = nn.Sequential(*MLP_layers_in)

        # 2. GAT图卷积层
        GAT_layers = []

        # 计算每个GAT块的层数
        self.element = 3  # GAT + Linear + AF
        if arch_arg.norm_type != 'None':
            self.element += 1
        if arch_arg.dropout:
            self.element += 1

        for n in range(self.nGAT_layer):
            in_dim = self.hid_dim
            out_dim = self.hid_dim

            # GAT卷积层
            GAT_layers.append(
                GATv2Conv(
                    in_dim, out_dim,
                    heads=self.heads, concat=True,
                    dropout=arch_arg.intra_drop,
                    add_self_loops=False, share_weights=False
                )
            )

            # 线性投影层（多头拼接 → 原始维度）
            GAT_layers.append(nn.Linear(self.heads * out_dim, out_dim))

            # 激活函数
            GAT_layers.append(AF)

            # 规范化层（可选）
            norm_layer = get_norm_layer(arch_arg.norm_type, out_dim)
            if norm_layer is not None:
                GAT_layers.append(norm_layer)

            # Dropout（可选）
            if arch_arg.dropout:
                GAT_layers.append(nn.Dropout(arch_arg.inter_drop))

        self.GAT_layers = nn.ModuleList(GAT_layers)

        # 3. MLP输出层
        MLP_layers_out = []
        for n in range(self.nMLP_layer):
            MLP_layers_out.append(nn.Linear(self.hid_dim, self.hid_dim))
            MLP_layers_out.append(AF)
        MLP_layers_out.append(nn.Linear(self.hid_dim, self.out_dim))

        self.MLP_layers_out = nn.Sequential(*MLP_layers_out)

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播

        Args:
            x: [num_nodes, hist_len, in_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_attr: 边属性（GAT不使用，为了接口统一）

        Returns:
            x: [num_nodes, pred_len] 预测结果
        """
        num_nodes = x.shape[0]

        # 1. 展平时间维度
        # [num_nodes, hist_len, in_dim] → [num_nodes, hist_len*in_dim]
        x = x.reshape(num_nodes, -1)

        # 2. MLP特征变换
        # [num_nodes, hist_len*in_dim] → [num_nodes, hid_dim]
        x = self.MLP_layers_in(x)

        # 3. GAT图卷积
        for i in range(self.nGAT_layer):
            base_idx = i * self.element

            # GAT卷积
            x = self.GAT_layers[base_idx](x, edge_index)  # [num_nodes, heads*hid_dim]

            # 后续层（Linear, AF, BN?, Dropout?）
            for j in range(1, self.element):
                x = self.GAT_layers[base_idx + j](x)

        # 4. MLP输出层
        # [num_nodes, hid_dim] → [num_nodes, pred_len]
        x = self.MLP_layers_out(x)

        return x


def test_gat_pure():
    """
    单元测试：验证GAT_Pure的前向传播
    """
    print("=" * 60)
    print("GAT_Pure模型单元测试")
    print("=" * 60)

    # 创建模拟配置
    class MockConfig:
        def __init__(self):
            self.in_dim = 30
            self.hist_len = 7
            self.pred_len = 3
            self.node_num = 28

    class MockArchConfig:
        def __init__(self):
            self.hid_dim = 16
            self.MLP_layer = 2
            self.GAT_layer = 3
            self.heads = 4
            self.AF = 'ReLU'
            self.norm_type = 'LayerNorm'
            self.dropout = True
            self.intra_drop = 0.1
            self.inter_drop = 0.1

    config = MockConfig()
    arch_config = MockArchConfig()

    # 创建模型
    print(f"\n创建模型...")
    print(f"  输入维度: {config.in_dim}")
    print(f"  历史窗口: {config.hist_len}")
    print(f"  预测步长: {config.pred_len}")
    print(f"  节点数量: {config.node_num}")
    print(f"  隐藏维度: {arch_config.hid_dim}")
    print(f"  GAT层数: {arch_config.GAT_layer}")
    print(f"  注意力头数: {arch_config.heads}")

    model = GAT_Pure(config, arch_config)

    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 创建模拟输入
    x = torch.randn(config.node_num, config.hist_len, config.in_dim)
    edge_index = torch.randint(0, config.node_num, (2, 100))

    print(f"\n输入形状:")
    print(f"  x: {list(x.shape)}")
    print(f"  edge_index: {list(edge_index.shape)}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)

    print(f"\n输出形状:")
    print(f"  output: {list(output.shape)}")

    # 验证输出形状
    expected_shape = (config.node_num, config.pred_len)
    assert output.shape == expected_shape, \
        f"输出形状错误: 期望 {expected_shape}, 实际 {output.shape}"

    print(f"\n✓ 前向传播测试通过")

    # 验证输出范围
    print(f"\n输出统计:")
    print(f"  均值: {output.mean().item():.4f}")
    print(f"  标准差: {output.std().item():.4f}")
    print(f"  最小值: {output.min().item():.4f}")
    print(f"  最大值: {output.max().item():.4f}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_gat_pure()
