"""
myGNN LSTM基线模型

功能：
1. 纯LSTM模型（无图结构）
2. 用作对比基线

模型结构：
1. 输入MLP层：(in_dim) → (hid_dim)
2. LSTM层：处理时序特征
3. 输出层：(hid_dim) → (pred_len)

修复内容：
1. 修复初始化参数（添加arch_arg参数）
2. 从arch_arg获取hid_dim
3. 添加详细注释

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn


class LSTM_direct(nn.Module):
    """
    直接预测LSTM模型

    不使用图结构，纯时序建模
    """

    def __init__(self, config, arch_arg):
        """
        初始化LSTM模型

        Args:
            config: 配置对象
                - hist_len: 历史窗口长度
                - in_dim: 输入特征维度
                - pred_len: 预测长度
            arch_arg: 架构参数对象
                - hid_dim: 隐藏层维度
        """
        super(LSTM_direct, self).__init__()

        self.hist_len = config.hist_len
        self.in_dim = config.in_dim
        self.out_dim = config.pred_len
        self.hid_dim = arch_arg.hid_dim
        self.num_layers = arch_arg.lstm_num_layers
        self.lstm_bidirectional = arch_arg.lstm_bidirectional

        # 输入层：特征维度映射
        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)

        # LSTM层：时序建模（使用配置参数）
        self.lstm = nn.LSTM(
            input_size=self.hid_dim,
            hidden_size=self.hid_dim,
            num_layers=self.num_layers*3,
            dropout=arch_arg.lstm_dropout if self.num_layers > 1 else 0,
            bidirectional=self.lstm_bidirectional,
            batch_first=False  # 期望输入 [seq_len, batch, features]
        )

        # 双向LSTM需要投影层将2*hid_dim映射回hid_dim
        if self.lstm_bidirectional:
            self.lstm_fc = nn.Linear(self.hid_dim * 2, self.hid_dim)

        # 输出层：预测
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, feature, edge_index=None, edge_attr=None):
        """
        前向传播

        Args:
            feature: [num_nodes, hist_len, in_dim]
            edge_index: 边索引（LSTM不使用，为了接口统一）
            edge_attr: 边属性（LSTM不使用，为了接口统一）

        Returns:
            out: [num_nodes, pred_len]
        """
        # 1. 转换维度用于LSTM
        # [num_nodes, hist_len, in_dim] → [hist_len, num_nodes, in_dim]
        feature = feature.permute(1, 0, 2)

        # 2. 输入层
        x_in = self.fc_in(feature)  # [hist_len, num_nodes, hid_dim]

        # 3. LSTM时序建模
        out, _ = self.lstm(x_in)  # [hist_len, num_nodes, hid_dim] 或 [hist_len, num_nodes, 2*hid_dim]

        # 4. 取最后一个时间步
        out = out[-1]  # [num_nodes, hid_dim] 或 [num_nodes, 2*hid_dim]

        # 双向LSTM投影
        if self.lstm_bidirectional:
            out = self.lstm_fc(out)  # [num_nodes, hid_dim]

        # 5. 输出层
        out = self.fc_out(out)  # [num_nodes, pred_len]

        return out


# ==================== 测试代码 ====================


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    from config import create_config

    # 创建配置
    config, arch_config = create_config()
    config.exp_model = 'LSTM'

    print("=" * 80)
    print("LSTM模型测试")
    print("=" * 80)

    # 创建模型
    model = LSTM_direct(config, arch_config)

    print(f"✓ 模型创建成功")
    print(f"  输入维度: {config.in_dim}")
    print(f"  隐藏维度: {arch_config.hid_dim}")
    print(f"  输出维度: {config.pred_len}")
    print(f"  历史长度: {config.hist_len}")

    # 测试前向传播
    batch_size = 4
    num_nodes = config.node_num
    x = torch.randn(batch_size * num_nodes, config.hist_len, config.in_dim)

    print(f"\n测试前向传播:")
    print(f"  输入形状: {x.shape}")

    with torch.no_grad():
        out = model(x)

    print(f"  输出形状: {out.shape}")
    print(f"  预期形状: [{batch_size * num_nodes}, {config.pred_len}]")

    assert out.shape == (batch_size * num_nodes, config.pred_len), "输出形状不匹配！"

    print(f"\n✓ 所有测试通过！")

    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数统计:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print("=" * 80)
