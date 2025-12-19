"""
GraphSAGE + LSTM 模型

修复内容：
1. 修复whichAF函数调用（删除多余的hid_dim参数）
2. 删除未使用的导入（wind_graph）

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


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


class GSAGE_LSTM(nn.Module):
    """
    GraphSAGE + LSTM 模型

    结构:
    1. MLP输入层 (hist_len, in_dim) → (hist_len, hid_dim)
    2. LSTM时序建模 (hist_len, hid_dim) → (hid_dim,)
    3. SAGE图卷积层 x N (hid_dim) → (hid_dim)
    4. MLP输出层 (hid_dim) → (pred_len,)
    """

    def __init__(self, config, arch_arg):
        super(GSAGE_LSTM, self).__init__()
        self.config = config
        self.in_dim = config.in_dim
        self.nMLP_layer = arch_arg.MLP_layer
        self.nSAGE_layer = arch_arg.SAGE_layer
        self.hid_dim = arch_arg.hid_dim
        self.aggr = arch_arg.aggr
        AF = whichAF(arch_arg.AF)  # 修复：删除hid_dim参数
        self.out_dim = config.pred_len

        MLP_layers_in = [nn.Linear(self.in_dim, self.hid_dim)]
        SAGE_layers = []
        MLP_layers_out = []

        # 计算每个SAGE块的层数
        self.element = 2  # SAGE + AF
        if arch_arg.norm_type != 'None':
            self.element += 1
        if arch_arg.dropout:
            self.element += 1

        # MLP输入层
        for n in range(self.nMLP_layer):
            MLP_layers_in.append(nn.Linear(self.hid_dim, self.hid_dim))
            MLP_layers_in.append(AF)

        # SAGE层
        for n in range(self.nSAGE_layer):
            SAGE_layers.append(
                SAGEConv(
                    self.hid_dim, self.hid_dim,
                    aggr=arch_arg.aggr,
                )
            )
            SAGE_layers.append(AF)
            # 添加规范化层
            norm_layer = get_norm_layer(arch_arg.norm_type, self.hid_dim)
            if norm_layer is not None:
                SAGE_layers.append(norm_layer)
            if arch_arg.dropout:
                SAGE_layers.append(nn.Dropout(arch_arg.inter_drop))

        # MLP输出层
        for n in range(self.nMLP_layer):
            MLP_layers_out.append(nn.Linear(self.hid_dim, self.hid_dim))
            MLP_layers_out.append(AF)
        MLP_layers_out.append(nn.Linear(self.hid_dim, self.out_dim))

        self.MLP_layers_in = nn.Sequential(*MLP_layers_in)

        # LSTM层（使用配置参数）
        self.lstm_bidirectional = arch_arg.lstm_bidirectional
        self.lstm = nn.LSTM(
            input_size=self.hid_dim,
            hidden_size=self.hid_dim,
            num_layers=arch_arg.lstm_num_layers,
            dropout=arch_arg.lstm_dropout if arch_arg.lstm_num_layers > 1 else 0,
            bidirectional=arch_arg.lstm_bidirectional,
            batch_first=False
        )
        # 双向LSTM需要投影层将2*hid_dim映射回hid_dim
        if arch_arg.lstm_bidirectional:
            self.lstm_fc = nn.Linear(self.hid_dim * 2, self.hid_dim)

        self.SAGE_layers = nn.ModuleList(SAGE_layers)
        self.MLP_layers_out = nn.Sequential(*MLP_layers_out)

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播

        Args:
            x: [num_nodes, hist_len, in_dim]
            edge_index: [2, num_edges]
            edge_attr: 边属性（SAGEConv不使用，为了接口统一）

        Returns:
            x: [num_nodes, pred_len]
        """
        # 1. 转换维度用于LSTM
        x = x.permute(1, 0, 2)  # [hist_len, num_nodes, in_dim]

        # 2. MLP输入层
        x = self.MLP_layers_in(x)  # [hist_len, num_nodes, hid_dim]

        # 3. LSTM时序建模
        out, _ = self.lstm(x)
        x = out.contiguous()
        x = x[-1]  # 取最后一个时间步 [num_nodes, hid_dim] 或 [num_nodes, 2*hid_dim]

        # 双向LSTM投影
        if self.lstm_bidirectional:
            x = self.lstm_fc(x)  # [num_nodes, hid_dim]

        # 4. SAGE图卷积层
        for i in range(self.nSAGE_layer):
            base_idx = i * self.element
            # SAGE卷积
            x = self.SAGE_layers[base_idx](x, edge_index)
            # 后续层（AF, BN?, Dropout?）
            for j in range(1, self.element):
                x = self.SAGE_layers[base_idx + j](x)

        # 5. MLP输出层
        x = self.MLP_layers_out(x)

        return x


def whichAF(AF):
    """
    激活函数选择

    Args:
        AF: 激活函数名称

    Returns:
        激活函数层
    """
    if AF == 'PReLU':
        return nn.PReLU()  # 修复：正确返回PReLU
    elif AF == "LeakyReLU":
        return nn.LeakyReLU()
    elif AF == "PReLUMulti":
        return nn.PReLU()
    elif AF == "ReLU":
        return nn.ReLU()
    else:
        return lambda x: x
