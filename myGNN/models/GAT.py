"""
GAT (Graph Attention Network) + LSTM 模型

修复内容：
1. 修复whichAF函数调用（删除多余的hid_dim参数）
2. 修复forward返回值（只返回x，不返回attention）
3. 修复循环索引问题（第60-62行）

作者: GNN气温预测项目
日期: 2025
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


class GAT_LSTM(torch.nn.Module):
    """
    GAT + LSTM 模型

    结构:
    1. MLP输入层 (hist_len, in_dim) → (hist_len, hid_dim)
    2. LSTM时序建模 (hist_len, hid_dim) → (hid_dim,)
    3. GAT图卷积层 x N (hid_dim) → (hid_dim)
    4. MLP输出层 (hid_dim) → (pred_len,)
    """

    def __init__(self, config, arch_arg):
        super(GAT_LSTM, self).__init__()
        self.in_dim = config.in_dim
        self.nMLP_layer = arch_arg.MLP_layer
        self.nGAT_layer = arch_arg.GAT_layer
        self.hid_dim = arch_arg.hid_dim
        self.heads = arch_arg.heads
        AF = whichAF(arch_arg.AF)  # 修复：删除hid_dim参数
        self.out_dim = config.pred_len

        MLP_layers_in = [nn.Linear(self.in_dim, self.hid_dim)]
        GAT_layers = []
        MLP_layers_out = []

        # 计算每个GAT块的层数
        self.element = 3  # GAT + Linear + AF
        if arch_arg.norm_type != 'None':
            self.element += 1
        if arch_arg.dropout:
            self.element += 1

        # MLP输入层
        for n in range(self.nMLP_layer):
            MLP_layers_in.append(nn.Linear(self.hid_dim, self.hid_dim))
            MLP_layers_in.append(AF)

        # GAT层
        for n in range(self.nGAT_layer):
            in_dim = self.hid_dim
            out_dim = self.hid_dim
            GAT_layers.append(
                GATv2Conv(
                    in_dim, out_dim,
                    heads=self.heads, concat=True,
                    dropout=arch_arg.intra_drop,
                    add_self_loops=False, share_weights=False
                )
            )
            GAT_layers.append(nn.Linear(self.heads * out_dim, out_dim))
            GAT_layers.append(AF)
            # 添加规范化层
            norm_layer = get_norm_layer(arch_arg.norm_type, out_dim)
            if norm_layer is not None:
                GAT_layers.append(norm_layer)
            if arch_arg.dropout:
                GAT_layers.append(nn.Dropout(arch_arg.inter_drop))

        self.MLP_layers_in = nn.Sequential(*MLP_layers_in)

        # LSTM层（使用配置参数）
        self.lstm_bidirectional = arch_arg.lstm_bidirectional
        self.lstm = nn.LSTM(
            input_size=self.hid_dim,
            hidden_size=self.hid_dim,
            num_layers=arch_arg.lstm_num_layers*2,
            dropout=arch_arg.lstm_dropout if arch_arg.lstm_num_layers > 1 else 0,
            bidirectional=arch_arg.lstm_bidirectional,
            batch_first=False
        )
        # 双向LSTM需要投影层将2*hid_dim映射回hid_dim
        if arch_arg.lstm_bidirectional:
            self.lstm_fc = nn.Linear(self.hid_dim * 2, self.hid_dim)

        self.GAT_layers = nn.ModuleList(GAT_layers)

        # 🔑 循环解码器（可选，用于多步预测）
        self.use_recurrent_decoder = getattr(
            arch_arg, 'use_recurrent_decoder', False
        )
        self.decoder_type = getattr(arch_arg, 'decoder_type', 'LSTM')
        self.decoder_use_context = getattr(
            arch_arg, 'decoder_use_context', False
        )

        if self.use_recurrent_decoder:
            decoder_num_layers = getattr(arch_arg, 'decoder_num_layers', 1)
            decoder_dropout = getattr(arch_arg, 'decoder_dropout', 0.1)
            decoder_dropout = decoder_dropout if decoder_num_layers > 1 else 0
            decoder_mlp_layers = getattr(arch_arg, 'decoder_mlp_layers', 0)

            # 解码器输入维度
            decoder_input_size = (
                1 + self.hid_dim if self.decoder_use_context else 1
            )

            # 解码器前置MLP层（可选）
            if decoder_mlp_layers > 0:
                decoder_mlp = []
                mlp_input_size = decoder_input_size
                for _ in range(decoder_mlp_layers):
                    decoder_mlp.append(nn.Linear(mlp_input_size, self.hid_dim))
                    decoder_mlp.append(nn.ReLU())
                    mlp_input_size = self.hid_dim
                self.decoder_mlp = nn.Sequential(*decoder_mlp)
                decoder_input_size = self.hid_dim
            else:
                self.decoder_mlp = None

            # 解码器RNN
            if self.decoder_type == 'LSTM':
                self.decoder = nn.LSTM(
                    input_size=decoder_input_size,
                    hidden_size=self.hid_dim,
                    num_layers=decoder_num_layers,
                    dropout=decoder_dropout,
                    batch_first=False
                )
            elif self.decoder_type == 'GRU':
                self.decoder = nn.GRU(
                    input_size=decoder_input_size,
                    hidden_size=self.hid_dim,
                    num_layers=decoder_num_layers,
                    dropout=decoder_dropout,
                    batch_first=False
                )
            else:
                raise ValueError(
                    f"未知的解码器类型: {self.decoder_type}，支持: 'LSTM', 'GRU'"
                )

            self.decoder_output_proj = nn.Linear(self.hid_dim, 1)
            self.decoder_init_proj = nn.Linear(self.hid_dim, 1)
        else:
            # MLP输出层（原有方式）
            for n in range(self.nMLP_layer):
                MLP_layers_out.append(nn.Linear(self.hid_dim, self.hid_dim))
                MLP_layers_out.append(AF)
            MLP_layers_out.append(nn.Linear(self.hid_dim, self.out_dim))
            self.MLP_layers_out = nn.Sequential(*MLP_layers_out)

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播

        Args:
            x: [num_nodes, hist_len, in_dim]
            edge_index: [2, num_edges]
            edge_attr: 边属性（GATv2Conv不使用，为了接口统一）

        Returns:
            x: [num_nodes, pred_len]
        """
        # 1. 转换维度用于LSTM: [num_nodes, hist_len, in_dim] → [hist_len, num_nodes, in_dim]
        x = x.permute(1, 0, 2)

        # 2. MLP输入层
        x = self.MLP_layers_in(x)  # [hist_len, num_nodes, hid_dim]

        # 3. LSTM时序建模
        out, _ = self.lstm(x)  # [hist_len, num_nodes, hid_dim] 或 [hist_len, num_nodes, 2*hid_dim]
        x = out.contiguous()
        x = x[-1]  # 取最后一个时间步 [num_nodes, hid_dim] 或 [num_nodes, 2*hid_dim]

        # 双向LSTM投影
        if self.lstm_bidirectional:
            x = self.lstm_fc(x)  # [num_nodes, hid_dim]

        # 4. GAT图卷积层
        for i in range(self.nGAT_layer):
            # 修复：正确计算GAT层的索引
            base_idx = i * self.element

            # GAT卷积（不需要返回attention）
            x = self.GAT_layers[base_idx](x, edge_index)  # [num_nodes, heads*hid_dim]

            # 后续层（Linear, AF, BN?, Dropout?）
            for j in range(1, self.element):
                x = self.GAT_layers[base_idx + j](x)

        # 5. 输出生成
        if self.use_recurrent_decoder:
            # 🔑 循环解码器：逐步生成预测序列
            outputs = []
            encoder_context = x  # [N, hid_dim]

            num_layers = self.decoder.num_layers
            if self.decoder_type == 'LSTM':
                h_0 = x.unsqueeze(0).repeat(num_layers, 1, 1)
                c_0 = torch.zeros_like(h_0)
                hidden = (h_0, c_0)
            else:
                hidden = x.unsqueeze(0).repeat(num_layers, 1, 1)

            prev_pred = self.decoder_init_proj(x)  # [N, 1]

            for t in range(self.out_dim):
                if self.decoder_use_context:
                    decoder_input = torch.cat(
                        [prev_pred, encoder_context], dim=1
                    )
                else:
                    decoder_input = prev_pred

                if self.decoder_mlp is not None:
                    decoder_input = self.decoder_mlp(decoder_input)

                decoder_input = decoder_input.unsqueeze(0)
                decoder_output, hidden = self.decoder(decoder_input, hidden)

                pred_t = self.decoder_output_proj(decoder_output.squeeze(0))
                outputs.append(pred_t)
                prev_pred = pred_t

            x = torch.cat(outputs, dim=1)  # [N, pred_len]
        else:
            # MLP输出层（原有方式）
            x = self.MLP_layers_out(x)  # [num_nodes, pred_len]

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
