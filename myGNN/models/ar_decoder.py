"""
自回归解码器模块 (ARDecoder)

适用于 GAT_SeparateEncoder 和 GSAGE_SeparateEncoder。

核心设计：
- 初始隐状态直接使用编码器输出（hid_dim 保持一致，无需投影）
- 每步输入：上一步预测值（标量 → Linear 投影到 input_size）
- 训练时支持 Teacher Forcing（按概率使用真实标签作为输入）
- 推理时纯自回归，ta_label=None

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn


class ARDecoder(nn.Module):
    """
    自回归 GRU/LSTM 解码器

    每步接收一个标量预测值（或真实标签）作为输入，
    通过 GRUCell/LSTMCell 更新隐状态，输出下一步预测值。

    初始隐状态 h_0 = 编码器输出 [N, hid_dim]（不需要投影）。
    第 0 步的输入为全零向量（start token）。
    """

    def __init__(self, hid_dim, pred_len, cell_type='GRU',
                 teacher_forcing_ratio=0.5,
                 tf_decay='none', tf_start=1.0, tf_end=0.0):
        """
        Args:
            hid_dim (int): 隐状态维度，需与编码器输出维度相同
            pred_len (int): 预测步数
            cell_type (str): 'GRU' 或 'LSTM'
            teacher_forcing_ratio (float): 训练时使用真实标签的概率 [0, 1]
            tf_decay (str): 衰减策略 'none', 'linear', 'exponential'
            tf_start (float): 衰减起始值
            tf_end (float): 衰减终止值
        """
        super(ARDecoder, self).__init__()

        self.hid_dim = hid_dim
        self.pred_len = pred_len
        self.cell_type = cell_type
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.tf_decay = tf_decay
        self.tf_start = tf_start
        self.tf_end = tf_end

        # 将标量输入投影到 hid_dim
        self.input_proj = nn.Linear(1, hid_dim)

        # RNN Cell
        if cell_type == 'GRU':
            self.cell = nn.GRUCell(
                input_size=hid_dim,
                hidden_size=hid_dim
            )
        elif cell_type == 'LSTM':
            self.cell = nn.LSTMCell(
                input_size=hid_dim,
                hidden_size=hid_dim
            )
        else:
            raise ValueError(
                f"不支持的 cell_type: {cell_type}，请使用 'GRU' 或 'LSTM'"
            )

        # 输出层：hid_dim → 1（每步输出一个预测值）
        self.output_layer = nn.Linear(hid_dim, 1)

    def update_teacher_forcing_ratio(self, epoch, total_epochs):
        """根据衰减策略更新 Teacher Forcing 比率

        Args:
            epoch (int): 当前 epoch（1-based）
            total_epochs (int): 总训练轮数
        """
        if self.tf_decay == 'none' or total_epochs <= 1:
            return

        progress = (epoch - 1) / (total_epochs - 1)

        if self.tf_decay == 'linear':
            ratio = self.tf_start - (self.tf_start - self.tf_end) * progress
        elif self.tf_decay == 'exponential':
            if self.tf_start <= 0:
                ratio = self.tf_end
            else:
                ratio = self.tf_start * (
                    (self.tf_end / self.tf_start) ** progress
                )
        else:
            return

        self.teacher_forcing_ratio = float(
            max(min(ratio, self.tf_start), self.tf_end)
        )

    def forward(self, encoder_out, ta_label=None):
        """
        自回归解码

        Args:
            encoder_out: [N, hid_dim]，编码器最终输出，作为初始隐状态
            ta_label: [N, pred_len] 或 None
                - 训练时传入真实标签（用于 Teacher Forcing）
                - 推理时传 None（纯自回归）

        Returns:
            predictions: [N, pred_len]，逐步预测结果（已拼接）
        """
        batch_size = encoder_out.shape[0]
        device = encoder_out.device

        # 初始化隐状态（直接使用编码器输出，维度已对齐）
        h = encoder_out
        c = None
        if self.cell_type == 'LSTM':
            c = torch.zeros_like(h)

        # 第 0 步的输入：start token（全零）
        step_input = torch.zeros(batch_size, 1, device=device)

        outputs = []

        for t in range(self.pred_len):
            # 投影输入：[N, 1] → [N, hid_dim]
            proj_input = self.input_proj(step_input)

            # RNN Cell 更新
            if self.cell_type == 'GRU':
                h = self.cell(proj_input, h)
            else:  # LSTM
                h, c = self.cell(proj_input, (h, c))

            # 预测当前步
            pred_t = self.output_layer(h)
            outputs.append(pred_t)

            # Teacher Forcing 逻辑
            use_teacher = (
                ta_label is not None
                and self.training
                and torch.rand(1).item() < self.teacher_forcing_ratio
            )
            if use_teacher:
                step_input = ta_label[:, t].unsqueeze(1)
            else:
                step_input = pred_t.detach()

        # 拼接所有步：[N, pred_len]
        predictions = torch.cat(outputs, dim=1)
        return predictions
