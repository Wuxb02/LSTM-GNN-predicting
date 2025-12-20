"""
RevIN (Reversible Instance Normalization) 模块

用于时间序列预测的实例级标准化技术。

参考文献:
- RevIN: A Simple Baseline for Time Series Forecasting (Kim et al., 2022)

核心思想:
- 对每个样本独立计算统计量（均值和标准差）
- 在模型输入前进行标准化
- 在模型输出后进行反标准化
- 可选的可学习仿射变换参数

优势:
- 处理非平稳时间序列
- 缓解分布偏移问题
- 保持原始数据尺度
- 提高模型泛化能力

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization

    对时间序列数据进行实例级标准化，每个样本独立计算均值和标准差。

    Args:
        num_features (int): 特征维度数
        eps (float): 数值稳定性常数，防止除零。默认 1e-5
        affine (bool): 是否使用可学习的仿射变换参数（gamma 和 beta）。默认 True
        subtract_last (bool): 是否使用最后一个时间步的值作为基准（替代均值）。默认 False

    Shape:
        - Input (normalize): (N, T, D)
            N: batch size
            T: 时间步长度（hist_len）
            D: 特征维度（num_features）
        - Output (normalize): (N, T, D)
        - Input (denormalize): (N, T', D) 或 (N, T', 1)
            T': 可能与 T 不同（如预测长度 pred_len）
        - Output (denormalize): (N, T', D) 或 (N, T', 1)

    Examples:
        >>> # 创建 RevIN 层
        >>> revin = RevIN(num_features=10, affine=True)
        >>>
        >>> # 标准化输入
        >>> x = torch.randn(32, 7, 10)  # [batch, hist_len, features]
        >>> x_norm = revin.normalize(x)
        >>>
        >>> # 模型处理...
        >>> output = model(x_norm)  # [batch, pred_len]
        >>>
        >>> # 反标准化输出
        >>> output_expanded = output.unsqueeze(-1)  # [batch, pred_len, 1]
        >>> output_denorm = revin.denormalize(output_expanded)
        >>> output_final = output_denorm.squeeze(-1)  # [batch, pred_len]
    """

    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        # 可学习的仿射变换参数（如果启用）
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

        # 用于存储统计量的缓存（在 normalize 中计算，在 denormalize 中使用）
        self.mean = None
        self.stdev = None

    def normalize(self, x):
        """
        前向标准化

        将输入时间序列标准化为均值0、标准差1的分布。
        统计量沿时间维度（dim=1）计算，每个样本独立。

        Args:
            x (torch.Tensor): 输入序列，形状 [N, T, D]

        Returns:
            torch.Tensor: 标准化后的序列，形状 [N, T, D]
        """
        # 计算统计量（沿时间维度 dim=1）
        if self.subtract_last:
            # 使用最后一个时间步作为基准（适合趋势明显的数据）
            self.mean = x[:, -1:, :].detach()  # [N, 1, D]
        else:
            # 使用均值作为基准（标准方法）
            self.mean = x.mean(dim=1, keepdim=True).detach()  # [N, 1, D]

        # 计算标准差（使用无偏估计：unbiased=False，与论文一致）
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [N, 1, D]

        # 标准化: (x - mean) / stdev
        x_norm = (x - self.mean) / self.stdev

        # 可选的仿射变换: gamma * x + beta
        if self.affine:
            # gamma 和 beta 形状: [D]，广播到 [N, T, D]
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def denormalize(self, x):
        """
        反标准化到原始尺度

        将标准化后的预测值恢复到原始数据的尺度。

        Args:
            x (torch.Tensor): 标准化空间中的张量，形状 [N, T', D] 或 [N, T', 1]
                             T' 可能与标准化时的 T 不同

        Returns:
            torch.Tensor: 恢复到原始尺度的张量，形状与输入相同

        Raises:
            RuntimeError: 如果在调用 denormalize 前未调用 normalize
        """
        if self.mean is None or self.stdev is None:
            raise RuntimeError(
                "必须先调用 normalize() 计算统计量，才能调用 denormalize()"
            )

        # 反仿射变换: (x - beta) / gamma
        if self.affine:
            # 处理维度广播
            if x.size(-1) == 1:
                # 单特征情况：x 形状 [N, T', 1]
                # 假设该特征对应第一个特征索引（或在外部已选择）
                # 这里不做假设，由调用者在外部处理特征选择
                x = (x - self.beta[0]) / self.gamma[0]
            else:
                # 多特征情况：x 形状 [N, T', D]
                x = (x - self.beta) / self.gamma

        # 反标准化: x * stdev + mean
        # mean 和 stdev 形状: [N, 1, D]
        # x 形状: [N, T', D] 或 [N, T', 1]
        x_denorm = x * self.stdev + self.mean

        return x_denorm

    def extra_repr(self):
        """返回模块的额外信息（用于 print(model) 时显示）"""
        return 'num_features={}, eps={}, affine={}, subtract_last={}'.format(
            self.num_features, self.eps, self.affine, self.subtract_last
        )
