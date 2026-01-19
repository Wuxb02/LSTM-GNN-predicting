"""
GNN模型包装器

从混合LSTM-GNN模型中提取纯GNN部分,用于GNNExplainer分析

关键设计:
- GATWrapper: 提取GAT_LSTM的GAT层和输出层
- GSAGEWrapper: 提取GSAGE_LSTM的SAGE层和输出层
- 共享原模型权重,无需重新训练
- 输入: LSTM输出特征 [num_nodes, hid_dim]
- 输出: 预测值 [num_nodes, pred_len]

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn


class GATWrapper(nn.Module):
    """
    GAT_LSTM模型的GNN部分包装器

    从完整的GAT_LSTM模型中提取GAT图卷积层和MLP输出层,
    用于GNNExplainer分析空间依赖关系

    Args:
        gat_lstm_model: 完整的GAT_LSTM模型实例

    输入: [num_nodes, hid_dim] - LSTM输出特征
    输出: [num_nodes, pred_len] - 预测值
    """

    def __init__(self, gat_lstm_model):
        super().__init__()
        # 引用原模型的GAT层和输出层(共享权重)
        self.GAT_layers = gat_lstm_model.GAT_layers
        self.nGAT_layer = gat_lstm_model.nGAT_layer
        self.element = gat_lstm_model.element

        # 保存维度信息
        self.hid_dim = gat_lstm_model.hid_dim

        self.MLP_layers_out = gat_lstm_model.MLP_layers_out

    def train(self, mode=True):
        """
        覆盖train()方法,确保循环解码器在需要梯度时处于训练模式

        这对于GNNExplainer非常重要,因为cuDNN RNN后端
        只能在训练模式下进行反向传播
        """
        super().train(mode)
        return self

    def eval(self):
        """
        覆盖eval()方法

        当模型使用循环解码器时,我们需要保持解码器在训练模式
        以支持GNNExplainer的反向传播。这是一个特殊处理,
        仅在可解释性分析时使用。
        """
        super().eval()
        return self

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        """
        前向传播(仅GAT部分)

        Args:
            x: [num_nodes, hid_dim] LSTM输出特征
            edge_index: [2, num_edges]
            edge_attr: 边属性(GATv2Conv不使用,保留接口统一)
            return_attention: 是否返回注意力权重

        Returns:
            if return_attention:
                output: [num_nodes, pred_len]
                attention_weights_list: List[(edge_index, attention_weights)]
                    每层GAT的注意力权重,已对多头取平均
            else:
                output: [num_nodes, pred_len]
        """
        attention_weights_list = []

        # GAT图卷积层(复刻原模型逻辑)
        for i in range(self.nGAT_layer):
            base_idx = i * self.element

            # GAT卷积
            if return_attention:
                # 提取注意力权重
                # GATv2Conv.forward(..., return_attention_weights=True)
                # 返回: (Tensor, (edge_index, attention_weights))
                x, (attn_edge_index, attn_weights) = self.GAT_layers[base_idx](
                    x, edge_index, return_attention_weights=True
                )
                # attn_weights shape: [num_edges, num_heads]
                # 对多头取平均
                attn_avg = attn_weights.mean(dim=1)  # [num_edges]
                attention_weights_list.append((attn_edge_index, attn_avg))
            else:
                x = self.GAT_layers[base_idx](x, edge_index)

            # 后续层(Linear, AF, BN?, Dropout?)
            for j in range(1, self.element):
                x = self.GAT_layers[base_idx + j](x)

        # 输出生成(根据解码器类型)
        # MLP输出层(原有方式)
        x = self.MLP_layers_out(x)

        if return_attention:
            return x, attention_weights_list
        return x

    @staticmethod
    def extract_lstm_features(full_model, input_data, edge_index, device):
        """
        从完整GAT_LSTM模型提取LSTM输出特征

        Args:
            full_model: 完整的GAT_LSTM模型
            input_data: [num_nodes, hist_len, in_dim] 原始输入
            edge_index: [2, num_edges]
            device: 'cuda' or 'cpu'

        Returns:
            lstm_features: [num_nodes, hid_dim] LSTM输出特征
        """
        full_model.eval()
        with torch.no_grad():
            # 1. 转换维度
            x = input_data.permute(1, 0, 2)  # [hist_len, num_nodes, in_dim]

            # 2. MLP输入层
            x = full_model.MLP_layers_in(x)  # [hist_len, num_nodes, hid_dim]

            # 3. LSTM时序建模
            out, _ = full_model.lstm(x)
            x = out.contiguous()
            x = x[-1]  # 取最后时间步 [num_nodes, hid_dim] 或 [num_nodes, 2*hid_dim]

            # 4. 双向LSTM投影
            if full_model.lstm_bidirectional:
                x = full_model.lstm_fc(x)  # [num_nodes, hid_dim]

        return x.to(device)


class GSAGEWrapper(nn.Module):
    """
    GSAGE_LSTM模型的GNN部分包装器

    从完整的GSAGE_LSTM模型中提取SAGE图卷积层和MLP输出层

    Args:
        gsage_lstm_model: 完整的GSAGE_LSTM模型实例

    输入: [num_nodes, hid_dim] - LSTM输出特征
    输出: [num_nodes, pred_len] - 预测值
    """

    def __init__(self, gsage_lstm_model):
        super().__init__()
        # 引用原模型的SAGE层和输出层(共享权重)
        self.SAGE_layers = gsage_lstm_model.SAGE_layers
        self.MLP_layers_out = gsage_lstm_model.MLP_layers_out
        self.nSAGE_layer = gsage_lstm_model.nSAGE_layer
        self.element = gsage_lstm_model.element

        # 保存维度信息
        self.hid_dim = gsage_lstm_model.hid_dim

    def forward(self, x, edge_index, edge_attr=None):
        """
        前向传播(仅SAGE部分)

        Args:
            x: [num_nodes, hid_dim] LSTM输出特征
            edge_index: [2, num_edges]
            edge_attr: 边属性(SAGEConv不使用,保留接口统一)

        Returns:
            output: [num_nodes, pred_len]
        """
        # SAGE图卷积层(复刻原模型逻辑)
        for i in range(self.nSAGE_layer):
            base_idx = i * self.element
            # SAGE卷积
            x = self.SAGE_layers[base_idx](x, edge_index)
            # 后续层(AF, BN?, Dropout?)
            for j in range(1, self.element):
                x = self.SAGE_layers[base_idx + j](x)

        # MLP输出层
        x = self.MLP_layers_out(x)

        return x

    @staticmethod
    def extract_lstm_features(full_model, input_data, edge_index, device):
        """
        从完整GSAGE_LSTM模型提取LSTM输出特征

        Args:
            full_model: 完整的GSAGE_LSTM模型
            input_data: [num_nodes, hist_len, in_dim] 原始输入
            edge_index: [2, num_edges]
            device: 'cuda' or 'cpu'

        Returns:
            lstm_features: [num_nodes, hid_dim] LSTM输出特征
        """
        full_model.eval()
        with torch.no_grad():
            # 1. 转换维度
            x = input_data.permute(1, 0, 2)  # [hist_len, num_nodes, in_dim]

            # 2. MLP输入层
            x = full_model.MLP_layers_in(x)  # [hist_len, num_nodes, hid_dim]

            # 3. LSTM时序建模
            out, _ = full_model.lstm(x)
            x = out.contiguous()
            x = x[-1]  # 取最后时间步

            # 4. 双向LSTM投影
            if full_model.lstm_bidirectional:
                x = full_model.lstm_fc(x)

        return x.to(device)


class GATSeparateEncoderWrapper(nn.Module):
    """
    GAT_SeparateEncoder模型的GNN部分包装器

    从完整的GAT_SeparateEncoder模型中提取GAT图卷积层和解码器,
    用于GNNExplainer分析空间依赖关系

    Args:
        model: 完整的GAT_SeparateEncoder模型实例

    输入: [num_nodes, hid_dim] - 编码器输出特征
    输出: [num_nodes, pred_len] - 预测值
    """

    def __init__(self, gat_separate_model):
        super().__init__()
        # 引用原模型的GAT层和解码器(共享权重)
        self.GAT_layers = gat_separate_model.GAT_layers
        self.nGAT_layer = gat_separate_model.nGAT_layer
        self.element = gat_separate_model.element

        # 保存维度信息
        self.hid_dim = gat_separate_model.hid_dim

        self.MLP_layers_out = gat_separate_model.MLP_layers_out

    def train(self, mode=True):
        """覆盖train()方法"""
        super().train(mode)
        return self

    def eval(self):
        super().eval()
        return self

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        """前向传播(仅GAT和解码器部分)"""
        attention_weights_list = []

        # GAT图卷积层
        for i in range(self.nGAT_layer):
            base_idx = i * self.element

            if return_attention:
                x, (attn_edge_index, attn_weights) = self.GAT_layers[base_idx](
                    x, edge_index, return_attention_weights=True
                )
                attn_avg = attn_weights.mean(dim=1)
                attention_weights_list.append((attn_edge_index, attn_avg))
            else:
                x = self.GAT_layers[base_idx](x, edge_index)

            for j in range(1, self.element):
                x = self.GAT_layers[base_idx + j](x)

        # 输出生成
        x = self.MLP_layers_out(x)

        if return_attention:
            return x, attention_weights_list
        return x

    @staticmethod
    def extract_encoder_features(full_model, input_data, edge_index, device):
        """
        从完整GAT_SeparateEncoder模型提取编码器输出特征

        支持 v2.0 和 v3.0 两种架构:
        - v2.0: static_encoder + dynamic_encoder + fusion(static_emb, dynamic_emb)
        - v3.0: dynamic_encoder + fusion(static_features, dynamic_emb)
                fusion内部包含CrossAttentionFusionV2.static_encoder

        Args:
            full_model: 完整的GAT_SeparateEncoder模型
            input_data: [num_nodes, hist_len, in_dim] 原始输入
            edge_index: [2, num_edges]
            device: 'cuda' or 'cpu'

        Returns:
            encoder_features: [num_nodes, hid_dim] 编码器输出特征
        """
        full_model.eval()
        with torch.no_grad():
            num_nodes, hist_len, in_dim = input_data.shape

            # 1. 特征分离
            static_features = input_data[:, 0, :full_model.static_dim]
            dynamic_features = input_data[:, :, full_model.static_dim:]

            # 2. 动态编码 (v2.0 和 v3.0 共用)
            dynamic_emb = full_model.dynamic_encoder(dynamic_features)

            # 3. 检测架构版本并进行特征融合
            if hasattr(full_model, 'static_encoder'):
                static_emb = full_model.static_encoder(static_features)
                x = full_model.fusion(static_emb, dynamic_emb)
            else:
                # v3.0: fusion(static_features, dynamic_emb)
                # CrossAttentionFusionV2.forward 内部调用 self.static_encoder
                x = full_model.fusion(static_features, dynamic_emb)

        return x.to(device)


def create_gnn_wrapper(model, model_type=None):
    """
    工厂函数: 根据模型类型创建对应的GNN包装器

    Args:
        model: 完整的LSTM-GNN混合模型
        model_type: 模型类型 ('GAT_LSTM', 'GSAGE_LSTM', 'GAT_SeparateEncoder', None)
                   如果为None,则自动检测

    Returns:
        GATWrapper/GSAGEWrapper/GATSeparateEncoderWrapper 实例

    Raises:
        ValueError: 如果模型类型不支持

    示例:
        >>> # 自动检测
        >>> wrapper = create_gnn_wrapper(model)
        >>>
        >>> # 手动指定
        >>> wrapper = create_gnn_wrapper(model, 'GAT_LSTM')
    """
    if model_type is None:
        # 自动检测模型类型
        from .utils import detect_model_type
        model_type = detect_model_type(model)

    if model_type == 'GAT_LSTM':
        return GATWrapper(model)
    elif model_type == 'GSAGE_LSTM':
        return GSAGEWrapper(model)
    elif model_type == 'GAT_SeparateEncoder':
        return GATSeparateEncoderWrapper(model)
    else:
        raise ValueError(
            f"不支持的模型类型: {model_type}. "
            f"支持的类型: GAT_LSTM, GSAGE_LSTM, GAT_SeparateEncoder"
        )


def verify_wrapper_consistency(full_model, wrapper, test_input, edge_index, device='cpu'):
    """
    验证Wrapper输出与原模型一致性

    Args:
        full_model: 完整模型
        wrapper: GNN包装器
        test_input: [num_nodes, hist_len, in_dim] 测试输入
        edge_index: [2, num_edges]
        device: 'cuda' or 'cpu'

    Returns:
        bool: 是否一致(误差<1e-5)
        float: 最大绝对误差

    示例:
        >>> is_consistent, max_error = verify_wrapper_consistency(
        >>>     model, wrapper, test_data, edge_index
        >>> )
        >>> assert is_consistent, f"不一致! 误差: {max_error}"
    """
    full_model.eval()
    wrapper.eval()

    # 原模型预测
    with torch.no_grad():
        pred_original = full_model(
            test_input.to(device),
            edge_index.to(device)
        )

    # Wrapper预测
    # 1. 提取编码器特征（根据模型类型选择方法）
    if isinstance(wrapper, GATWrapper):
        encoder_features = GATWrapper.extract_lstm_features(
            full_model, test_input, edge_index, device
        )
    elif isinstance(wrapper, GSAGEWrapper):
        encoder_features = GSAGEWrapper.extract_lstm_features(
            full_model, test_input, edge_index, device
        )
    elif isinstance(wrapper, GATSeparateEncoderWrapper):
        encoder_features = GATSeparateEncoderWrapper.extract_encoder_features(
            full_model, test_input, edge_index, device
        )
    else:
        raise ValueError(f"未知的wrapper类型: {type(wrapper)}")

    # 2. Wrapper前向传播
    with torch.no_grad():
        pred_wrapper = wrapper(
            encoder_features.to(device),
            edge_index.to(device)
        )

    # 计算误差
    max_error = (pred_original - pred_wrapper).abs().max().item()
    is_consistent = max_error < 1e-5

    return is_consistent, max_error
