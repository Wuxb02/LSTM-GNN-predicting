"""
GAT + 分离式编码器 模型 (优化版 v3.0)

核心改进：
1. Cross-Attention Fusion V2: 特征级交叉注意力（动态查询各静态特征维度）
2. Learnable Node Embeddings: 可学习节点嵌入捕获隐式站点特征
3. Skip Connection: GAT输入到输出的残差连接（防止过度平滑）
4. 可解释性增强: 支持提取特征级注意力权重

架构对比：
- v1.0: 分离式编码器(静态MLP + 动态LSTM) → 简单融合 → GAT → 解码器
- v2.0: 分离式编码器 → Cross-Attention融合 → GAT(+Skip) → 解码器
- v3.0: 分离式编码器 → 特征级Cross-Attention → GAT(+Skip) → 解码器
        + 注意力权重可解释性

优势：
1. 动态特征自适应地"查询"最相关的静态地理信息
2. 特征级注意力权重可解释：展示模型对各静态特征的关注程度
3. 节点嵌入捕获数据未记录的微气候效应（如街道峡谷效应）
4. 残差连接保留站点自身历史趋势，防止图卷积过度平滑

作者: GNN气温预测项目
日期: 2025-12
版本: 3.0
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

# 导入 RevIN 层
from .layers import RevIN


def get_norm_layer(norm_type, dim):
    """规范化层选择"""
    if norm_type == 'BatchNorm':
        return nn.BatchNorm1d(dim)
    elif norm_type == 'LayerNorm':
        return nn.LayerNorm(dim)
    elif norm_type == 'None' or norm_type is None:
        return None
    else:
        raise ValueError(f"未知的规范化类型: {norm_type}")


def whichAF(AF):
    """激活函数选择"""
    if AF == 'PReLU':
        return nn.PReLU()
    elif AF == "LeakyReLU":
        return nn.LeakyReLU()
    elif AF == "ReLU":
        return nn.ReLU()
    elif AF == 'GELU':
        return nn.GELU()
    else:
        return nn.Identity()


class LightweightStaticEncoder(nn.Module):
    """
    轻量级静态特征编码器 (优化版)

    保留 [N, 12, dim] 的输出结构以支持特征级注意力，
    但移除笨重的 MLP，改用 "特征值 * 可学习基向量" 的方式。

    参数量极低，极难过拟合，且完全保留可解释性。
    """

    def __init__(self, num_features, output_dim):
        super(LightweightStaticEncoder, self).__init__()
        self.num_features = num_features
        self.output_dim = output_dim

        # 定义基向量 (Basis Vectors)
        # 形状: [1, 12, output_dim]
        # 这里的每一个向量代表一种特征的"语义身份"
        self.feature_embeddings = nn.Parameter(
            torch.randn(1, num_features, output_dim) * 0.02
        )

        # 可选：如果你担心简单的乘法表达能力不够，可以加一个共享的层归一化
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_features]  (原始静态特征值，标量)

        Returns:
            [batch_size, num_features, output_dim] (特征Token序列)
        """
        batch_size = x.shape[0]

        # 1. 扩展输入维度
        # x: [batch, 12] -> [batch, 12, 1]
        x_expanded = x.unsqueeze(-1)

        # 2. 广播乘法 (Scaling)
        # 将标量特征值 乘以 对应的特征身份向量
        # [batch, 12, 1] * [1, 12, dim] -> [batch, 12, dim]
        # 广播机制会自动处理维度匹配
        tokens = x_expanded * self.feature_embeddings

        # 3. 归一化 (让训练更稳定)
        tokens = self.norm(tokens)

        return tokens


class StaticFeatureEncoder(nn.Module):
    """
    静态特征独立编码器 (v3.0 新增)

    将每个静态特征维度独立编码为token，用于特征级Cross-Attention。
    每个静态特征经过独立的MLP编码，保持特征间的区分度。

    输入: [N, num_static_features] 原始静态特征（12维）
    输出: [N, num_static_features, token_dim] 每个特征一个token
    """

    def __init__(self, num_features, token_dim, dropout=0.1):
        """
        Args:
            num_features: 静态特征数量（如12）
            token_dim: 每个token的维度
            dropout: Dropout率
        """
        super(StaticFeatureEncoder, self).__init__()

        self.num_features = num_features
        self.token_dim = token_dim

        # 每个静态特征独立的编码器
        # 输入: 1维标量 → 输出: token_dim维向量
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, token_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(token_dim, token_dim)
            ) for _ in range(num_features)
        ])

        # 可学习的位置编码（区分不同特征的位置）
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_features, token_dim) * 0.02
        )

    def forward(self, x):
        """
        Args:
            x: [num_nodes, num_features] 静态特征

        Returns:
            [num_nodes, num_features, token_dim] 特征token序列
        """
        batch_size = x.shape[0]
        tokens = []

        # 每个特征独立编码
        for i in range(self.num_features):
            # 提取第i个特征: [N, 1]
            feat_i = x[:, i:i+1]
            # 编码为token: [N, token_dim]
            token_i = self.feature_encoders[i](feat_i)
            tokens.append(token_i)

        # 堆叠: [N, num_features, token_dim]
        tokens = torch.stack(tokens, dim=1)

        # 添加位置编码
        tokens = tokens + self.position_embedding

        return tokens


class DynamicEncoder(nn.Module):
    """
    动态特征编码器

    使用LSTM对动态特征（气象要素等）进行时序编码。
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1,
                 dropout=0.1, bidirectional=False):
        super(DynamicEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=False
        )

        # 双向LSTM投影
        if bidirectional:
            self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [num_nodes, hist_len, dynamic_dim] 动态特征

        Returns:
            [num_nodes, hidden_dim] 动态嵌入（取最后时间步）
        """
        # [num_nodes, hist_len, dynamic_dim] → [hist_len, num_nodes, dynamic_dim]
        x = x.permute(1, 0, 2)

        # 输入投影
        x = self.input_proj(x)  # [hist_len, num_nodes, hidden_dim]

        # LSTM编码
        out, _ = self.lstm(x)
        # 取最后时间步 [num_nodes, hidden_dim] 或 [num_nodes, 2*hidden_dim]
        out = out[-1]

        # 双向投影
        if self.bidirectional:
            out = self.output_proj(out)

        return out


class CrossAttentionFusionV2(nn.Module):
    """
    特征级交叉注意力融合模块 (v3.0 新增)

    核心改进：
    - 将12个静态特征作为独立的K/V序列
    - 注意力权重形状: [N, num_heads, 1, 12] → 对每个静态特征的关注程度
    - 支持提取注意力权重用于可解释性分析

    架构：
    - Query (Q): 动态特征编码 [N, 1, dim]
    - Key (K) & Value (V): 静态特征tokens [N, 12, dim]
    - 输出: 融合表示 [N, dim] + 可选的注意力权重 [N, num_heads, 12]
    """

    def __init__(self, num_static_features, dynamic_dim, output_dim,
                 num_heads=4, dropout=0.1, use_pre_ln=True):
        """
        Args:
            num_static_features: 静态特征数量（如12）
            dynamic_dim: 动态特征维度（LSTM输出维度）
            output_dim: 输出维度
            num_heads: 注意力头数
            dropout: Dropout率
            use_pre_ln: 是否使用Pre-LN
        """
        super(CrossAttentionFusionV2, self).__init__()

        self.num_static_features = num_static_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_pre_ln = use_pre_ln

        # 静态特征独立编码器
        self.static_encoder = StaticFeatureEncoder(
            num_features=num_static_features,
            token_dim=output_dim,
            dropout=dropout
        )

        # self.static_encoder = LightweightStaticEncoder(
        #     num_features=num_static_features,
        #     output_dim=output_dim  # 确保这里维度和 dynamic_emb 投影后的维度一致
        # )

        # 动态特征投影
        self.dynamic_proj = nn.Linear(dynamic_dim, output_dim)

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm层
        self.ln1_q = nn.LayerNorm(output_dim)
        self.ln1_kv = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

        # FeedForward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, static_features, dynamic_emb, return_attention=False):
        """
        Args:
            static_features: [N, num_static_features] 原始静态特征（12维）
            dynamic_emb: [N, dynamic_dim] 动态特征编码（LSTM输出）
            return_attention: 是否返回注意力权重

        Returns:
            如果 return_attention=False:
                [N, output_dim] 融合后的表示
            如果 return_attention=True:
                ([N, output_dim], [N, num_heads, num_static_features])
        """
        # 静态特征编码为token序列
        # [N, 12] → [N, 12, output_dim]
        static_tokens = self.static_encoder(static_features)

        # 动态特征投影
        # [N, dynamic_dim] → [N, output_dim]
        dynamic_out = self.dynamic_proj(dynamic_emb)

        # 构建Q, K, V
        # Q: [N, 1, output_dim] (动态特征作为查询)
        # K, V: [N, 12, output_dim] (静态特征tokens作为键值)
        q = dynamic_out.unsqueeze(1)  # [N, 1, output_dim]
        k = static_tokens              # [N, 12, output_dim]
        v = static_tokens              # [N, 12, output_dim]

        # ==================== 注意力块 ====================
        if self.use_pre_ln:
            q_norm = self.ln1_q(q)
            k_norm = self.ln1_kv(k)
            v_norm = self.ln1_kv(v)
            attn_out, attn_weights = self.cross_attention(
                q_norm, k_norm, v_norm,
                need_weights=True,
                average_attn_weights=False  # 保留多头信息
            )
            # attn_out: [N, 1, output_dim]
            # attn_weights: [N, num_heads, 1, 12]
            attn_out = attn_out.squeeze(1)  # [N, output_dim]
            x = dynamic_out + attn_out
        else:
            attn_out, attn_weights = self.cross_attention(
                q, k, v,
                need_weights=True,
                average_attn_weights=False
            )
            attn_out = attn_out.squeeze(1)
            x = dynamic_out + attn_out
            x = self.ln1_q(x)

        # ==================== FFN块 ====================
        if self.use_pre_ln:
            ffn_out = self.ffn(self.ln2(x))
            x = x + ffn_out
        else:
            ffn_out = self.ffn(x)
            x = x + ffn_out
            x = self.ln2(x)

        if return_attention:
            # attn_weights: [N, num_heads, 1, 12] → [N, num_heads, 12]
            attn_weights = attn_weights.squeeze(2)
            return x, attn_weights
        else:
            return x


class GAT_SeparateEncoder(nn.Module):
    """
    GAT + 分离式编码器 模型 (优化版 v3.0)

    架构:
    1. 可学习节点嵌入 ⭐
       - 捕获隐式站点特征（如微气候、街道峡谷效应等）
    2. 分离式编码器:
       - 动态特征 → DynamicEncoder(LSTM) → 动态嵌入
       - 静态特征(12维) → CrossAttentionFusionV2 → 特征级注意力融合
    3. GAT图卷积层 x N (带残差连接) ⭐
    4. LSTM解码器（多步预测）

    v3.0 新特性:
    - 特征级Cross-Attention: 12个静态特征作为独立K/V
    - 注意力权重可解释: 展示模型对各静态特征的关注程度
    - 支持提取注意力权重用于可视化分析

    输入格式:
    x: [num_nodes, hist_len, in_dim]
    其中 in_dim = static_dim(12) + dynamic_dim(12) + temporal_dim(4)
    """

    def __init__(self, config, arch_arg):
        super(GAT_SeparateEncoder, self).__init__()

        # 保存配置
        self.config = config

        # 从config获取特征分离参数
        self.use_feature_separation = getattr(
            config, 'use_feature_separation', True
        )
        self.static_dim = getattr(config, 'static_encoded_dim', 8)
        self.dynamic_dim = len(getattr(
            config, 'dynamic_feature_indices', list(range(12))
        ))
        self.temporal_dim = (
            config.temporal_features if config.add_temporal_encoding else 0
        )

        # 模型参数
        self.hid_dim = arch_arg.hid_dim
        self.nGAT_layer = arch_arg.GAT_layer
        self.heads = arch_arg.heads
        self.out_dim = config.pred_len

        AF = whichAF(arch_arg.AF)

        # ==================== 🔥 可学习节点嵌入 ====================
        # 捕获未被数据记录的隐式站点特征（如微气候效应）
        self.node_emb_dim = getattr(arch_arg, 'node_emb_dim', 4)  # 默认4维
        self.use_node_embedding = getattr(arch_arg, 'use_node_embedding', True)

        if self.use_node_embedding:
            # 初始化可训练的节点嵌入 [num_nodes, node_emb_dim]
            self.node_embedding = nn.Parameter(
                torch.randn(config.node_num, self.node_emb_dim) * 0.01
            )
        else:
            self.node_embedding = None

        # ==================== 分离式编码器 ====================
        # 动态编码器（LSTM处理时序特征）
        dynamic_input_dim = self.dynamic_dim + self.temporal_dim
        self.dynamic_encoder = DynamicEncoder(
            input_dim=dynamic_input_dim,
            hidden_dim=self.hid_dim // 2,
            num_layers=arch_arg.lstm_num_layers,
            dropout=arch_arg.lstm_dropout,
            bidirectional=arch_arg.lstm_bidirectional
        )

        # ==================== 🔥 v3.0: 特征级交叉注意力融合 ====================
        # 12个静态特征作为独立的K/V，支持提取特征级注意力权重
        fusion_num_heads = getattr(arch_arg, 'fusion_num_heads', 4)
        fusion_use_pre_ln = getattr(arch_arg, 'fusion_use_pre_ln', True)

        # 静态特征数量（包含节点嵌入）
        num_static_features = self.static_dim
        if self.use_node_embedding:
            num_static_features += self.node_emb_dim
        self.num_static_features = num_static_features

        self.fusion = CrossAttentionFusionV2(
            num_static_features=num_static_features,
            dynamic_dim=self.hid_dim // 2,
            output_dim=self.hid_dim,
            num_heads=fusion_num_heads,
            dropout=arch_arg.inter_drop,
            use_pre_ln=fusion_use_pre_ln
        )

        # ==================== GAT层 ====================
        GAT_layers = []

        # 计算每个GAT块的层数
        self.element = 3  # GAT + Linear + AF
        if arch_arg.norm_type and arch_arg.norm_type != 'None':
            self.element += 1
        if arch_arg.dropout:
            self.element += 1

        for n in range(self.nGAT_layer):
            GAT_layers.append(
                GATv2Conv(
                    self.hid_dim, self.hid_dim,
                    heads=self.heads, concat=True,
                    dropout=arch_arg.intra_drop,
                    add_self_loops=True, share_weights=False
                )
            )
            GAT_layers.append(
                nn.Linear(self.heads * self.hid_dim, self.hid_dim))
            GAT_layers.append(AF)

            norm_layer = get_norm_layer(arch_arg.norm_type, self.hid_dim)
            if norm_layer is not None:
                GAT_layers.append(norm_layer)
            if arch_arg.dropout:
                GAT_layers.append(nn.Dropout(arch_arg.inter_drop))

        self.GAT_layers = nn.ModuleList(GAT_layers)

        # ==================== 🔥 新增2: 残差连接控制 ====================
        # 在GAT输入（融合输出）和GAT输出之间添加跳跃连接
        self.use_skip_connection = getattr(
            arch_arg, 'use_skip_connection', True)

        # ==================== 解码器 ====================
        self.use_recurrent_decoder = getattr(
            arch_arg, 'use_recurrent_decoder', True
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

            decoder_input_size = (
                1 + self.hid_dim if self.decoder_use_context else 1
            )

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
                raise ValueError(f"未知的解码器类型: {self.decoder_type}")

            self.decoder_output_proj = nn.Linear(self.hid_dim, 1)
            self.decoder_init_proj = nn.Linear(self.hid_dim, 1)
        else:
            # MLP输出层
            MLP_layers_out = []
            for n in range(arch_arg.MLP_layer):
                MLP_layers_out.append(nn.Linear(self.hid_dim, self.hid_dim))
                MLP_layers_out.append(AF)
            MLP_layers_out.append(nn.Linear(self.hid_dim, self.out_dim))
            self.MLP_layers_out = nn.Sequential(*MLP_layers_out)

        # ==================== RevIN 层（新增）⭐ ====================
        # 用于处理非平稳时间序列的分布偏移问题
        self.use_revin = getattr(arch_arg, 'use_revin', False)
        if self.use_revin:
            # 仅对动态特征应用 RevIN（静态特征不随时间变化）
            # dynamic_dim: 动态气象要素数量
            # temporal_dim: 时间编码维度（sin/cos）
            revin_num_features = self.dynamic_dim + self.temporal_dim
            self.revin_layer = RevIN(
                num_features=revin_num_features,
                eps=getattr(arch_arg, 'revin_eps', 1e-5),
                affine=getattr(arch_arg, 'revin_affine', True),
                subtract_last=getattr(arch_arg, 'revin_subtract_last', False)
            )
            print(f"✓ RevIN 已启用 (特征数={revin_num_features}, "
                  f"affine={self.revin_layer.affine}, "
                  f"subtract_last={self.revin_layer.subtract_last})")
        else:
            self.revin_layer = None

    def forward(self, x, edge_index, edge_attr=None, return_cross_attention=False):
        """
        前向传播

        Args:
            x: [batch_size * num_nodes, hist_len, in_dim] 或 [num_nodes, hist_len, in_dim]
               其中 in_dim = static_dim + dynamic_dim + temporal_dim
               数据格式: [静态特征(12), 动态特征(12), 时间编码(4)]
            edge_index: [2, num_edges]
            edge_attr: 边属性（可选）
            return_cross_attention: 是否返回Cross-Attention权重（用于可解释性分析）

        Returns:
            如果 return_cross_attention=False:
                [batch_size * num_nodes, pred_len] 预测结果
            如果 return_cross_attention=True:
                (预测结果, Cross-Attention权重 [N, num_heads, num_static_features])
        """
        total_nodes, hist_len, in_dim = x.shape

        # ==================== 1. 特征分离 ====================
        # 从组合输入中分离静态和动态部分
        # 输入格式: [静态特征(12), 动态特征(12), 时间编码(4)]
        # [total_nodes, static_dim]
        static_features = x[:, 0, :self.static_dim]
        # [total_nodes, hist_len, dynamic_dim+temporal_dim]
        dynamic_features = x[:, :, self.static_dim:]

        # ==================== 新增: RevIN 标准化 ⭐ ====================
        if self.use_revin:
            # 仅对动态特征应用 RevIN（时间编码也包含在内）
            # dynamic_features: [total_nodes, hist_len, dynamic_dim+temporal_dim]
            dynamic_features = self.revin_layer.normalize(dynamic_features)

        # ==================== 2. 节点嵌入增强 ⭐ ====================
        if self.use_node_embedding:
            # 计算实际的节点数和批次大小
            batch_size = total_nodes // self.node_embedding.shape[0]

            # 将节点嵌入扩展到批次维度
            node_emb_expanded = self.node_embedding.unsqueeze(0).expand(
                batch_size, -1, -1
            ).reshape(total_nodes, -1)

            # 将可学习的节点嵌入与静态特征拼接
            # [total_nodes, static_dim + node_emb_dim]
            static_features = torch.cat(
                [static_features, node_emb_expanded], dim=-1)

        # ==================== 3. 动态编码 ====================
        dynamic_emb = self.dynamic_encoder(
            dynamic_features)  # [total_nodes, hid_dim//2]

        # ==================== 4. 特征级交叉注意力融合 (v3.0) ⭐ ====================
        # 动态特征作为Query，查询最相关的静态地理信息
        # 返回融合表示 + 可选的特征级注意力权重
        if return_cross_attention:
            fusion_out, cross_attn_weights = self.fusion(
                static_features, dynamic_emb, return_attention=True
            )
            # cross_attn_weights: [N, num_heads, num_static_features]
        else:
            fusion_out = self.fusion(
                static_features, dynamic_emb, return_attention=False)

        # ==================== 5. GAT图卷积（带残差连接）⭐ ====================
        x = fusion_out  # 保存融合输出，用于后续残差连接

        for i in range(self.nGAT_layer):
            base_idx = i * self.element
            x = self.GAT_layers[base_idx](x, edge_index)
            for j in range(1, self.element):
                x = self.GAT_layers[base_idx + j](x)

        # 🔥 新增: 残差连接（防止过度平滑，保留站点自身历史趋势）
        if self.use_skip_connection:
            # Element-wise Addition: GAT输出 + 融合输出
            x = x + fusion_out

        # ==================== 6. 解码器 ====================
        if self.use_recurrent_decoder:
            outputs = []
            encoder_context = x

            num_layers = self.decoder.num_layers
            if self.decoder_type == 'LSTM':
                h_0 = x.unsqueeze(0).repeat(num_layers, 1, 1)
                c_0 = h_0.clone()
                hidden = (h_0, c_0)
            else:
                hidden = x.unsqueeze(0).repeat(num_layers, 1, 1)

            prev_pred = self.decoder_init_proj(x)

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

            x = torch.cat(outputs, dim=1)
        else:
            x = self.MLP_layers_out(x)

        # ==================== 新增: RevIN 反标准化 ⭐ ====================
        if self.use_revin:
            # 计算目标特征在动态特征中的索引
            # config.target_feature_idx 是全局索引（0-27）
            # 需要映射到动态特征中的索引（0-9）
            target_global_idx = self.config.target_feature_idx
            dynamic_indices = self.config.dynamic_feature_indices

            if target_global_idx in dynamic_indices:
                target_idx_in_dynamic = dynamic_indices.index(target_global_idx)
            else:
                # 如果目标不在动态特征中，使用第一个动态特征的统计量
                print(f"警告: 目标特征索引 {target_global_idx} 不在动态特征列表中，"
                      f"使用第一个动态特征的统计量")
                target_idx_in_dynamic = 0

            # 扩展输出维度: [total_nodes, pred_len] → [total_nodes, pred_len, 1]
            output_expanded = x.unsqueeze(-1)

            # 提取目标特征的统计量
            # mean/stdev 形状: [total_nodes, 1, dynamic_dim+temporal_dim]
            # 选择第 target_idx_in_dynamic 个特征: [total_nodes, 1, 1]
            mean_target = self.revin_layer.mean[:, :, target_idx_in_dynamic:target_idx_in_dynamic+1]
            stdev_target = self.revin_layer.stdev[:, :, target_idx_in_dynamic:target_idx_in_dynamic+1]

            # 反仿射变换（如果启用）
            if self.revin_layer.affine:
                gamma_target = self.revin_layer.gamma[target_idx_in_dynamic]
                beta_target = self.revin_layer.beta[target_idx_in_dynamic]
                output_expanded = (output_expanded - beta_target) / gamma_target

            # 反标准化: output * stdev + mean
            output_expanded = output_expanded * stdev_target + mean_target

            # 压缩回原始形状: [total_nodes, pred_len, 1] → [total_nodes, pred_len]
            x = output_expanded.squeeze(-1)

        # 返回结果
        if return_cross_attention:
            return x, cross_attn_weights
        else:
            return x


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("GAT_SeparateEncoder 模型测试 (v3.0 - 特征级注意力)")
    print("=" * 70)

    # 模拟配置
    class MockConfig:
        in_dim = 28  # 12(静态) + 12(动态) + 4(时间编码)
        pred_len = 3
        node_num = 28
        use_feature_separation = True
        static_encoded_dim = 12  # 12个静态特征
        dynamic_feature_indices = [3, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 23]
        add_temporal_encoding = True
        temporal_features = 4

    class MockArchArg:
        hid_dim = 32
        MLP_layer = 1
        AF = 'ReLU'
        norm_type = 'LayerNorm'
        dropout = True
        GAT_layer = 2
        heads = 2
        intra_drop = 0.1
        inter_drop = 0.1
        lstm_num_layers = 1
        lstm_dropout = 0.1
        lstm_bidirectional = False
        use_recurrent_decoder = True
        decoder_type = 'LSTM'
        decoder_num_layers = 1
        decoder_dropout = 0.1
        decoder_use_context = False
        decoder_mlp_layers = 0

        # v3.0 参数
        use_node_embedding = True   # 启用节点嵌入
        node_emb_dim = 4            # 节点嵌入维度
        fusion_num_heads = 4        # 交叉注意力头数
        fusion_use_pre_ln = True    # 使用Pre-LN
        use_skip_connection = True  # 启用残差连接

    config = MockConfig()
    arch_arg = MockArchArg()

    print(f"\n{'='*50}")
    print("测试 v3.0 特征级注意力模型")
    print(f"  - 节点嵌入: {'启用' if arch_arg.use_node_embedding else '禁用'}")
    print(f"  - 静态特征数: {config.static_encoded_dim}")
    print(f"  - 交叉注意力: {arch_arg.fusion_num_heads} 头")
    print(f"  - 残差连接: {'启用' if arch_arg.use_skip_connection else '禁用'}")
    print(f"{'='*50}")

    model = GAT_SeparateEncoder(config, arch_arg)
    model.eval()  # Set to evaluation mode for testing

    # 模拟输入
    batch_size = 28
    hist_len = 7
    x = torch.randn(batch_size, hist_len, config.in_dim)
    edge_index = torch.randint(0, batch_size, (2, 100))

    print(f"\n输入形状: {x.shape}")

    # Test 1: Normal forward pass
    print("\n[Test 1] Normal forward pass")
    out = model(x, edge_index)
    print(f"  Output shape: {out.shape}")
    print(f"  Expected shape: [{batch_size}, {config.pred_len}]")
    assert out.shape == (
        batch_size, config.pred_len), f"Shape mismatch: {out.shape}"
    print("  [OK] Shape validation passed")

    # Test 2: Forward pass with attention weights
    print("\n[Test 2] Forward pass with Cross-Attention weights")
    out, attn_weights = model(x, edge_index, return_cross_attention=True)
    print(f"  Output shape: {out.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")
    num_static_features = config.static_encoded_dim + arch_arg.node_emb_dim
    expected_attn_shape = (
        batch_size, arch_arg.fusion_num_heads, num_static_features)
    print(f"  Expected attention shape: {expected_attn_shape}")
    assert attn_weights.shape == expected_attn_shape, \
        f"Attention shape mismatch: {attn_weights.shape}"
    print("  [OK] Attention weights shape validation passed")

    # Verify attention weights are normalized (sum to 1)
    attn_sum = attn_weights[0, 0, :].sum().item()
    print(f"  Attention weights sum (should be ~1.0): {attn_sum:.4f}")
    assert abs(
        attn_sum - 1.0) < 0.01, f"Attention weights not normalized: {attn_sum}"
    print("  [OK] Attention weights normalization passed")

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"\nParameter Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Check node embeddings
    if arch_arg.use_node_embedding:
        print(f"  - Node embedding shape: {model.node_embedding.shape}")
        print(f"  - Node embedding params: {model.node_embedding.numel()}")

    # Check static features count
    print(f"  - Static features (with node emb): {model.num_static_features}")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
