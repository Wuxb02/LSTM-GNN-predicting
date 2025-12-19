"""
空间关系可解释性分析器

使用GNNExplainer分析GNN模型的空间依赖关系,
解释哪些气象站之间的边连接对预测最重要

作者: GNN气温预测项目
日期: 2025
"""

import torch
import numpy as np
from torch_geometric.explain import Explainer, GNNExplainer

from .gnn_wrapper import (
    create_gnn_wrapper,
    GATWrapper,
    GSAGEWrapper,
    GATSeparateEncoderWrapper
)
from .utils import get_top_k_edges, get_original_dataset


class SpatialExplainer:
    """
    GNN空间关系解释器

    使用GNNExplainer分析GAT/GSAGE模型的边重要性,
    量化不同气象站之间空间依赖关系的重要程度

    Args:
        model: 完整的LSTM-GNN混合模型
        config: Config配置对象
        explainer_config: ExplainerConfig可解释性配置

    示例:
        >>> explainer = SpatialExplainer(model, config, exp_config)
        >>> result = explainer.explain_batch(test_loader, num_samples=100)
        >>> print("Top-5重要边:", result['important_edges'][:5])
    """

    def __init__(self, model, config, explainer_config):
        self.full_model = model
        self.config = config
        self.exp_config = explainer_config
        self.device = config.device

        # 创建GNN wrapper
        self.gnn_wrapper = create_gnn_wrapper(model)
        # 注意: GNNExplainer需要模型在训练模式下进行反向传播
        # 特别是当使用循环解码器(LSTM/GRU)时,cuDNN后端要求训练模式
        self.gnn_wrapper.train()  # 设置为训练模式以支持梯度计算
        self.gnn_wrapper.to(self.device)

        print(f"✓ 创建GNN包装器: {type(self.gnn_wrapper).__name__}")

        # 初始化PyG Explainer
        self.explainer = Explainer(
            model=self.gnn_wrapper,
            algorithm=GNNExplainer(epochs=explainer_config.epochs),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='regression',
                task_level='node',
                return_type='raw',
            ),
        )

        print(f"✓ 初始化GNNExplainer (epochs={explainer_config.epochs})")

    def explain_single(self, input_data, edge_index, target_node=None):
        """
        解释单个样本

        Args:
            input_data: [num_nodes, hist_len, in_dim] 原始输入
            edge_index: [2, num_edges]
            target_node: 目标节点索引(None=全图解释)

        Returns:
            dict: {
                'edge_mask': [num_edges] 边重要性权重,
                'node_mask': [num_nodes, hid_dim] 节点特征重要性(可选)
            }
        """
        # 1. 提取编码器特征
        if isinstance(self.gnn_wrapper, GATWrapper):
            encoder_features = GATWrapper.extract_lstm_features(
                self.full_model, input_data, edge_index, self.device
            )
        elif isinstance(self.gnn_wrapper, GSAGEWrapper):
            encoder_features = GSAGEWrapper.extract_lstm_features(
                self.full_model, input_data, edge_index, self.device
            )
        elif isinstance(self.gnn_wrapper, GATSeparateEncoderWrapper):
            encoder_features = GATSeparateEncoderWrapper.extract_encoder_features(
                self.full_model, input_data, edge_index, self.device
            )
        else:
            raise ValueError(f"未知的wrapper类型: {type(self.gnn_wrapper)}")

        # 2. 运行GNNExplainer
        if target_node is None:
            # 全图解释
            explanation = self.explainer(
                x=encoder_features,
                edge_index=edge_index,
            )
        else:
            # 单节点解释
            explanation = self.explainer(
                x=encoder_features,
                edge_index=edge_index,
                index=target_node,
            )

        return {
            'edge_mask': explanation.edge_mask,
            'node_mask': getattr(explanation, 'node_mask', None),
        }

    def explain_batch(self, data_loader, num_samples=100):
        """
        批量解释并统计聚合

        Args:
            data_loader: 测试集DataLoader
            num_samples: 分析样本数

        Returns:
            dict: {
                'edge_importance_mean': [num_edges] 平均边重要性,
                'edge_importance_std': [num_edges] 标准差,
                'important_edges': List[(src, dst, importance)] Top-K重要边,
                'num_samples': int 实际分析的样本数
            }
        """
        edge_masks = []
        sample_count = 0
        ref_edge_index = None  # 参考edge_index（所有样本共享）

        print(f"\n{'='*60}")
        print(f"空间关系分析: 批量解释 (目标样本数={num_samples})")
        print(f"{'='*60}")

        for i, batch in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            try:
                # 获取批次数据
                if isinstance(batch, tuple) or isinstance(batch, list):
                    pyg_data = batch[0]
                else:
                    pyg_data = batch

                # PyG DataLoader会批处理图,导致edge_index变化
                # 我们需要从batch中提取单个样本
                # 由于所有样本共享同一图结构,我们使用原始图的edge_index

                # 保存第一个batch的edge_index作为参考
                if ref_edge_index is None:
                    # 从原始图获取edge_index（不受batch影响）
                    # 使用 get_original_dataset 处理 Subset 情况
                    original_dataset = get_original_dataset(data_loader.dataset)
                    ref_edge_index = original_dataset.graph.edge_index.to(self.device)
                    num_nodes = original_dataset.config.node_num

                # 从batch中提取第一个样本的特征
                # pyg_data.x可能是[batch_size * num_nodes, hist_len, in_dim]
                # 或者是[num_nodes, hist_len, in_dim]（batch_size=1时）
                if pyg_data.x.shape[0] == num_nodes:
                    # 单样本情况
                    sample_x = pyg_data.x
                else:
                    # 多样本批次,提取第一个
                    sample_x = pyg_data.x[:num_nodes]

                # 解释（使用固定的edge_index）
                explanation = self.explain_single(
                    sample_x.to(self.device),
                    ref_edge_index
                )

                edge_masks.append(explanation['edge_mask'].cpu())
                sample_count += 1

                # 进度条
                if sample_count % 10 == 0:
                    print(f"  进度: {sample_count}/{num_samples}")

            except Exception as e:
                print(f"  ⚠ 样本{i}解释失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(edge_masks) == 0:
            raise RuntimeError("没有成功解释任何样本,请检查数据和模型")

        print(f"  ✓ 完成 {len(edge_masks)} 个样本的解释")

        # 统计聚合
        edge_masks = torch.stack(edge_masks)  # [num_samples, num_edges]
        edge_mean = edge_masks.mean(dim=0)
        edge_std = edge_masks.std(dim=0)

        print(f"\n边重要性统计:")
        print(f"  均值: {edge_mean.mean():.4f}")
        print(f"  标准差: {edge_std.mean():.4f}")
        print(f"  最大值: {edge_mean.max():.4f}")

        # 提取Top-K重要边
        top_k = min(self.exp_config.top_k_edges, len(edge_mean))
        # 使用参考edge_index（所有样本共享）
        important_edges = get_top_k_edges(edge_mean, ref_edge_index.cpu(), k=top_k)

        print(f"\nTop-{min(5, top_k)} 重要边:")
        for i, (src, dst, imp) in enumerate(important_edges[:5], 1):
            print(f"  {i}. 站点{src} → 站点{dst}: {imp:.4f}")

        return {
            'edge_importance_mean': edge_mean,
            'edge_importance_std': edge_std,
            'important_edges': important_edges,
            'num_samples': len(edge_masks),
        }

    def _match_attention_to_original_edges(self, attn_edge_index, attn_weights, original_edge_index):
        """
        将GAT返回的注意力权重（可能包含自环边）匹配到原始边索引

        Args:
            attn_edge_index: [2, num_attn_edges] GAT返回的边索引（可能包含自环）
            attn_weights: [num_attn_edges] 对应的注意力权重
            original_edge_index: [2, num_original_edges] 原始边索引

        Returns:
            matched_attention: [num_original_edges] 匹配后的注意力权重
        """
        device = attn_edge_index.device
        num_original_edges = original_edge_index.shape[1]

        # 创建边的唯一标识符：src * max_node_id + dst
        max_node_id = max(attn_edge_index.max().item(), original_edge_index.max().item()) + 1

        # GAT返回的边标识
        attn_edge_ids = attn_edge_index[0] * max_node_id + attn_edge_index[1]

        # 原始边标识
        original_edge_ids = original_edge_index[0] * max_node_id + original_edge_index[1]

        # 匹配：对于每条原始边，在GAT返回的边中找到对应的注意力权重
        matched_attention = torch.zeros(num_original_edges, device=device)

        for i, orig_id in enumerate(original_edge_ids):
            # 在attn_edge_ids中找到匹配的位置
            mask = (attn_edge_ids == orig_id)
            if mask.any():
                # 找到匹配，取该位置的注意力权重
                matched_attention[i] = attn_weights[mask].mean()
            else:
                # 未找到匹配（理论上不应该发生），设为0
                matched_attention[i] = 0.0

        return matched_attention

    def extract_attention_weights_batch(self, data_loader, num_samples=100):
        """
        批量提取GAT注意力权重并聚合

        仅适用于GAT模型。对GSAGE模型会返回None并给出提示。

        Args:
            data_loader: 测试集DataLoader
            num_samples: 分析样本数

        Returns:
            dict: {
                'attention_mean': [num_edges] 多层多样本平均的注意力权重,
                'attention_std': [num_edges] 标准差,
                'num_samples': int 实际分析的样本数
            }
            或 None (如果模型不支持注意力提取)
        """
        # 检查模型类型
        from .gnn_wrapper import GATWrapper, GSAGEWrapper, GATSeparateEncoderWrapper

        if not isinstance(self.gnn_wrapper, (GATWrapper, GATSeparateEncoderWrapper)):
            print("  ⚠ 当前模型不支持注意力权重提取(仅GAT模型支持)")
            return None

        all_attention_layers = []  # 存储每个样本的注意力权重
        sample_count = 0
        ref_edge_index = None

        print(f"\n{'='*60}")
        print(f"GAT注意力权重提取: 批量分析 (目标样本数={num_samples})")
        print(f"{'='*60}")

        for i, batch in enumerate(data_loader):
            if sample_count >= num_samples:
                break

            try:
                # 获取批次数据
                if isinstance(batch, tuple) or isinstance(batch, list):
                    pyg_data = batch[0]
                else:
                    pyg_data = batch

                # 保存第一个batch的edge_index作为参考
                if ref_edge_index is None:
                    from .utils import get_original_dataset
                    original_dataset = get_original_dataset(data_loader.dataset)
                    ref_edge_index = original_dataset.graph.edge_index.to(self.device)
                    num_nodes = original_dataset.config.node_num

                # 从batch中提取单个样本的特征
                if pyg_data.x.shape[0] == num_nodes:
                    sample_x = pyg_data.x
                else:
                    # 多样本批次,提取第一个
                    sample_x = pyg_data.x[:num_nodes]

                # 提取编码器特征（根据模型类型选择方法）
                if isinstance(self.gnn_wrapper, GATSeparateEncoderWrapper):
                    encoder_features = GATSeparateEncoderWrapper.extract_encoder_features(
                        self.full_model, sample_x.to(self.device),
                        ref_edge_index, self.device
                    )
                else:
                    encoder_features = GATWrapper.extract_lstm_features(
                        self.full_model, sample_x.to(self.device),
                        ref_edge_index, self.device
                    )

                # 获取注意力权重
                with torch.no_grad():
                    _, attention_list = self.gnn_wrapper(
                        encoder_features, ref_edge_index, return_attention=True
                    )

                # attention_list: [(edge_index, attn), ...] 每层一个元组
                # GAT可能添加了自环边，需要过滤只保留原始边的注意力权重
                sample_attentions = []
                for attn_edge_index, attn in attention_list:
                    # 找到attn_edge_index中与ref_edge_index匹配的边
                    # 使用torch的高效匹配方法
                    matched_attention = self._match_attention_to_original_edges(
                        attn_edge_index, attn, ref_edge_index
                    )
                    sample_attentions.append(matched_attention.cpu())

                all_attention_layers.append(sample_attentions)
                sample_count += 1

                # 进度条
                if sample_count % 10 == 0:
                    print(f"  进度: {sample_count}/{num_samples}")

            except Exception as e:
                print(f"  ⚠ 样本{i}注意力提取失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(all_attention_layers) == 0:
            print("  ⚠ 没有成功提取任何样本的注意力权重")
            return None

        print(f"  ✓ 完成 {len(all_attention_layers)} 个样本的注意力提取")

        # 聚合: 先沿层维度平均，再沿样本维度平均
        # 1. 每个样本的多层平均: [num_samples, num_edges]
        aggregated_per_sample = []
        for sample_attns in all_attention_layers:
            layer_stacked = torch.stack(sample_attns)  # [num_layers, num_edges]
            sample_avg = layer_stacked.mean(dim=0)     # [num_edges]
            aggregated_per_sample.append(sample_avg)

        # 2. 多样本平均: [num_edges]
        all_samples = torch.stack(aggregated_per_sample)  # [num_samples, num_edges]
        attention_mean = all_samples.mean(dim=0)
        attention_std = all_samples.std(dim=0)

        print(f"\nGAT注意力权重统计:")
        print(f"  均值: {attention_mean.mean():.4f}")
        print(f"  标准差: {attention_std.mean():.4f}")
        print(f"  最大值: {attention_mean.max():.4f}")

        return {
            'attention_mean': attention_mean,
            'attention_std': attention_std,
            'num_samples': len(all_attention_layers)
        }
