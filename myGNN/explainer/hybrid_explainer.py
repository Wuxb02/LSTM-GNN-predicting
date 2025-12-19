"""
混合可解释性分析器

整合时序和空间两种解释方法,提供统一的分析接口

作者: GNN气温预测项目
日期: 2025
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from .temporal_analyzer import TemporalAnalyzer
from .spatial_explainer import SpatialExplainer
from .utils import filter_by_season, detect_model_type, get_original_dataset


class HybridExplainer:
    """
    混合模型统一解释接口

    整合时序分析(Integrated Gradients)和空间分析(GNNExplainer),
    提供完整的可解释性分析管道

    Args:
        model: 完整的LSTM-GNN混合模型
        config: Config配置对象
        explainer_config: ExplainerConfig可解释性配置

    示例:
        >>> explainer = HybridExplainer(model, config, exp_config)
        >>> explanation = explainer.explain_full(
        >>>     test_loader,
        >>>     num_samples=100,
        >>>     save_path='checkpoints/GAT_LSTM/explanations/'
        >>> )
    """

    def __init__(self, model, config, explainer_config):
        self.model = model
        self.config = config
        self.exp_config = explainer_config

        # 检测模型类型
        self.model_type = detect_model_type(model)
        print(f"\n{'='*70}")
        print(f"初始化混合可解释性分析器")
        print(f"{'='*70}")
        print(f"模型类型: {self.model_type}")
        print(f"分析样本数: {explainer_config.num_samples}")
        print(f"季节筛选: {explainer_config.season or '全年'}")
        print(f"GNNExplainer训练轮数: {explainer_config.epochs}")
        print(f"{'='*70}\n")

        # 初始化两个分析器
        self.temporal_analyzer = TemporalAnalyzer(model, config, explainer_config)
        self.spatial_explainer = SpatialExplainer(model, config, explainer_config)

    def explain_full(self, data_loader, num_samples=None, save_path=None):
        """
        完整解释管道

        Args:
            data_loader: 测试集DataLoader
            num_samples: 分析样本数(None=使用配置值)
            save_path: 结果保存路径(None=不保存)

        Returns:
            explanation_dict: {
                'temporal': {
                    'time_importance': [hist_len],
                    'feature_importance': [in_dim],
                    'temporal_heatmap': [hist_len, in_dim],
                    'num_samples': int
                },
                'spatial': {
                    'edge_importance_mean': [num_edges],
                    'edge_importance_std': [num_edges],
                    'important_edges': List[(src, dst, importance)],
                    'num_samples': int
                },
                'metadata': {
                    'model_type': str,
                    'num_samples': int,
                    'season': str,
                    'hist_len': int,
                    'pred_len': int,
                    'in_dim': int,
                    'timestamp': str
                }
            }
        """
        if num_samples is None:
            num_samples = self.exp_config.num_samples

        print(f"\n{'='*70}")
        print(f"GNN气温预测模型 - 完整可解释性分析")
        print(f"{'='*70}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型类型: {self.model_type}")
        print(f"目标样本数: {num_samples}")
        print(f"季节筛选: {self.exp_config.season or '全年'}")
        print(f"{'='*70}\n")

        # 季节筛选(如果需要)
        filtered_loader = data_loader
        if self.exp_config.season is not None:
            print(f"[阶段0/3] 季节筛选({self.exp_config.season})...")
            try:
                filtered_indices = filter_by_season(
                    data_loader.dataset,
                    self.exp_config.season,
                    self.config
                )
                if len(filtered_indices) > 0:
                    # 创建子集数据集
                    from torch.utils.data import Subset, DataLoader
                    filtered_dataset = Subset(data_loader.dataset, filtered_indices)
                    filtered_loader = DataLoader(
                        filtered_dataset,
                        batch_size=data_loader.batch_size,
                        shuffle=False,
                        collate_fn=getattr(data_loader, 'collate_fn', None)
                    )
                    # 注意: Subset 对象无法直接设置 graph/config 属性
                    # 空间分析时会通过 get_original_dataset() 获取原始数据集
                    print(f"  ✓ 筛选完成: {len(filtered_indices)}个样本")
                else:
                    print(f"  ⚠ 未找到符合条件的样本,使用全部数据")
            except Exception as e:
                print(f"  ⚠ 季节筛选失败: {e}")
                print(f"  使用所有样本进行分析")

        # 阶段1: 时序分析
        print(f"\n[阶段1/4] 时序特征重要性分析 (Integrated Gradients)")
        print(f"-" * 60)
        temporal_result = self.temporal_analyzer.analyze_batch(
            filtered_loader, num_samples
        )

        # 阶段2: 空间分析
        print(f"\n[阶段2/4] 空间关系重要性分析 (GNNExplainer)")
        print(f"-" * 60)
        spatial_result = self.spatial_explainer.explain_batch(
            filtered_loader, num_samples
        )

        # 阶段3: GAT注意力权重提取(如果模型支持)
        attention_result = None
        if self.exp_config.extract_attention:
            print(f"\n[阶段3/4] GAT注意力权重提取")
            print(f"-" * 60)
            attention_result = self.spatial_explainer.extract_attention_weights_batch(
                filtered_loader, num_samples
            )

            if attention_result is not None:
                print(f"  ✓ 注意力权重提取成功")
            else:
                print(f"  ⚠ 模型不支持注意力权重提取或提取失败")
        else:
            print(f"\n[跳过] GAT注意力权重提取 (extract_attention=False)")

        # ==================== 阶段4: 节点嵌入提取 (新增) ====================
        print(f"\n[阶段4/4] 提取节点嵌入...")
        node_embeddings = None
        tsne_2d = None

        if hasattr(self.model, 'node_embedding') and self.model.node_embedding is not None:
            from myGNN.explainer.utils import extract_node_embeddings, tsne_reduce_embeddings

            node_embeddings = extract_node_embeddings(self.model)
            print(f"  ✓ 节点嵌入形状: {node_embeddings.shape}")

            # t-SNE降维
            tsne_2d = tsne_reduce_embeddings(
                node_embeddings,
                perplexity=min(10, node_embeddings.shape[0] // 3)
            )
            print(f"  ✓ t-SNE 2D投影: {tsne_2d.shape}")
        else:
            print("  ⚠ 模型不支持节点嵌入，跳过此步骤")

        # 整合结果
        explanation = {
            'temporal': temporal_result,
            'spatial': spatial_result,
            'attention': attention_result,
            'node_embeddings': {
                'embeddings': node_embeddings,
                'tsne_2d': tsne_2d,
                'has_embeddings': node_embeddings is not None
            } if node_embeddings is not None else None,
            'metadata': {
                'model_type': self.model_type,
                'num_samples': num_samples,
                'season': self.exp_config.season,
                'hist_len': self.config.hist_len,
                'pred_len': self.config.pred_len,
                'in_dim': self.config.in_dim,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        }

        # 打印总结
        print(f"\n{'='*70}")
        print(f"可解释性分析完成!")
        print(f"{'='*70}")
        print(f"\n时序分析总结:")
        # 数据索引与时间步对应: 索引0→T-hist_len, 索引hist_len-1→T-1
        hist_len = len(temporal_result['time_importance'])
        max_idx = torch.argmax(temporal_result['time_importance']).item()
        max_time_step = hist_len - max_idx  # 转换为T-X格式
        print(f"  最重要的时间步: T-{max_time_step} (索引{max_idx})")
        print(f"  最重要的特征索引: {torch.argmax(temporal_result['feature_importance']).item()}")
        print(f"\n空间分析总结:")
        print(f"  Top-5 重要边:")
        for i, (src, dst, imp) in enumerate(spatial_result['important_edges'][:5], 1):
            print(f"    {i}. 站点{src} → 站点{dst}: {imp:.4f}")

        # 保存结果
        if save_path:
            # 获取edge_index用于可视化（使用get_original_dataset处理Subset情况）
            original_dataset = get_original_dataset(data_loader.dataset)
            edge_index = original_dataset.graph.edge_index
            self._save_explanation(explanation, save_path, edge_index)
            print(f"\n✓ 解释结果已保存到: {save_path}")

        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        return explanation

    def _save_explanation(self, explanation, save_path, edge_index=None):
        """
        保存解释结果

        Args:
            explanation: 解释结果字典
            save_path: 保存路径
            edge_index: 图的边索引 [2, num_edges],用于可视化
        """
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)

        # 保存为.npz格式
        save_file = save_dir / 'explanation_data.npz'

        # 准备保存数据
        save_data = {
            # 时序
            'time_importance': explanation['temporal']['time_importance'].numpy(),
            'feature_importance': explanation['temporal']['feature_importance'].numpy(),
            'temporal_heatmap': explanation['temporal']['temporal_heatmap'].numpy(),
            # 空间
            'edge_importance_mean': explanation['spatial']['edge_importance_mean'].numpy(),
            'edge_importance_std': explanation['spatial']['edge_importance_std'].numpy(),
            # 元数据(作为字符串保存)
            'model_type': explanation['metadata']['model_type'],
            'season': str(explanation['metadata']['season']),
            'num_samples': explanation['metadata']['num_samples'],
            'hist_len': explanation['metadata']['hist_len'],
            'pred_len': explanation['metadata']['pred_len'],
            'in_dim': explanation['metadata']['in_dim'],
            'timestamp': explanation['metadata']['timestamp'],
        }

        # 添加注意力权重(如果存在)
        if explanation['attention'] is not None:
            save_data['attention_mean'] = explanation['attention']['attention_mean'].numpy()
            save_data['attention_std'] = explanation['attention']['attention_std'].numpy()

        # 添加edge_index用于空间可视化
        if edge_index is not None:
            if hasattr(edge_index, 'cpu'):
                save_data['edge_index'] = edge_index.cpu().numpy()
            elif hasattr(edge_index, 'numpy'):
                save_data['edge_index'] = edge_index.numpy()
            else:
                save_data['edge_index'] = edge_index

        # 添加节点嵌入数据(如果存在)
        if explanation['node_embeddings'] is not None:
            save_data['node_embeddings'] = explanation['node_embeddings']['embeddings']
            save_data['tsne_2d'] = explanation['node_embeddings']['tsne_2d']
            save_data['has_node_embeddings'] = True
        else:
            save_data['has_node_embeddings'] = False

        np.savez(save_file, **save_data)

        # 保存重要边列表为单独的txt文件
        edges_file = save_dir / 'important_edges.txt'
        with open(edges_file, 'w', encoding='utf-8') as f:
            f.write(f"Top-{len(explanation['spatial']['important_edges'])} 重要边\n")
            f.write(f"{'='*50}\n\n")
            for i, (src, dst, imp) in enumerate(explanation['spatial']['important_edges'], 1):
                f.write(f"{i:3d}. 站点{src:3d} → 站点{dst:3d}: {imp:.6f}\n")

        print(f"  ✓ 数据文件: {save_file}")
        print(f"  ✓ 重要边列表: {edges_file}")
