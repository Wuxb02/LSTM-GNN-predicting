"""
时序特征可解释性分析器

使用Integrated Gradients分析LSTM输入特征的重要性,
解释哪些历史时刻和哪些气象要素对预测最重要

作者: GNN气温预测项目
日期: 2025
"""

import torch
import numpy as np
from captum.attr import IntegratedGradients


class TemporalAnalyzer:
    """
    LSTM时序特征重要性分析器

    使用Integrated Gradients方法量化每个历史时间步
    和每个气象要素对预测的贡献

    Args:
        model: 完整的LSTM-GNN混合模型
        config: Config配置对象
        exp_config: ExplainerConfig可解释性配置对象（可选）

    示例:
        >>> analyzer = TemporalAnalyzer(model, config, exp_config)
        >>> result = analyzer.analyze_batch(test_loader, num_samples=100)
        >>> print("最重要的时间步:", torch.argmax(result['time_importance']))
        >>> print("最重要的特征:", torch.argmax(result['feature_importance']))
    """

    def __init__(self, model, config, exp_config=None):
        self.model = model
        self.config = config
        self.exp_config = exp_config
        self.device = config.device
        self.model.eval()

        # 从 ExplainerConfig 获取 ig_steps，如果没有则使用默认值 50
        self.ig_steps = exp_config.ig_steps if exp_config else 50

        # 初始化IntegratedGradients
        self.ig = IntegratedGradients(self._forward_func)

        print(f"✓ 初始化时序分析器 (Integrated Gradients, steps={self.ig_steps})")

    def _forward_func(self, x, edge_index):
        """
        Captum需要的前向函数包装

        Args:
            x: [num_nodes, hist_len, in_dim]
            edge_index: [2, num_edges] (frozen,不参与梯度计算)

        Returns:
            output: [num_nodes] - 对所有预测步长求平均,得到单一输出
        """
        # 模型原始输出: [num_nodes, pred_len]
        output = self.model(x, edge_index)

        # 对pred_len维度求平均,得到 [num_nodes]
        # 这样每个节点有一个标量输出,可以直接计算梯度
        return output.mean(dim=1)

    def analyze_single(self, input_data, edge_index, baseline=None, n_steps=None):
        """
        分析单个样本的时序特征重要性

        Args:
            input_data: [num_nodes, hist_len, in_dim]
            edge_index: [2, num_edges]
            baseline: 基线输入(默认全零)
            n_steps: 积分步数(默认使用配置值)

        Returns:
            attributions: [num_nodes, hist_len, in_dim] 特征归因得分
        """
        if baseline is None:
            baseline = torch.zeros_like(input_data)

        if n_steps is None:
            n_steps = self.ig_steps

        # 确保edge_index不需要梯度
        edge_index = edge_index.detach()

        # 临时切换到train模式以支持cuDNN LSTM的反向传播
        # PyTorch的cuDNN优化LSTM在eval模式下无法计算梯度
        was_training = self.model.training
        self.model.train()

        try:
            # 计算Integrated Gradients
            attributions = self.ig.attribute(
                inputs=input_data,
                baselines=baseline,
                additional_forward_args=(edge_index,),
                n_steps=n_steps,
                internal_batch_size=1,
            )
        finally:
            # 恢复原来的模式
            if not was_training:
                self.model.eval()

        return attributions

    def analyze_batch(self, data_loader, num_samples=100):
        """
        批量分析并聚合

        Args:
            data_loader: 测试集DataLoader
            num_samples: 分析样本数

        Returns:
            dict: {
                'time_importance': [hist_len] 每个时间步的平均重要性,
                'feature_importance': [in_dim] 每个特征的平均重要性,
                'temporal_heatmap': [hist_len, in_dim] 时空热图,
                'num_samples': int 实际分析的样本数
            }
        """
        all_attributions = []
        sample_count = 0

        print(f"\n{'='*60}")
        print(f"时序特征分析: 批量分析 (目标样本数={num_samples})")
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

                # 分析单个样本
                attr = self.analyze_single(
                    pyg_data.x.to(self.device),
                    pyg_data.edge_index.to(self.device)
                )

                # 取绝对值(重要性不考虑方向)
                # attr shape: [num_nodes, hist_len, in_dim]
                # 对节点维度求平均,得到 [hist_len, in_dim]
                attr_avg = attr.abs().mean(dim=0).cpu()  # 平均所有节点
                all_attributions.append(attr_avg)
                sample_count += 1

                # 进度条
                if sample_count % 10 == 0:
                    print(f"  进度: {sample_count}/{num_samples}")

            except Exception as e:
                print(f"  ⚠ 样本{i}分析失败: {e}")
                continue

        if len(all_attributions) == 0:
            raise RuntimeError("没有成功分析任何样本,请检查数据和模型")

        print(f"  ✓ 完成 {len(all_attributions)} 个样本的分析")

        # 聚合统计
        # 现在每个样本已经是 [hist_len, in_dim] (已对节点求平均)
        all_attr = torch.stack(all_attributions)  # [num_samples, hist_len, in_dim]

        # 按时间步聚合 (平均所有样本,对特征维度求和)
        time_importance = all_attr.mean(dim=0).sum(dim=1)  # [hist_len]

        # 按特征聚合 (平均所有样本,对时间维度求和)
        feature_importance = all_attr.mean(dim=0).sum(dim=0)  # [in_dim]

        # 时空热图 (平均所有样本)
        temporal_heatmap = all_attr.mean(dim=0)  # [hist_len, in_dim]

        # 归一化到[0, 1]
        time_importance = time_importance / time_importance.sum()
        feature_importance = feature_importance / feature_importance.sum()

        # 数据索引与时间步的对应关系:
        # - time_importance[0] → T-hist_len (最早的历史时间步)
        # - time_importance[hist_len-1] → T-1 (最近的历史时间步)
        hist_len = len(time_importance)
        max_idx = torch.argmax(time_importance).item()
        # 索引0对应T-hist_len, 索引hist_len-1对应T-1
        # 公式: T-(hist_len - idx)
        max_time_step = hist_len - max_idx

        print(f"\n时序特征分析结果:")
        print(f"  最重要的时间步: T-{max_time_step} (索引{max_idx})")
        print(f"  最重要的特征索引: {torch.argmax(feature_importance).item()}")

        # 打印时间步重要性分布
        # 按T-hist_len, T-(hist_len-1), ..., T-1的顺序打印
        print(f"\n时间步重要性 (归一化):")
        print(f"  [最早] T-{hist_len}: {time_importance[0]:.4f}")
        if hist_len > 2:
            print(f"  ...")
        print(f"  [最近] T-1: {time_importance[-1]:.4f}")

        return {
            'time_importance': time_importance,
            'feature_importance': feature_importance,
            'temporal_heatmap': temporal_heatmap,
            'num_samples': len(all_attributions),
        }
