"""
多模型性能对比脚本

功能:
1. 读取checkpoints目录下所有模型的训练结果
2. 解析metrics.txt文件提取评估指标
3. 生成对比CSV文件
4. (可选) 生成对比可视化图表

使用方法:
    python myGNN/experiments/compare_models.py

输出:
    - model_comparison.csv: 对比结果表格
    - model_comparison.png: (可选) 对比柱状图

作者: GNN气温预测项目
日期: 2025-12-20
"""

import os
import re
from pathlib import Path
import pandas as pd


def parse_metrics_file(metrics_path):
    """
    解析metrics.txt文件，提取评估指标

    Args:
        metrics_path: metrics.txt文件路径

    Returns:
        metrics_dict: 包含RMSE, MAE, R², Bias的字典
                     分为train/val/test三个部分
    """
    with open(metrics_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'train': {}, 'val': {}, 'test': {}
    }

    # 正则表达式匹配指标
    # 格式: "RMSE: 1.2345 °C"
    rmse_pattern = r'RMSE:\s+([\d.]+)\s+°C'
    mae_pattern = r'MAE:\s+([\d.]+)\s+°C'
    r2_pattern = r'R²:\s+([\d.]+)'
    bias_pattern = r'Bias:\s+([+-]?[\d.]+)\s+°C'

    # 按数据集分割内容
    sections = {
        'train': re.search(r'训练集:.*?(?=\n验证集:|\n测试集:|$)', content, re.DOTALL),
        'val': re.search(r'验证集:.*?(?=\n测试集:|$)', content, re.DOTALL),
        'test': re.search(r'测试集:.*?(?=$)', content, re.DOTALL)
    }

    for dataset, match in sections.items():
        if match:
            section_text = match.group(0)

            # 提取各项指标
            rmse_match = re.search(rmse_pattern, section_text)
            mae_match = re.search(mae_pattern, section_text)
            r2_match = re.search(r2_pattern, section_text)
            bias_match = re.search(bias_pattern, section_text)

            if rmse_match:
                metrics[dataset]['rmse'] = float(rmse_match.group(1))
            if mae_match:
                metrics[dataset]['mae'] = float(mae_match.group(1))
            if r2_match:
                metrics[dataset]['r2'] = float(r2_match.group(1))
            if bias_match:
                metrics[dataset]['bias'] = float(bias_match.group(1))

    return metrics


def extract_model_name(checkpoint_dir):
    """
    从checkpoint目录名提取模型名称

    Args:
        checkpoint_dir: checkpoint目录名 (如 'GAT_LSTM_20251220_153042')

    Returns:
        model_name: 模型名称 (如 'GAT_LSTM')
    """
    # 移除时间戳部分
    # 格式: {模型名}_{时间戳}
    parts = checkpoint_dir.split('_')

    # 找到时间戳的起始位置（8位数字）
    timestamp_idx = None
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            timestamp_idx = i
            break

    if timestamp_idx is not None:
        # 返回时间戳之前的所有部分
        return '_'.join(parts[:timestamp_idx])
    else:
        # 如果没有时间戳，返回完整目录名
        return checkpoint_dir


def collect_all_results(checkpoints_dir='checkpoints'):
    """
    收集所有模型的训练结果

    Args:
        checkpoints_dir: checkpoints目录路径

    Returns:
        results: List[Dict] 包含所有模型结果的列表
    """
    checkpoints_path = Path(__file__).parent.parent / checkpoints_dir

    if not checkpoints_path.exists():
        print(f"错误: checkpoints目录不存在: {checkpoints_path}")
        return []

    results = []

    # 遍历checkpoints目录
    for checkpoint_dir in sorted(checkpoints_path.iterdir()):
        if not checkpoint_dir.is_dir():
            continue

        metrics_file = checkpoint_dir / 'metrics.txt'

        if not metrics_file.exists():
            print(f"警告: {checkpoint_dir.name} 中未找到 metrics.txt，跳过")
            continue

        # 解析指标
        try:
            metrics = parse_metrics_file(metrics_file)

            # 提取模型名称
            model_name = extract_model_name(checkpoint_dir.name)

            # 构建结果字典
            result = {
                'Model': model_name,
                'Checkpoint': checkpoint_dir.name,
                # 测试集指标
                'Test_RMSE': metrics['test'].get('rmse', None),
                'Test_MAE': metrics['test'].get('mae', None),
                'Test_R2': metrics['test'].get('r2', None),
                'Test_Bias': metrics['test'].get('bias', None),
                # 验证集指标
                'Val_RMSE': metrics['val'].get('rmse', None),
                'Val_MAE': metrics['val'].get('mae', None),
                'Val_R2': metrics['val'].get('r2', None),
                'Val_Bias': metrics['val'].get('bias', None),
            }

            results.append(result)
            print(f"✓ 读取: {checkpoint_dir.name}")

        except Exception as e:
            print(f"错误: 解析 {checkpoint_dir.name} 失败: {e}")
            continue

    return results


def generate_comparison_table(results, save_path='model_comparison.csv'):
    """
    生成对比表格并保存

    Args:
        results: List[Dict] 模型结果列表
        save_path: 保存路径

    Returns:
        df: pandas DataFrame
    """
    if not results:
        print("错误: 没有可用的结果数据")
        return None

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 按测试集RMSE排序
    df = df.sort_values('Test_RMSE')

    # 保存CSV
    save_path = Path(__file__).parent / save_path
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"\n✓ 对比结果已保存到: {save_path}")

    return df


def print_comparison_summary(df):
    """
    打印对比结果摘要

    Args:
        df: pandas DataFrame
    """
    print("\n" + "=" * 80)
    print("模型性能对比 (按测试集RMSE排序)")
    print("=" * 80)

    # 选择关键列显示
    display_cols = ['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_Bias']
    summary_df = df[display_cols].copy()

    # 格式化显示
    summary_df['Test_RMSE'] = summary_df['Test_RMSE'].map(lambda x: f"{x:.4f}")
    summary_df['Test_MAE'] = summary_df['Test_MAE'].map(lambda x: f"{x:.4f}")
    summary_df['Test_R2'] = summary_df['Test_R2'].map(lambda x: f"{x:.4f}")
    summary_df['Test_Bias'] = summary_df['Test_Bias'].map(lambda x: f"{x:+.4f}")

    print(summary_df.to_string(index=False))
    print("=" * 80)

    # 打印最佳模型
    best_model = df.iloc[0]
    print(f"\n🏆 最佳模型 (RMSE最低): {best_model['Model']}")
    print(f"   测试集RMSE: {best_model['Test_RMSE']:.4f} °C")
    print(f"   测试集MAE:  {best_model['Test_MAE']:.4f} °C")
    print(f"   测试集R²:   {best_model['Test_R2']:.4f}")
    print(f"   测试集Bias: {best_model['Test_Bias']:+.4f} °C")


def plot_comparison(df, save_path='model_comparison.png'):
    """
    绘制对比柱状图

    Args:
        df: pandas DataFrame
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # RMSE对比
        ax1 = axes[0, 0]
        df.plot(x='Model', y='Test_RMSE', kind='bar', ax=ax1, color='steelblue', legend=False)
        ax1.set_title('测试集RMSE对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE (°C)', fontsize=12)
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # MAE对比
        ax2 = axes[0, 1]
        df.plot(x='Model', y='Test_MAE', kind='bar', ax=ax2, color='coral', legend=False)
        ax2.set_title('测试集MAE对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE (°C)', fontsize=12)
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)

        # R²对比
        ax3 = axes[1, 0]
        df.plot(x='Model', y='Test_R2', kind='bar', ax=ax3, color='mediumseagreen', legend=False)
        ax3.set_title('测试集R²对比', fontsize=14, fontweight='bold')
        ax3.set_ylabel('R² (决定系数)', fontsize=12)
        ax3.set_xlabel('')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)

        # Bias对比
        ax4 = axes[1, 1]
        df.plot(x='Model', y='Test_Bias', kind='bar', ax=ax4, color='mediumpurple', legend=False)
        ax4.set_title('测试集Bias对比', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Bias (°C)', fontsize=12)
        ax4.set_xlabel('')
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # 保存图表
        save_path = Path(__file__).parent / save_path
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化图表已保存到: {save_path}")

    except ImportError:
        print("警告: matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"警告: 生成可视化图表失败: {e}")


def main():
    """主函数"""
    print("=" * 80)
    print("多模型性能对比分析")
    print("=" * 80)

    # 1. 收集所有结果
    print("\n[1/3] 收集训练结果...")
    results = collect_all_results()

    if not results:
        print("\n错误: 未找到任何训练结果")
        print("请先运行以下命令训练模型:")
        print("  - python myGNN/train.py  (训练GNN模型)")
        print("  - python myGNN/baselines/train_xgboost.py  (训练XGBoost模型)")
        return

    print(f"\n✓ 成功收集 {len(results)} 个模型结果")

    # 2. 生成对比表格
    print("\n[2/3] 生成对比表格...")
    df = generate_comparison_table(results)

    if df is not None:
        print_comparison_summary(df)

    # 3. 生成可视化图表
    print("\n[3/3] 生成可视化图表...")
    plot_comparison(df)

    print("\n" + "=" * 80)
    print("对比分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
