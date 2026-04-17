"""
高温日 vs 普通日 交叉注意力权重对比分析

分组逻辑：
- 每个分析单元 = (时间步, 站点) 对
- 以 data.y[:, 0] 反归一化后的真实 tmax 与站点-日内动态阈值比较
  - 高温样本: tmax >= threshold_map[doy, station]
  - 普通样本: tmax <  threshold_map[doy, station]
- 动态阈值 = 训练集 ±7 天窗口的 90 分位数（365×28 表格）
- 回答: "当真实高温发生时，模型注意力分配有何不同？"

输出: figdraw/result/cross_attn_high_temp_analysis.png

前置要求: 已使用 10 个静态特征重新训练的 GAT_SeparateEncoder checkpoint
"""

import os
import sys
import glob

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
import torch

# ──────────────── 路径设置 ────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from myGNN.models.GAT_SeparateEncoder import GAT_SeparateEncoder
from myGNN.dataset import create_dataloaders

# ──────────────── 字体设置 ────────────────
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ──────────────── 静态特征名称 ────────────────
# 对应 static_feature_indices = [0, 1, 2, 10, 11, 12, 16, 17, 18, 25]
STATIC_FEATURE_LABELS = [
    'x\n(经度)',
    'y\n(纬度)',
    'height\n(海拔)',
    'BH\n(建筑高度)',
    'BHstd\n(高度偏差)',
    'SCD\n(建筑拥挤度)',
    'POI\n(不透水面比)',
    'POW\n(水体面比)',
    'POV\n(植被覆盖比)',
    'VegH\n(植被高度)',
]

STATIC_FEATURE_SHORT = [
    'x', 'y', 'height', 'BH', 'BHstd', 'SCD', 'POI', 'POW', 'POV', 'VegH'
]

# ──────────────── 颜色方案 ────────────────
COLOR_HIGH = '#D62728'    # 红色：高温日
COLOR_NORMAL = '#1F77B4'  # 蓝色：普通日
COLOR_POS = '#D62728'     # 差异图正值（高温更关注）
COLOR_NEG = '#1F77B4'     # 差异图负值（普通日更关注）

# 不展示前 N 个纯地理特征（x, y, height），只展示城市/植被特征
SKIP_GEO_FEATURES = 3


def find_latest_checkpoint(model_name='GAT_SeparateEncoder'):
    """自动搜索最新的 checkpoint 目录"""
    pattern = os.path.join(
        PROJECT_ROOT, 'myGNN', 'checkpoints max', f'{model_name}', 'best_model.pth'
    )
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"未找到 checkpoint: {pattern}\n"
            "请先训练模型（使用 10 个静态特征后重新训练）"
        )
    latest = candidates[-1]
    print(f"✓ 使用 checkpoint: {latest}")
    return latest


def load_model_and_config(checkpoint_path):
    """加载模型权重与配置"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['config']
    arch_config = checkpoint['arch_config']
    graph = checkpoint['graph']

    model = GAT_SeparateEncoder(config, arch_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    num_static = config.static_encoded_dim
    print(f"✓ 模型加载完成")
    print(f"  静态特征数: {num_static}")
    print(f"  fusion_num_heads: {arch_config.fusion_num_heads}")
    print(f"  预测目标: 特征索引 {config.target_feature_idx} (tmax)")

    return model, config, arch_config, graph


def collect_attention_weights(model, config, graph, device, split='test'):
    """
    遍历数据集，收集所有样本的注意力权重与温度标签。

    Returns:
        all_attn: np.ndarray [N_total, num_static]
        all_tmax: np.ndarray [N_total]  (反归一化后的真实 tmax, °C)
        all_is_high: np.ndarray [N_total] bool
    """
    # 以 batch_size=1 加载，方便逐时间步处理
    config.batch_size = 1
    train_loader, val_loader, test_loader, stats = create_dataloaders(config, graph)

    ta_mean = stats['ta_mean']
    ta_std = stats['ta_std']
    threshold_map = stats.get('threshold_map', None)

    if threshold_map is None:
        raise RuntimeError(
            "stats 中缺少 threshold_map，请确认 config.loss_config."
            "use_station_day_threshold = True"
        )

    print(f"\n✓ 标准化参数: ta_mean={ta_mean:.4f}, ta_std={ta_std:.4f}")
    print(f"✓ 动态阈值表形状: {threshold_map.shape}")
    print(f"✓ 阈值范围: [{threshold_map.min():.2f}, {threshold_map.max():.2f}] °C")

    if split == 'test':
        loader = test_loader
    elif split == 'val':
        loader = val_loader
    elif split == 'all':
        # 合并全部数据集以增加样本量
        from itertools import chain
        loader = chain(train_loader, val_loader, test_loader)
    else:
        loader = test_loader

    all_attn = []
    all_tmax = []
    all_is_high = []

    model = model.to(device)
    model.eval()

    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            # 解包 batch
            if len(batch) == 3:
                data, time_indices, doy_indices = batch
            elif len(batch) == 2:
                data, time_indices = batch
                doy_indices = None
            else:
                continue

            x = data.x.to(device)           # [28, hist_len, in_dim]
            y = data.y.to(device)            # [28, pred_len]
            edge_index = data.edge_index.to(device)

            # 前向传播，获取注意力权重
            _, attn_weights = model(
                x, edge_index, return_cross_attention=True
            )
            # attn_weights: [28, num_heads, num_static]
            # 对注意力头取平均 → [28, num_static]
            attn_avg = attn_weights.mean(dim=1).cpu().numpy()

            # 反归一化第 0 预测步的真实 tmax
            tmax_actual = y[:, 0].cpu().numpy() * ta_std + ta_mean  # [28]

            # 获取该时间步对应的 0-based doy
            if doy_indices is not None:
                doy_0 = int(doy_indices[0, 0].item())
            else:
                # 兜底：从时间索引推算
                time_idx = int(time_indices[0].item())
                # 简单推算（实际 doy 可能有闰年偏差，但影响极小）
                doy_0 = time_idx % 365

            # 确保 doy_0 在合法范围内
            doy_0 = min(doy_0, threshold_map.shape[0] - 1)

            # 动态阈值：每个站点独立阈值 [28]
            threshold = threshold_map[doy_0, :]  # [28]
            is_high = tmax_actual >= threshold    # [28] bool

            all_attn.append(attn_avg)
            all_tmax.append(tmax_actual)
            all_is_high.append(is_high)
            n_batches += 1

    print(f"\n✓ 处理批次数: {n_batches}")

    all_attn = np.vstack(all_attn)              # [N_total, num_static]
    all_tmax = np.concatenate(all_tmax)          # [N_total]
    all_is_high = np.concatenate(all_is_high)    # [N_total] bool

    return all_attn, all_tmax, all_is_high, ta_mean, ta_std


def compute_stats(all_attn, all_is_high):
    """计算两组的统计量"""
    high_attn = all_attn[all_is_high]
    normal_attn = all_attn[~all_is_high]

    n_high = len(high_attn)
    n_normal = len(normal_attn)
    num_features = all_attn.shape[1]

    high_mean = high_attn.mean(axis=0)
    normal_mean = normal_attn.mean(axis=0)
    high_se = high_attn.std(axis=0) / np.sqrt(n_high)
    normal_se = normal_attn.std(axis=0) / np.sqrt(n_normal)
    delta = high_mean - normal_mean

    # 独立样本 t-test
    p_values = np.zeros(num_features)
    for i in range(num_features):
        _, p = scipy_stats.ttest_ind(high_attn[:, i], normal_attn[:, i])
        p_values[i] = p

    return {
        'n_high': n_high,
        'n_normal': n_normal,
        'high_mean': high_mean,
        'normal_mean': normal_mean,
        'high_se': high_se,
        'normal_se': normal_se,
        'delta': delta,
        'p_values': p_values,
    }


def significance_label(p):
    """将 p 值转换为显著性标注"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


def plot_analysis(result, num_static, output_path):
    """
    绘制两行布局的对比图：
    - 上图: 分组柱状图（高温 vs 普通，带误差线）
    - 下图: 差异图（Δ = high - normal，水平条形图，按 |Δ| 排序）
    """
    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(12, num_static * 1.2), 11),
        gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.45}
    )

    n_high = result['n_high']
    n_normal = result['n_normal']
    high_mean = result['high_mean']
    normal_mean = result['normal_mean']
    high_se = result['high_se']
    normal_se = result['normal_se']
    delta = result['delta']
    p_values = result['p_values']

    labels = STATIC_FEATURE_LABELS[SKIP_GEO_FEATURES:SKIP_GEO_FEATURES + num_static]
    short_labels = STATIC_FEATURE_SHORT[SKIP_GEO_FEATURES:SKIP_GEO_FEATURES + num_static]
    x = np.arange(num_static)
    width = 0.36

    # ──────────── 上图：分组柱状图 ────────────
    ax1 = axes[0]
    bars_normal = ax1.bar(
        x - width / 2, normal_mean,
        width=width, color=COLOR_NORMAL, alpha=0.85,
        label=f'普通日 (N={n_normal:,})', zorder=3
    )
    bars_high = ax1.bar(
        x + width / 2, high_mean,
        width=width, color=COLOR_HIGH, alpha=0.85,
        label=f'高温日 (N={n_high:,})', zorder=3
    )
    ax1.errorbar(
        x - width / 2, normal_mean, yerr=normal_se,
        fmt='none', color='#333333', capsize=3, linewidth=1.2, zorder=4
    )
    ax1.errorbar(
        x + width / 2, high_mean, yerr=high_se,
        fmt='none', color='#333333', capsize=3, linewidth=1.2, zorder=4
    )

    # 显著性标注（标注在高温柱顶部）
    for i in range(num_static):
        sig = significance_label(p_values[i])
        if sig:
            y_pos = high_mean[i] + high_se[i] + 0.002
            ax1.text(
                x[i] + width / 2, y_pos, sig,
                ha='center', va='bottom', fontsize=12, color='#333333'
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('平均注意力权重', fontsize=11)
    ax1.set_title(
        '高温日 vs 普通日 — 静态特征注意力权重对比\n'
        f'（动态阈值：站点-日内90分位数，测试集 {n_high + n_normal:,} 个样本）',
        fontsize=12, fontweight='bold', pad=10
    )
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='x', which='both', length=0)

    # 注意力权重之和提示
    attn_sum_check = (high_mean.sum() + normal_mean.sum()) / 2
    ax1.text(
        0.01, 0.97,
        f'注意力权重均值之和≈{high_mean.sum():.3f} (高温) / {normal_mean.sum():.3f} (普通)',
        transform=ax1.transAxes,
        fontsize=8, color='#555555', va='top'
    )

    # ──────────── 下图：差异图 ────────────
    ax2 = axes[1]

    # 按 |Δ| 降序排列
    sort_idx = np.argsort(np.abs(delta))[::-1]
    delta_sorted = delta[sort_idx]
    labels_sorted = [short_labels[i] for i in sort_idx]
    p_sorted = p_values[sort_idx]

    y_pos = np.arange(num_static)
    colors = [COLOR_POS if d > 0 else COLOR_NEG for d in delta_sorted]
    bars = ax2.barh(y_pos, delta_sorted, color=colors, alpha=0.85, height=0.6)

    # 显著性标注
    for i, (d, p) in enumerate(zip(delta_sorted, p_sorted)):
        sig = significance_label(p)
        offset = 0.0005 if d >= 0 else -0.0005
        ha = 'left' if d >= 0 else 'right'
        if sig:
            ax2.text(d + offset, i, f' {sig}', va='center', ha=ha, fontsize=10)

    ax2.axvline(0, color='#333333', linewidth=1.2, zorder=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels_sorted, fontsize=10)
    ax2.set_xlabel('Δ 注意力权重 = 高温日均值 − 普通日均值', fontsize=11)
    ax2.set_title('静态特征注意力权重差异（按 |Δ| 降序）', fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.35)
    ax2.set_axisbelow(True)

    # 图例：正/负说明
    patch_pos = mpatches.Patch(color=COLOR_POS, alpha=0.85, label='高温日更关注 (Δ > 0)')
    patch_neg = mpatches.Patch(color=COLOR_NEG, alpha=0.85, label='普通日更关注 (Δ < 0)')
    ax2.legend(handles=[patch_pos, patch_neg], fontsize=9, loc='lower right')

    # 显著性说明
    fig.text(
        0.99, 0.01,
        '显著性: * p<0.05  ** p<0.01  *** p<0.001  (独立样本 t 检验)',
        ha='right', va='bottom', fontsize=8, color='#666666'
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存: {output_path}")
    plt.close(fig)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1. 加载模型
    checkpoint_path = find_latest_checkpoint('GAT_SeparateEncoder_WEIGHTED')
    model, config, arch_config, graph = load_model_and_config(checkpoint_path)

    num_static = config.static_encoded_dim
    if num_static != len(STATIC_FEATURE_LABELS):
        print(
            f"警告: 模型静态特征数 ({num_static}) 与标签列表长度 "
            f"({len(STATIC_FEATURE_LABELS)}) 不一致，自动截取。"
        )

    # 2. 收集注意力权重（使用测试集）
    print("\n" + "=" * 60)
    print("收集注意力权重（测试集）...")
    all_attn, all_tmax, all_is_high, ta_mean, ta_std = collect_attention_weights(
        model, config, graph, device, split='test'
    )

    n_total = len(all_tmax)
    n_high = all_is_high.sum()
    n_normal = n_total - n_high
    print(f"\n样本统计:")
    print(f"  总样本数: {n_total:,}")
    print(f"  高温样本: {n_high:,} ({100 * n_high / n_total:.1f}%)")
    print(f"  普通样本: {n_normal:,} ({100 * n_normal / n_total:.1f}%)")
    print(f"  tmax 范围: [{all_tmax.min():.1f}, {all_tmax.max():.1f}] °C")

    # 验证：注意力权重求和
    attn_sums = all_attn.sum(axis=1)
    print(f"\n注意力权重验证:")
    print(f"  每行之和 — mean: {attn_sums.mean():.4f}, std: {attn_sums.std():.6f}")
    if abs(attn_sums.mean() - 1.0) > 0.01:
        print("  警告: 注意力权重之和偏离 1.0，请检查模型输出。")

    # 3. 计算统计量（仅使用需要展示的特征，跳过 x/y/height）
    display_attn = all_attn[:, SKIP_GEO_FEATURES:num_static]
    result = compute_stats(display_attn, all_is_high)
    result['n_high'] = int(n_high)
    result['n_normal'] = int(n_normal)

    print(f"\n特征注意力对比 (高温均值 vs 普通均值):")
    feat_labels = STATIC_FEATURE_SHORT[SKIP_GEO_FEATURES:num_static]
    for i, label in enumerate(feat_labels):
        sig = significance_label(result['p_values'][i])
        print(
            f"  {label:<10} 高温: {result['high_mean'][i]:.4f}  "
            f"普通: {result['normal_mean'][i]:.4f}  "
            f"Δ: {result['delta'][i]:+.4f}  {sig}"
        )

    # 4. 绘图
    output_path = os.path.join(
        SCRIPT_DIR, 'result', 'cross_attn_high_temp_analysis.png'
    )
    print(f"\n绘制图表...")
    plot_analysis(result, num_static - SKIP_GEO_FEATURES, output_path)
    print("完成！")


if __name__ == '__main__':
    main()
