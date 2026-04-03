# GNN可解释性分析使用指南

本指南提供GNN可解释性分析模块的详细使用方法,包括基础功能和最新的GAT注意力深度分析功能。

---

## 📌 版本更新

### v2.1.0 (2025-12-16) ⭐最新

本次更新新增GAT注意力深度分析功能,验证模型学习的空间依赖正确性:

**新增功能:**
- ✨ **全局注意力矩阵可视化**: 28×28热力图,展示所有站点对的注意力强度
- ✨ **距离-注意力关系分析**: 使用Haversine公式计算地理距离,验证物理规律
- ✨ **温度相关性-注意力关系分析**: 基于训练集温度数据,验证气象模式
- ✨ **新增工具函数**: `haversine_distance()`, `compute_edge_distances()`, `compute_temperature_correlation()`, `extract_edge_correlations()`, `edge_attention_to_matrix()`
- ✨ **3种新可视化图表**: 注意力矩阵热力图、距离-注意力散点图、相关性-注意力散点图
- 📊 **可视化数量**: 从8种增加到11种

**技术改进:**
- 使用Haversine公式精确计算球面距离
- 基于训练集(2010-2017)计算温度相关性,避免数据泄露
- 线性回归+统计检验(Pearson r, p-value, R²)
- 边级注意力→节点级矩阵转换(支持mean/max/sum聚合)

### v2.0.0 (2025-12-01)

**新增功能:**
- ✨ 全边可视化(overlay/separate模式)
- ✨ GAT注意力权重提取与聚合
- ✨ GNNExplainer vs GAT注意力对比图
- 🔧 配置参数: `extract_attention`, `all_edges_mode`

### v1.0.0 (初始版本)

- 时序分析(Integrated Gradients)
- 空间分析(GNNExplainer)
- 季节筛选功能
- 5种基础可视化图表

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install captum>=0.6.0 cartopy>=0.21.0 scipy>=1.7.0
```

或使用requirements.txt:
```bash
pip install -r myGNN/requirements.txt
```

**依赖说明:**
- `captum`: Integrated Gradients时序分析
- `cartopy`: Mapbox WMTS地理底图
- `scipy`: 统计检验(Pearson相关、线性回归) ⭐新增v2.1.0

### 2. 基础使用

```python
from myGNN.explainer import HybridExplainer, ExplainerConfig
from myGNN.config import create_config
from myGNN.dataset import create_dataloaders
from myGNN.graph.distance_graph import create_graph_from_config
import numpy as np
import torch

# 1. 加载配置和数据
config, arch_config, loss_config = create_config()
MetData = np.load(config.MetData_fp)
station_info = np.load(config.station_info_fp)

# 2. 构建图结构
graph = create_graph_from_config(config, station_info)

# 3. 创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    config, graph, MetData,
    batch_size=config.batch_size,
    shuffle_train=True
)

# 4. 加载训练好的模型
model = torch.load('checkpoints/GAT_LSTM_best/best_model.pth')
model.eval()

# 5. 配置解释器(使用所有新功能)
exp_config = ExplainerConfig(
    num_samples=100,           # 分析100个样本
    epochs=200,                # GNNExplainer训练轮数
    season='summer',           # 季节筛选(可选)
    extract_attention=True,    # 提取GAT注意力权重
    all_edges_mode='both'      # 全边可视化模式
)

# 6. 运行完整分析
explainer = HybridExplainer(model, config, exp_config)
explanation = explainer.explain_full(
    test_loader,
    save_path='checkpoints/GAT_LSTM_best/explanations/summer/'
)

# 7. 访问结果
print("=== 时序分析结果 ===")
print(f"最重要的时间步: {torch.argmax(explanation['temporal']['time_importance']).item()}")
print(f"最重要的特征: {torch.argmax(explanation['temporal']['feature_importance']).item()}")

print("\n=== 空间分析结果 ===")
print("Top-5重要边:")
for src, dst, imp in explanation['spatial']['important_edges'][:5]:
    print(f"  站点{src} → 站点{dst}: {imp:.4f}")

# 8. GAT注意力分析(如果提取了)
if 'attention' in explanation['spatial']:
    attention_mean = explanation['spatial']['attention']['mean']
    print(f"\n=== GAT注意力统计 ===")
    print(f"注意力均值: {attention_mean.mean():.4f}")
    print(f"注意力标准差: {attention_mean.std():.4f}")
    print(f"最大注意力: {attention_mean.max():.4f}")
    print(f"最小注意力: {attention_mean.min():.4f}")
```

### 3. 命令行使用

```bash
# 基本分析(分析100个样本,生成所有可视化)
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --visualize

# 夏季分析 + GAT注意力提取
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --season summer \
    --extract_attention \
    --visualize

# 自定义保存目录
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --save_dir custom_explanations/ \
    --visualize
```

---

## 📊 生成的输出文件

### 数据文件

**explanation_data.npz** - 包含所有分析结果:

```python
data = np.load('explanations/explanation_data.npz')

# 时序分析结果
data['time_importance']          # [hist_len] 时间步重要性
data['feature_importance']       # [in_dim] 特征重要性
data['temporal_heatmap']         # [hist_len, in_dim] 时空热图

# 空间分析结果
data['edge_importance_mean']     # [num_edges] 边重要性均值
data['edge_importance_std']      # [num_edges] 边重要性标准差
data['edge_index']               # [2, num_edges] 边索引

# GAT注意力权重(v2.0.0+)
data['attention_mean']           # [num_edges] 注意力均值
data['attention_std']            # [num_edges] 注意力标准差
```

**important_edges.txt** - Top-K重要边列表:

```
站点59264 → 站点59287: 0.8523
站点59287 → 站点59316: 0.8201
站点59316 → 站点59324: 0.7856
...
```

### 可视化图表 (11种)

**基础可视化 (5种) - v1.0.0**:
1. `temporal_heatmap.png` - 时序特征热图 `[hist_len × in_dim]`
2. `spatial_edges.png` - Top-K重要边地理图 (Mapbox底图)
3. `edge_distribution.png` - 边重要性分布直方图
4. `time_importance.png` - 时间步重要性柱状图
5. `feature_importance.png` - 特征重要性排名图

**全边可视化 (3种) - v2.0.0**:
6. `spatial_all_edges_overlay.png` - 全边叠加模式图 (灰色全边 + 红色Top-K)
7. `spatial_all_edges_separate.png` - 全边分离模式图 (左右对比)
8. `comparison_explainer_vs_attention.png` - GNNExplainer vs GAT注意力对比图

**GAT注意力深度分析 (3种) - v2.1.0** ⭐⭐⭐:
9. `attention_matrix_heatmap.png` - 全局注意力矩阵热力图 (28×28)
10. `distance_vs_attention.png` - 距离-注意力散点图 (趋势线 + 统计检验)
11. `correlation_vs_attention.png` - 温度相关性-注意力散点图 (R² + p值)

---

## 🔬 高级用法

### 1. GAT注意力权重深度分析 ⭐⭐⭐ (v2.1.0新增)

对GAT模型学习的注意力权重进行全面验证:

```python
from myGNN.explainer import SpatialExplainer, ExplainerConfig
from myGNN.explainer.utils import (
    edge_attention_to_matrix,
    compute_edge_distances,
    compute_temperature_correlation,
    extract_edge_correlations,
    haversine_distance
)
from scipy.stats import pearsonr, linregress
import numpy as np
import matplotlib.pyplot as plt

# === 步骤1: 提取GAT注意力权重 ===
exp_config = ExplainerConfig(num_samples=100, extract_attention=True)
spatial_explainer = SpatialExplainer(model, config, exp_config)

attention_result = spatial_explainer.extract_attention_weights_batch(
    test_loader, num_samples=100
)

attention_mean = attention_result['attention_mean']  # [num_edges]
attention_std = attention_result['attention_std']    # [num_edges]
edge_index = attention_result['edge_index']         # [2, num_edges]

print(f"成功提取 {len(attention_mean)} 条边的注意力权重")
print(f"注意力均值: {attention_mean.mean():.4f} ± {attention_mean.std():.4f}")

# === 步骤2: 分析注意力-距离关系 ===
# 2.1 计算所有边的地理距离(使用Haversine公式)
edge_distances = compute_edge_distances(edge_index, station_coords)
print(f"\n距离统计: {edge_distances.min():.1f}km - {edge_distances.max():.1f}km")

# 2.2 统计分析
r_dist, p_dist = pearsonr(edge_distances, attention_mean.numpy())
slope, intercept, r_value, p_value, std_err = linregress(
    edge_distances, attention_mean.numpy()
)

print(f"\n=== 距离-注意力关系 ===")
print(f"Pearson相关系数: r = {r_dist:.3f}")
print(f"p值(显著性): p = {p_dist:.2e}")
print(f"线性回归: attention = {slope:.6f} × distance + {intercept:.4f}")
print(f"R²(拟合优度): {r_value**2:.3f}")

if r_dist < 0:
    print("✓ 负相关: 距离越远,注意力越小(符合物理规律)")
else:
    print("⚠ 正相关: 需要进一步检查模型")

# 2.3 可视化距离-注意力关系
from myGNN.explainer.visualize_explainer import plot_distance_vs_attention
plot_distance_vs_attention(
    edge_distances, attention_mean,
    save_path='analysis_distance_vs_attention.png',
    dpi=300
)

# === 步骤3: 分析注意力-温度相关性关系 ===
# 3.1 计算训练集温度相关性矩阵(避免数据泄露)
weather_data = np.load('data/real_weather_data_2010_2019.npy')
corr_matrix = compute_temperature_correlation(
    weather_data,
    train_indices=(0, 2922),  # 仅使用训练集(2010-2017)
    target_feature_idx=4      # tmax最高气温
)
print(f"\n温度相关性矩阵形状: {corr_matrix.shape}")  # (28, 28)

# 3.2 提取边级相关系数
edge_corrs = extract_edge_correlations(edge_index, corr_matrix)
print(f"边级相关性统计: {edge_corrs.min():.3f} - {edge_corrs.max():.3f}")

# 3.3 统计分析
r_corr, p_corr = pearsonr(edge_corrs, attention_mean.numpy())
print(f"\n=== 温度相关性-注意力关系 ===")
print(f"Pearson相关系数: r = {r_corr:.3f}")
print(f"p值(显著性): p = {p_corr:.2e}")

if r_corr > 0 and p_corr < 0.05:
    print("✓ 显著正相关: 温度模式相似的站点注意力高(符合气象规律)")
else:
    print("⚠ 相关性不显著或为负,需要进一步检查")

# 3.4 可视化温度相关性-注意力关系
from myGNN.explainer.visualize_explainer import plot_correlation_vs_attention
plot_correlation_vs_attention(
    edge_corrs, attention_mean,
    save_path='analysis_correlation_vs_attention.png',
    dpi=300
)

# === 步骤4: 检查最高/最低注意力的边 ===
top_indices = np.argsort(attention_mean.numpy())[-10:][::-1]
bottom_indices = np.argsort(attention_mean.numpy())[:10]

print("\n=== 最高注意力的10条边 ===")
print(f"{'源站点':<8} {'目标站点':<8} {'注意力':<10} {'距离(km)':<10} {'温度相关性':<10}")
print("-" * 60)
for idx in top_indices:
    src, dst = edge_index[:, idx]
    dist = edge_distances[idx]
    corr = edge_corrs[idx]
    attn = attention_mean[idx]
    print(f"{src:<8} {dst:<8} {attn:.4f}     {dist:>6.1f}     {corr:>6.3f}")

print("\n=== 最低注意力的10条边 ===")
print(f"{'源站点':<8} {'目标站点':<8} {'注意力':<10} {'距离(km)':<10} {'温度相关性':<10}")
print("-" * 60)
for idx in bottom_indices:
    src, dst = edge_index[:, idx]
    dist = edge_distances[idx]
    corr = edge_corrs[idx]
    attn = attention_mean[idx]
    print(f"{src:<8} {dst:<8} {attn:.4f}     {dist:>6.1f}     {corr:>6.3f}")

# === 步骤5: 全局注意力矩阵可视化 ===
attention_matrix = edge_attention_to_matrix(
    edge_index, attention_mean,
    num_nodes=28, aggregation='mean'
)
print(f"\n全局注意力矩阵形状: {attention_matrix.shape}")  # (28, 28)

from myGNN.explainer.visualize_explainer import plot_attention_matrix_heatmap
plot_attention_matrix_heatmap(
    attention_matrix,
    save_path='analysis_attention_matrix_heatmap.png',
    dpi=300
)

# === 步骤6: 案例研究 - 特定站点对分析 ===
def analyze_station_pair(src_id, dst_id, edge_index, attention_mean,
                         edge_distances, edge_corrs, station_coords):
    """分析特定站点对的注意力权重"""
    # 查找边索引
    edge_mask = (edge_index[0] == src_id) & (edge_index[1] == dst_id)
    if not edge_mask.any():
        print(f"未找到站点{src_id}→{dst_id}的边")
        return

    idx = torch.where(edge_mask)[0][0].item()

    # 获取信息
    attn = attention_mean[idx].item()
    dist = edge_distances[idx]
    corr = edge_corrs[idx]

    # 计算相对排名
    rank = (attention_mean >= attn).sum().item()
    total = len(attention_mean)

    print(f"\n=== 站点对分析: {src_id} → {dst_id} ===")
    print(f"注意力权重: {attn:.4f}")
    print(f"重要性排名: {rank}/{total} (前{rank/total*100:.1f}%)")
    print(f"地理距离: {dist:.1f} km")
    print(f"温度相关性: {corr:.3f}")

    # 计算实际距离
    src_coord = station_coords[src_id]
    dst_coord = station_coords[dst_id]
    actual_dist = haversine_distance(
        src_coord[1], src_coord[0],  # lat, lon
        dst_coord[1], dst_coord[0]
    )
    print(f"实际地理距离: {actual_dist:.1f} km (Haversine公式)")

# 示例: 分析站点0→5的连接
analyze_station_pair(0, 5, edge_index, attention_mean,
                    edge_distances, edge_corrs, station_coords)
```

### 2. 仅提取注意力权重(不运行完整分析)

```python
from myGNN.explainer import SpatialExplainer, ExplainerConfig

# 配置仅提取注意力
exp_config = ExplainerConfig(num_samples=100, extract_attention=True)
spatial_explainer = SpatialExplainer(model, config, exp_config)

# 批量提取
attention_result = spatial_explainer.extract_attention_weights_batch(
    data_loader=test_loader,
    num_samples=100
)

if attention_result is not None:
    # 保存结果
    np.savez('attention_weights.npz',
             attention_mean=attention_result['attention_mean'].numpy(),
             attention_std=attention_result['attention_std'].numpy(),
             edge_index=attention_result['edge_index'].numpy())
    print("✓ 注意力权重已保存到 attention_weights.npz")
else:
    print("⚠ 模型不支持注意力提取(可能不是GAT模型)")
```

### 3. 自定义可视化

使用可视化函数生成自定义图表:

```python
from myGNN.explainer.visualize_explainer import (
    plot_spatial_edges_with_all,
    plot_gat_attention_comparison,
    plot_attention_matrix_heatmap,
    plot_distance_vs_attention,
    plot_correlation_vs_attention
)
import numpy as np

# 加载数据
data = np.load('explanations/explanation_data.npz')
edge_index = data['edge_index']
edge_importance = data['edge_importance_mean']
attention_mean = data['attention_mean']

# 1. 自定义全边可视化(仅显示Top-30)
plot_spatial_edges_with_all(
    edge_importance=edge_importance,
    edge_index=edge_index,
    station_coords=station_coords,
    save_path='custom_top30_all_edges.png',
    top_k=30,
    show_mode='overlay',
    dpi=300
)

# 2. 自定义GNNExplainer vs GAT对比图
plot_gat_attention_comparison(
    explainer_importance=edge_importance,
    attention_weights=attention_mean,
    edge_index=edge_index,
    station_coords=station_coords,
    save_path='custom_comparison.png',
    top_k=15,
    dpi=300
)

# 3. 自定义注意力矩阵热力图(使用max聚合)
from myGNN.explainer.utils import edge_attention_to_matrix
attention_matrix_max = edge_attention_to_matrix(
    edge_index, attention_mean,
    num_nodes=28, aggregation='max'
)
plot_attention_matrix_heatmap(
    attention_matrix_max,
    save_path='custom_attention_matrix_max.png',
    title='GAT Attention Matrix (Max Aggregation)',
    dpi=300
)

# 4. 自定义距离-注意力分析(高分辨率)
edge_distances = compute_edge_distances(edge_index, station_coords)
plot_distance_vs_attention(
    edge_distances, attention_mean,
    save_path='custom_high_res_distance.png',
    dpi=600,  # 高分辨率
    figsize=(12, 8)
)

# 5. 自定义温度相关性-注意力分析
weather_data = np.load('data/real_weather_data_2010_2019.npy')
corr_matrix = compute_temperature_correlation(
    weather_data,
    train_indices=(0, 2922),
    target_feature_idx=4  # tmax
)
edge_corrs = extract_edge_correlations(edge_index, corr_matrix)
plot_correlation_vs_attention(
    edge_corrs, attention_mean,
    save_path='custom_correlation.png',
    dpi=300
)
```

### 4. 季节差异分析

对比不同季节的模型行为:

```python
from myGNN.explainer import HybridExplainer, ExplainerConfig
import numpy as np
import matplotlib.pyplot as plt

seasons = ['spring', 'summer', 'autumn', 'winter']
results = {}

# 分别分析四个季节
for season in seasons:
    exp_config = ExplainerConfig(
        num_samples=100,
        season=season,
        extract_attention=True
    )
    explainer = HybridExplainer(model, config, exp_config)
    results[season] = explainer.explain_full(
        test_loader,
        save_path=f'explanations/{season}/'
    )
    print(f"✓ 完成 {season} 分析")

# 对比特征重要性
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
feature_names = ['x', 'y', 'height', 'tmin', 'tmax', 'tave', 'pre', 'prs', 'rh', 'win']

for idx, season in enumerate(seasons):
    ax = axes[idx // 2, idx % 2]
    feat_imp = results[season]['temporal']['feature_importance'][:10]  # 前10个特征
    ax.barh(range(10), feat_imp.numpy())
    ax.set_yticks(range(10))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Importance')
    ax.set_title(f'{season.capitalize()} - Feature Importance')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('seasonal_comparison_features.png', dpi=300, bbox_inches='tight')
print("✓ 季节对比图已保存")

# 对比边重要性差异
print("\n=== 季节边重要性差异分析 ===")
for i, season1 in enumerate(seasons):
    for season2 in seasons[i+1:]:
        data1 = np.load(f'explanations/{season1}/explanation_data.npz')
        data2 = np.load(f'explanations/{season2}/explanation_data.npz')

        diff = data1['edge_importance_mean'] - data2['edge_importance_mean']
        max_diff_idx = np.argmax(np.abs(diff))

        print(f"{season1} vs {season2}:")
        print(f"  最大差异边索引: {max_diff_idx}")
        print(f"  差异值: {diff[max_diff_idx]:.4f}")
```

### 5. 多模型对比分析

对比不同模型的解释:

```python
from myGNN.explainer import HybridExplainer, ExplainerConfig
import torch

models = {
    'GAT_LSTM': 'checkpoints/GAT_LSTM_best/best_model.pth',
    'GSAGE_LSTM': 'checkpoints/GSAGE_LSTM_best/best_model.pth',
    'GAT_SeparateEncoder': 'checkpoints/GAT_SeparateEncoder_best/best_model.pth'
}

exp_config = ExplainerConfig(num_samples=100, extract_attention=True)
explanations = {}

for model_name, model_path in models.items():
    print(f"\n分析模型: {model_name}")
    model = torch.load(model_path)
    model.eval()

    explainer = HybridExplainer(model, config, exp_config)
    explanations[model_name] = explainer.explain_full(
        test_loader,
        save_path=f'results/{model_name}/explanations/'
    )

# 对比分析
print("\n=== 模型对比分析 ===")
print(f"\n{'模型':<25} {'Top-1特征':<12} {'Top-1边权重':<12} {'注意力均值':<12}")
print("-" * 70)

for model_name, explanation in explanations.items():
    # 最重要特征
    top_feat_idx = torch.argmax(explanation['temporal']['feature_importance']).item()

    # 最重要边的权重
    top_edge_imp = explanation['spatial']['edge_importance_mean'].max().item()

    # 注意力均值(如果有)
    if 'attention' in explanation['spatial']:
        attn_mean = explanation['spatial']['attention']['mean'].mean().item()
        attn_str = f"{attn_mean:.4f}"
    else:
        attn_str = "N/A"

    print(f"{model_name:<25} {top_feat_idx:<12} {top_edge_imp:<12.4f} {attn_str:<12}")
```

---

## 🔧 配置参数详解

### ExplainerConfig参数

| 参数 | 类型 | 默认值 | 版本 | 说明 |
|------|------|--------|------|------|
| `num_samples` | int | 100 | v1.0 | 分析样本数量 |
| `epochs` | int | 200 | v1.0 | GNNExplainer训练轮数 |
| `season` | str\|None | None | v1.0 | 季节筛选 ('spring', 'summer', 'autumn', 'winter', None) |
| `ig_steps` | int | 50 | v1.0 | Integrated Gradients积分步数 |
| `lr` | float | 0.01 | v1.0 | GNNExplainer学习率 |
| `top_k_edges` | int | 20 | v1.0 | 保存的Top-K重要边数量 |
| `extract_attention` | bool | True | v2.0 | 是否提取GAT注意力权重 |
| `all_edges_mode` | str\|None | 'both' | v2.0 | 全边可视化模式 |
| `use_basemap` | bool | True | v2.0 | 是否使用Mapbox地图底图 |
| `viz_dpi` | int | 300 | v2.0 | 图表分辨率 |

### all_edges_mode选项

- `'overlay'`: 只生成叠加模式图(灰色全边 + 红色Top-K)
- `'separate'`: 只生成分离模式图(左右对比子图)
- `'both'`: 生成两种模式图(推荐)
- `None`: 不生成全边可视化图

---

## 📐 技术细节

### GAT注意力权重聚合策略

**三级聚合流程:**

1. **多头聚合** (Head-level Aggregation):
   ```python
   # GAT每层有H个attention head
   attn_weights: [num_edges, num_heads]
   attn_avg = attn_weights.mean(dim=1)  # → [num_edges]
   ```

2. **多层聚合** (Layer-level Aggregation):
   ```python
   # GAT有L层,每层产生一组注意力权重
   layer_attns = [layer_1_attn, layer_2_attn, ...]  # 每个 [num_edges]
   layer_stacked = torch.stack(layer_attns)  # [num_layers, num_edges]
   sample_avg = layer_stacked.mean(dim=0)    # [num_edges]
   ```

3. **多样本聚合** (Sample-level Aggregation):
   ```python
   # 对N个样本的注意力求统计量
   all_samples = torch.stack(sample_attns)  # [num_samples, num_edges]
   attention_mean = all_samples.mean(dim=0)  # [num_edges]
   attention_std = all_samples.std(dim=0)    # [num_edges]
   ```

**设计理由:**
- 多头聚合: GAT多头机制捕获不同模式,平均可获得整体依赖
- 多层聚合: 深层注意力更关注高级特征,浅层关注低级特征,平均兼顾
- 多样本聚合: 提高统计鲁棒性,减少单样本随机性

### 距离计算 - Haversine公式 (v2.1.0)

使用Haversine公式计算球面距离,考虑地球曲率:

```python
def haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    计算球面距离(公里)

    Args:
        lat1, lon1: 点1的纬度和经度(度)
        lat2, lon2: 点2的纬度和经度(度)
        radius: 地球半径(km, 默认6371)

    Returns:
        distance: 距离(km)
    """
    # 转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius * c
```

**优势:**
- 精度高: 考虑地球曲率,适合中短距离(数百公里)
- 计算稳定: 避免数值溢出
- 标准化: 符合地理信息系统标准

### 温度相关性计算 (v2.1.0)

基于训练集温度数据计算站点间相关性:

```python
def compute_temperature_correlation(weather_data, train_indices, target_feature_idx):
    """
    计算训练集温度相关性矩阵

    Args:
        weather_data: [time_steps, num_stations, num_features]
        train_indices: (start, end) 训练集索引范围
        target_feature_idx: 目标特征索引(如4=tmax)

    Returns:
        corr_matrix: [num_stations, num_stations] 相关系数矩阵
    """
    train_start, train_end = train_indices

    # 提取训练集目标特征
    train_data = weather_data[train_start:train_end, :, target_feature_idx]
    # shape: [train_len, num_stations]

    # 计算Pearson相关系数矩阵
    corr_matrix = np.corrcoef(train_data.T)  # [num_stations, num_stations]

    return corr_matrix
```

**设计考虑:**
- **避免数据泄露**: 仅使用训练集(2010-2017, 索引0-2921)
- **目标特征**: 通常使用tmax(索引4),也可选择tmin/tave
- **时间范围**: 8年数据(2922天),统计显著性高

### 边索引一致性

- GATv2Conv可能内部重排edge_index用于消息传递优化
- 本实现使用原始edge_index进行前向传播,确保返回的注意力权重与输入edge_index对应
- 验证方法: 检查返回的edge_index与输入edge_index是否一致

---

## 🔍 常见问题 (FAQ)

### Q1: 如何关闭注意力权重提取?

```python
exp_config = ExplainerConfig(extract_attention=False)
```

### Q2: 如何只生成叠加模式的全边图?

```python
exp_config = ExplainerConfig(all_edges_mode='overlay')
# 或在可视化时指定
generate_all_visualizations(..., all_edges_mode='overlay')
```

### Q3: 为什么GNNExplainer和GAT注意力权重不同?

**两者关注点不同:**

| 维度 | GNNExplainer | GAT注意力 |
|------|-------------|----------|
| **分析对象** | 整个模型(LSTM+GAT+MLP) | 仅GAT层 |
| **方法** | 事后扰动分析 | 模型原生权重 |
| **输出** | 边对最终预测的整体重要性 | GAT层内邻域聚合权重 |
| **优势** | 全面,考虑所有组件 | 直观,反映空间依赖 |
| **用途** | 理解整体决策机制 | 验证空间建模正确性 |

**互补关系:**
- GNNExplainer更全面,适合解释"为什么模型这样预测"
- GAT注意力更直观,适合验证"模型学到了什么空间规律"
- 两者对比可以发现模型的优缺点

### Q4: 如何理解距离-注意力的负相关?

**负相关(r<0)是符合预期的:**

```
距离 ↑ → 注意力 ↓  (负相关)
```

**物理解释:**
- 气象站之间的空间影响随距离衰减
- 距离近的站点温度模式更相似
- GAT模型正确学习了这种空间依赖

**统计检验:**
- |r| > 0.3: 中等相关
- |r| > 0.5: 强相关
- p < 0.05: 统计显著

**示例:**
```
r = -0.45, p = 1.2e-15  → 强负相关,高度显著
```

### Q5: 如何理解温度相关性-注意力的关系?

**正相关(r>0)表示模型学到了气象模式:**

```
温度相关性 ↑ → 注意力 ↑  (正相关)
```

**气象解释:**
- 温度模式相似的站点通常受相同天气系统影响
- GAT应该对这些站点分配更高注意力
- 验证模型是否学到了真实的气象规律

**注意:**
- 相关性基于训练集计算,避免数据泄露
- 仅使用2010-2017年数据(索引0-2921)
- 不包含验证集(2018)和测试集(2019)

### Q6: 如何处理大图(边数过多)?

**优化策略:**

1. **关闭全边可视化:**
   ```python
   exp_config = ExplainerConfig(all_edges_mode=None)
   ```

2. **调整Top-K参数:**
   ```python
   exp_config = ExplainerConfig(top_k_edges=10)  # 只保存Top-10
   ```

3. **使用叠加模式:**
   ```python
   exp_config = ExplainerConfig(all_edges_mode='overlay')  # 避免生成多张大图
   ```

4. **降低分辨率:**
   ```python
   exp_config = ExplainerConfig(viz_dpi=150)  # 降低DPI
   ```

### Q7: 旧的explanation_data.npz文件兼容吗?

**完全兼容!** 新代码向后兼容:

- v1.0.0文件: 缺少`attention_mean`字段,自动跳过注意力相关可视化
- v2.0.0+文件: 包含`attention_mean`字段,生成所有11种可视化
- 所有原有功能保持不变

### Q8: GSAGE模型支持注意力提取吗?

**不支持。** SAGEConv使用固定聚合(mean/max/add),没有可学习的注意力权重。

**处理方式:**
- 设置`extract_attention=False`或忽略此参数
- `extract_attention_weights_batch()`会返回None并提示
- 可视化会自动跳过注意力相关图表(图9-11)
- 其他功能正常使用(图1-8)

### Q9: 如何解读注意力矩阵热力图?

**理解热力图:**

```
attention_matrix[i, j] = 站点i对站点j的平均注意力
```

**观察要点:**
1. **对角线**: 通常较暗(自连接注意力低或不存在)
2. **亮点**: 高注意力连接,表示强空间依赖
3. **模式**:
   - 块状: 区域内站点相互关注
   - 条带状: 某些站点作为"枢纽"被广泛关注
   - 稀疏: 选择性关注,符合K近邻图结构

**验证:**
- 对比距离矩阵,检查近距离站点是否注意力高
- 对比温度相关性矩阵,检查相似站点是否注意力高

### Q10: 如何选择最佳的num_samples?

**权衡考虑:**

| num_samples | 分析时间 | 统计稳定性 | 推荐场景 |
|------------|---------|-----------|---------|
| 50 | 快 | 低 | 快速探索 |
| 100 | 中等 | 中等 | 标准分析(推荐) |
| 200+ | 慢 | 高 | 论文发表 |

**建议:**
- **开发阶段**: 50个样本,快速迭代
- **正式分析**: 100个样本,平衡性能和精度
- **学术发表**: 200+个样本,最大化统计可信度

---

## 📋 示例脚本

完整的使用示例请参考:
- [README.md](README.md) - 模块概览
- [../../CLAUDE.md](../../CLAUDE.md) - 项目架构详细说明
- [../../myGNN/README.md](../README.md) - myGNN框架文档

---

## 📖 参考文献

**方法论:**
- **GNNExplainer**: Ying et al. "GNNExplainer: Generating Explanations for Graph Neural Networks." NeurIPS 2019.
- **Integrated Gradients**: Sundararajan et al. "Axiomatic Attribution for Deep Networks." ICML 2017.

**地理计算:**
- **Haversine公式**: R.W. Sinnott. "Virtues of the Haversine." Sky and Telescope 68(2):159, 1984.

---

## 🙏 反馈与贡献

如有问题或建议,请联系项目维护者或提交Issue。

---

<div align="center">

**版本**: v2.1.0
**最后更新**: 2025-12-16
**维护者**: GNN气温预测项目组

</div>
