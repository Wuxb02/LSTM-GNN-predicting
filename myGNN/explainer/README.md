# GNN模型可解释性分析模块

本模块为GAT_LSTM和GSAGE_LSTM模型提供全面的可解释性分析,包含时序和空间两个维度。

---

## ✨ 核心功能

### 1. 时序特征分析 (Temporal Analysis)

**使用Integrated Gradients解释哪些历史时刻和气象要素最重要**

- **方法**: Integrated Gradients (Sundararajan et al., ICML 2017)
- **分析对象**: 完整模型的输入特征重要性
- **输出维度**:
  - 时间步重要性: `[hist_len]` - 哪些历史时刻最重要
  - 特征重要性: `[in_dim]` - 哪些气象要素最重要
  - 时空热图: `[hist_len, in_dim]` - 时空交叉分析

**技术细节**:
- 基线选择: 零基线(所有特征为0)
- 积分步数: 默认50步(可配置)
- 支持批量分析: 对多个样本统计平均

### 2. 空间关系分析 (Spatial Analysis)

**使用GNNExplainer解释哪些气象站之间的连接最重要**

- **方法**: GNNExplainer (Ying et al., NeurIPS 2019)
- **分析对象**: GNN层的边重要性
- **输出维度**:
  - 边重要性均值: `[num_edges]` - 每条边的平均重要性
  - 边重要性标准差: `[num_edges]` - 重要性的稳定性
  - Top-K重要边: List[(src, dst, importance)] - 最重要的K条边

**技术细节**:
- 训练轮数: 默认200轮(可配置)
- 损失函数: 负对数似然 + 熵正则化
- Wrapper模式: 从完整模型提取GNN层,共享原模型权重

### 3. GAT注意力分析 ⭐⭐⭐

**深度分析GAT模型学习的空间依赖关系**

本模块提供GAT注意力权重的全面分析,验证模型是否学习到正确的空间依赖规律。

#### 3.1 全局注意力矩阵可视化

将稀疏的边级注意力权重转换为密集的28×28节点级矩阵:

```python
from myGNN.explainer.utils import edge_attention_to_matrix

# 转换边级注意力为矩阵
attention_matrix = edge_attention_to_matrix(
    edge_index,           # [2, num_edges]
    attention_weights,    # [num_edges]
    num_nodes=28,
    aggregation='mean'    # 'mean', 'max', 'sum'
)
# 返回: [28, 28] 全局注意力矩阵
```

**可视化**: 生成28×28热力图,显示所有站点对之间的注意力强度。

#### 3.2 距离-注意力关系验证

验证模型是否学习到"距离近的站点注意力高"的物理规律:

```python
from myGNN.explainer.utils import compute_edge_distances

# 计算所有边的地理距离(使用Haversine公式)
edge_distances = compute_edge_distances(
    edge_index,      # [2, num_edges]
    station_coords   # [28, 2] 经纬度
)
# 返回: [num_edges] 距离(公里)

# 统计分析
from scipy.stats import pearsonr, linregress
r, p = pearsonr(edge_distances, attention_weights.numpy())
slope, intercept, r_value, p_value, std_err = linregress(
    edge_distances, attention_weights.numpy()
)
```

**可视化**: 生成散点图 + 线性回归趋势线,显示Pearson相关系数r、p值、R²。

#### 3.3 温度相关性-注意力关系验证

验证模型是否学习到"温度模式相似的站点注意力高"的气象规律:

```python
from myGNN.explainer.utils import (
    compute_temperature_correlation,
    extract_edge_correlations
)
import numpy as np

# 1. 计算训练集温度相关性矩阵(避免数据泄露)
weather_data = np.load('data/real_weather_data_2010_2017.npy')
corr_matrix = compute_temperature_correlation(
    weather_data,
    train_indices=(0, 2191),  # 仅使用训练集(2010-2015)
    target_feature_idx=4      # tmax最高气温
)
# 返回: [28, 28] 皮尔逊相关系数矩阵

# 2. 提取边级相关系数
edge_corrs = extract_edge_correlations(edge_index, corr_matrix)
# 返回: [num_edges]

# 3. 统计分析
r_corr, p_corr = pearsonr(edge_corrs, attention_weights.numpy())
print(f"温度相关性-注意力相关性: r={r_corr:.3f}, p={p_corr:.2e}")
```

**可视化**: 生成散点图,显示相关系数r、p值,验证模型是否学到气象模式。

#### 3.4 多层多头注意力聚合

GAT模型包含多层(L层)和多头(H头)注意力,本模块支持多种聚合策略:

```python
# 聚合策略
aggregation_strategy = 'mean'  # 'mean', 'max', 'sum', 'last_layer'

# 'mean': 对所有层和头求平均(默认)
# 'max': 取最大注意力值
# 'sum': 求和(需后续归一化)
# 'last_layer': 只使用最后一层(认为最后一层最重要)
```

**实现细节**: 见`myGNN/explainer/spatial_explainer.py:extract_attention_weights_batch()`

### 4. 季节筛选分析

支持针对特定季节(春夏秋冬)进行分析,发现季节性规律:

```python
from myGNN.explainer import ExplainerConfig

exp_config = ExplainerConfig(
    season='summer'  # 'spring', 'summer', 'autumn', 'winter', None
)
```

**季节定义** (基于月份):
- 春季: 3, 4, 5月
- 夏季: 6, 7, 8月
- 秋季: 9, 10, 11月
- 冬季: 12, 1, 2月

**实现**: 见`myGNN/explainer/utils.py:filter_samples_by_season()`

### 5. 地理可视化

**使用Mapbox WMTS在线地图底图,生成专业级地理可视化**

- **底图来源**: Mapbox Satellite Streets (WMTS 1.0.0标准)
- **投影系统**: Web Mercator (EPSG:3857)
- **绘图库**: Cartopy + Matplotlib
- **降级方案**: 网络不可用时使用Natural Earth离线数据

**支持的可视化类型**:
1. Top-K重要边地理图: 在地图上绘制最重要的K条边
2. 全边叠加图: 所有边在同一图上叠加,边宽度表示重要性
3. 全边分离图: 每条边单独绘制在网格子图中

**实现**: 见`myGNN/explainer/visualize_explainer.py:plot_spatial_edges()`

详细配置指南: [MAPBOX_WMTS_GUIDE.md](../../MAPBOX_WMTS_GUIDE.md)

---

## 📊 输出结果

### 1. 数据文件

**explanation_data.npz** - 原始数据(含注意力权重):

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

# GAT注意力权重 ⭐
data['attention_mean']           # [num_edges] 注意力均值
data['attention_std']            # [num_edges] 注意力标准差
```

**important_edges.txt** - Top-K重要边列表:

```
站点59264 → 站点59287: 0.8523
站点59287 → 站点59316: 0.8201
...
```

### 2. 可视化图表 (11种) ⭐

**基础可视化 (8种)**:
1. `temporal_heatmap.png` - 时序特征热图 `[hist_len × in_dim]`
2. `spatial_edges.png` - Top-K重要边地理图 (Mapbox底图)
3. `spatial_all_edges_overlay.png` - 全边叠加图
4. `spatial_all_edges_separate.png` - 全边分离图 (网格子图)
5. `comparison_explainer_vs_attention.png` - GNNExplainer vs GAT注意力对比
6. `edge_distribution.png` - 边重要性分布直方图
7. `time_importance.png` - 时间步重要性柱状图
8. `feature_importance.png` - 特征重要性排名图

**GAT注意力深度分析 (3种)** ⭐⭐⭐:
9. `attention_matrix_heatmap.png` - 全局注意力矩阵热力图 (28×28)
10. `distance_vs_attention.png` - 距离-注意力散点图 (趋势线 + 统计检验)
11. `correlation_vs_attention.png` - 温度相关性-注意力散点图 (R² + p值)

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

**依赖说明**:
- `captum`: Integrated Gradients分析
- `cartopy`: 地理可视化 (Mapbox WMTS底图)
- `scipy`: 统计检验 (Pearson相关、线性回归)

### 2. 命令行使用

```bash
# 基本分析 (分析100个样本)
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --visualize

# 仅分析夏季样本
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --season summer \
    --visualize

# 提取GAT注意力权重并进行深度分析
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --extract_attention \
    --visualize
```

**参数说明**:
- `--model_path`: 训练好的模型权重路径 (必需)
- `--num_samples`: 分析样本数量 (default: 100)
- `--season`: 季节筛选 (choices: spring, summer, autumn, winter)
- `--epochs`: GNNExplainer训练轮数 (default: 200)
- `--ig_steps`: Integrated Gradients积分步数 (default: 50)
- `--top_k_edges`: 保存Top-K重要边 (default: 20)
- `--extract_attention`: 提取GAT注意力权重 (仅GAT模型)
- `--save_dir`: 结果保存目录 (default: 模型目录/explanations/)
- `--visualize`: 生成可视化图表

### 3. Python API使用

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

# 5. 配置解释器
exp_config = ExplainerConfig(
    num_samples=100,           # 分析100个样本
    epochs=200,                # GNNExplainer训练轮数
    season='summer',           # 仅分析夏季样本
    extract_attention=True,    # 提取GAT注意力(仅GAT模型)
    top_k_edges=20            # 保存Top-20重要边
)

# 6. 运行完整分析
explainer = HybridExplainer(model, config, exp_config)
explanation = explainer.explain_full(
    test_loader,
    save_path='checkpoints/GAT_LSTM_best/explanations/summer/'
)

# 7. 访问结果
print("最重要的时间步:", torch.argmax(explanation['temporal']['time_importance']).item())
print("最重要的特征:", torch.argmax(explanation['temporal']['feature_importance']).item())
print("\nTop-5重要边:")
for src, dst, imp in explanation['spatial']['important_edges'][:5]:
    print(f"  站点{src} → 站点{dst}: {imp:.4f}")

# 8. GAT注意力分析 (如果提取了注意力)
if 'attention' in explanation['spatial']:
    attention_mean = explanation['spatial']['attention']['mean']
    print(f"\nGAT注意力均值: {attention_mean.mean():.4f}")
    print(f"注意力标准差: {attention_mean.std():.4f}")
```

---

## 🔑 核心组件详解

### 1. GNNWrapper - GNN层提取器

**作用**: 从完整的LSTM-GNN模型中提取纯GNN部分,用于GNNExplainer分析。

**支持的模型**:
- `GAT_LSTM` → `GATWrapper`
- `GSAGE_LSTM` → `GSAGEWrapper`

**实现原理**:
```python
# 原始模型
LSTM-GNN模型:
  输入 [batch, nodes, hist_len, in_dim]
  → LSTM [batch×nodes, hist_len×in_dim → hid_dim]
  → GAT/SAGE [batch×nodes, hid_dim → hid_dim]
  → MLP [batch×nodes, hid_dim → pred_len]

# Wrapper模型
GNNWrapper:
  输入 [batch×nodes, hid_dim]  # 固定维度特征
  → GAT/SAGE [batch×nodes, hid_dim → hid_dim]  # 提取的GNN层
  → 返回 [batch×nodes, hid_dim]
```

**使用方法**:
```python
from myGNN.explainer import create_gnn_wrapper

# 自动识别模型类型并创建wrapper
wrapper = create_gnn_wrapper(model)

# 验证wrapper一致性
from myGNN.explainer.gnn_wrapper import verify_wrapper_consistency
is_consistent, max_error = verify_wrapper_consistency(
    model, wrapper, test_input, edge_index
)
print(f"Wrapper一致性: {is_consistent}, 误差: {max_error}")
```

**实现**: 见`myGNN/explainer/gnn_wrapper.py`

### 2. TemporalAnalyzer - 时序特征分析器

**基于Integrated Gradients分析时序特征重要性**

**核心方法**:

```python
from myGNN.explainer import TemporalAnalyzer

analyzer = TemporalAnalyzer(model, config)

# 分析单个样本
result_single = analyzer.analyze_single(
    x,            # [nodes, hist_len, in_dim]
    edge_index    # [2, num_edges]
)
# 返回: attributions [nodes, hist_len, in_dim]

# 批量分析
result_batch = analyzer.analyze_batch(
    test_loader,
    num_samples=100
)
# 返回:
# {
#     'time_importance': [hist_len],
#     'feature_importance': [in_dim],
#     'temporal_heatmap': [hist_len, in_dim]
# }
```

**技术细节**:
- 基线选择: 零基线(所有特征为0)
- 积分步数: 默认50步,通过`exp_config.ig_steps`配置
- 聚合方式: 对所有节点和样本求平均

**实现**: 见`myGNN/explainer/temporal_analyzer.py`

### 3. SpatialExplainer - 空间关系分析器

**基于GNNExplainer分析空间关系重要性,并可提取GAT注意力权重**

**核心方法**:

```python
from myGNN.explainer import SpatialExplainer, ExplainerConfig

exp_config = ExplainerConfig(
    num_samples=100,
    epochs=200,
    extract_attention=True  # 提取GAT注意力
)
explainer = SpatialExplainer(model, config, exp_config)

# 分析单个样本
result_single = explainer.explain_single(
    x,            # [nodes, hist_len, in_dim]
    edge_index,   # [2, num_edges]
    target_node=5 # 解释站点5的预测
)
# 返回: edge_mask [num_edges]

# 批量分析
result_batch = explainer.explain_batch(
    test_loader,
    num_samples=100
)
# 返回:
# {
#     'edge_importance_mean': [num_edges],
#     'edge_importance_std': [num_edges],
#     'important_edges': List[(src, dst, importance)],
#     'attention': {  # 如果extract_attention=True
#         'mean': [num_edges],
#         'std': [num_edges]
#     }
# }
```

**GAT注意力提取**:
```python
# 提取GAT注意力权重
attention_result = explainer.extract_attention_weights_batch(
    test_loader,
    num_samples=100
)
# 返回:
# {
#     'attention_mean': [num_edges],
#     'attention_std': [num_edges],
#     'edge_index': [2, num_edges]
# }
```

**实现**: 见`myGNN/explainer/spatial_explainer.py`

### 4. HybridExplainer - 混合解释器

**整合时序和空间分析,提供完整的可解释性分析**

**核心方法**:

```python
from myGNN.explainer import HybridExplainer

explainer = HybridExplainer(model, config, exp_config)

# 运行完整分析
explanation = explainer.explain_full(
    test_loader,
    save_path='checkpoints/model/explanations/',
    visualize=True  # 生成11种可视化
)

# 返回结果
explanation = {
    'temporal': {
        'time_importance': [hist_len],
        'feature_importance': [in_dim],
        'temporal_heatmap': [hist_len, in_dim]
    },
    'spatial': {
        'edge_importance_mean': [num_edges],
        'edge_importance_std': [num_edges],
        'important_edges': List[(src, dst, importance)],
        'attention': {  # 如果extract_attention=True
            'mean': [num_edges],
            'std': [num_edges]
        }
    }
}
```

**实现**: 见`myGNN/explainer/hybrid_explainer.py`

---

## 📈 高级用法

### 1. 季节对比分析

对比不同季节的模型行为差异:

```bash
# 分别分析四个季节
for season in spring summer autumn winter; do
    python myGNN/explain_model.py \
        --model_path checkpoints/GAT_LSTM_best/best_model.pth \
        --num_samples 100 \
        --season $season \
        --save_dir checkpoints/GAT_LSTM_best/explanations/$season/ \
        --visualize
done

# 对比不同季节的特征重要性、边重要性等
```

### 2. 多模型对比

对比GAT_LSTM和GSAGE_LSTM的解释差异:

```python
from myGNN.explainer import HybridExplainer, ExplainerConfig
import torch

models = {
    'GAT_LSTM': 'checkpoints/GAT_LSTM_best/best_model.pth',
    'GSAGE_LSTM': 'checkpoints/GSAGE_LSTM_best/best_model.pth'
}

exp_config = ExplainerConfig(num_samples=100, extract_attention=True)
explanations = {}

for model_name, model_path in models.items():
    model = torch.load(model_path)
    model.eval()

    explainer = HybridExplainer(model, config, exp_config)
    explanations[model_name] = explainer.explain_full(
        test_loader,
        save_path=f'results/{model_name}/explanations/'
    )

# 对比分析
print("特征重要性对比:")
for model_name, explanation in explanations.items():
    feat_imp = explanation['temporal']['feature_importance']
    print(f"{model_name}: Top-3特征索引 = {torch.topk(feat_imp, 3).indices.tolist()}")
```

### 3. 单样本深度分析

针对特定样本进行详细分析:

```python
from myGNN.explainer import SpatialExplainer, TemporalAnalyzer

# 选择特定样本
sample_idx = 42
test_sample = test_dataset[sample_idx]

# 时序分析
temporal_analyzer = TemporalAnalyzer(model, config)
attr = temporal_analyzer.analyze_single(
    test_sample.x,
    test_sample.edge_index
)
# attr: [nodes, hist_len, in_dim]

# 可视化特定站点的时序特征重要性
import matplotlib.pyplot as plt
station_id = 5
plt.figure(figsize=(12, 6))
plt.imshow(attr[station_id].cpu().numpy(), aspect='auto', cmap='RdBu_r')
plt.colorbar(label='Attribution')
plt.xlabel('Feature Index')
plt.ylabel('Time Step')
plt.title(f'Temporal Feature Attribution - Station {station_id}')
plt.savefig('station_5_attribution.png', dpi=300, bbox_inches='tight')

# 空间分析(针对特定站点)
spatial_explainer = SpatialExplainer(model, config, exp_config)
explanation = spatial_explainer.explain_single(
    test_sample.x,
    test_sample.edge_index,
    target_node=5  # 解释站点5的预测
)
# explanation: edge_mask [num_edges]
```

### 4. GAT注意力权重深度分析 ⭐⭐⭐

对GAT模型学习的注意力权重进行全面验证:

```python
from myGNN.explainer import SpatialExplainer, ExplainerConfig
from myGNN.explainer.utils import (
    edge_attention_to_matrix,
    compute_edge_distances,
    compute_temperature_correlation,
    extract_edge_correlations
)
from scipy.stats import pearsonr, linregress
import numpy as np
import matplotlib.pyplot as plt

# 1. 提取GAT注意力权重
exp_config = ExplainerConfig(num_samples=100, extract_attention=True)
spatial_explainer = SpatialExplainer(model, config, exp_config)

attention_result = spatial_explainer.extract_attention_weights_batch(
    test_loader, num_samples=100
)

attention_mean = attention_result['attention_mean']  # [num_edges]
attention_std = attention_result['attention_std']    # [num_edges]
edge_index = attention_result['edge_index']         # [2, num_edges]

# 2. 分析注意力-距离关系
edge_distances = compute_edge_distances(edge_index, station_coords)

# 计算相关系数和线性回归
r_dist, p_dist = pearsonr(edge_distances, attention_mean.numpy())
slope, intercept, r_value, p_value, std_err = linregress(
    edge_distances, attention_mean.numpy()
)

print(f"距离-注意力关系:")
print(f"  Pearson r = {r_dist:.3f}, p-value = {p_dist:.2e}")
print(f"  线性回归: y = {slope:.6f}x + {intercept:.4f}")
print(f"  R² = {r_value**2:.3f}")

# 3. 分析注意力-温度相关性关系
weather_data = np.load('data/real_weather_data_2010_2017.npy')
corr_matrix = compute_temperature_correlation(
    weather_data,
    train_indices=(0, 2191),  # 仅使用训练集
    target_feature_idx=4      # tmax
)
edge_corrs = extract_edge_correlations(edge_index, corr_matrix)

r_corr, p_corr = pearsonr(edge_corrs, attention_mean.numpy())
print(f"\n温度相关性-注意力关系:")
print(f"  Pearson r = {r_corr:.3f}, p-value = {p_corr:.2e}")

# 4. 检查最高/最低注意力的边
top_indices = np.argsort(attention_mean.numpy())[-5:]
bottom_indices = np.argsort(attention_mean.numpy())[:5]

print("\n最高注意力的5条边:")
for idx in top_indices:
    src, dst = edge_index[:, idx]
    dist = edge_distances[idx]
    corr = edge_corrs[idx]
    attn = attention_mean[idx]
    print(f"  站点{src}→{dst}: 注意力={attn:.4f}, 距离={dist:.1f}km, 相关性={corr:.3f}")

print("\n最低注意力的5条边:")
for idx in bottom_indices:
    src, dst = edge_index[:, idx]
    dist = edge_distances[idx]
    corr = edge_corrs[idx]
    attn = attention_mean[idx]
    print(f"  站点{src}→{dst}: 注意力={attn:.4f}, 距离={dist:.1f}km, 相关性={corr:.3f}")

# 5. 可视化全局注意力矩阵
attention_matrix = edge_attention_to_matrix(
    edge_index, attention_mean,
    num_nodes=28, aggregation='mean'
)

plt.figure(figsize=(10, 8))
plt.imshow(attention_matrix.cpu().numpy(), cmap='viridis', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xlabel('Target Node')
plt.ylabel('Source Node')
plt.title('Global GAT Attention Matrix (28×28)')
plt.savefig('attention_matrix_custom.png', dpi=300, bbox_inches='tight')
```

### 5. 自定义可视化

使用可视化函数生成自定义图表:

```python
from myGNN.explainer.visualize_explainer import (
    plot_temporal_heatmap,
    plot_spatial_edges,
    plot_edge_distribution,
    plot_time_importance,
    plot_feature_importance,
    plot_attention_matrix_heatmap,
    plot_distance_vs_attention,
    plot_correlation_vs_attention
)
import torch
import numpy as np

# 加载数据
data = np.load('explanations/explanation_data.npz')
edge_index = data['edge_index']
attention_mean = data['attention_mean']

# 1. 时序热图
plot_temporal_heatmap(
    torch.from_numpy(data['temporal_heatmap']),
    feature_names=['x', 'y', 'height', 'tmin', 'tmax', 'tave', 'pre', 'prs', 'rh', 'win'],
    save_path='custom_temporal_heatmap.png',
    dpi=300
)

# 2. 空间边图 (自定义Top-K)
plot_spatial_edges(
    torch.from_numpy(data['edge_importance_mean']),
    edge_index,
    station_coords,
    save_path='custom_top10_edges.png',
    top_k=10,  # 只显示Top-10
    use_basemap=True
)

# 3. 全局注意力矩阵
from myGNN.explainer.utils import edge_attention_to_matrix
attention_matrix = edge_attention_to_matrix(
    edge_index, attention_mean,
    num_nodes=28, aggregation='mean'
)
plot_attention_matrix_heatmap(
    attention_matrix,
    save_path='custom_attention_matrix.png',
    dpi=300
)

# 4. 距离-注意力分析
edge_distances = compute_edge_distances(edge_index, station_coords)
plot_distance_vs_attention(
    edge_distances, attention_mean,
    save_path='custom_distance_vs_attention.png',
    dpi=300
)

# 5. 温度相关性-注意力分析
weather_data = np.load('data/real_weather_data_2010_2017.npy')
corr_matrix = compute_temperature_correlation(
    weather_data,
    train_indices=(0, 2191),
    target_feature_idx=4
)
edge_corrs = extract_edge_correlations(edge_index, corr_matrix)
plot_correlation_vs_attention(
    edge_corrs, attention_mean,
    save_path='custom_correlation_vs_attention.png',
    dpi=300
)
```

---

## 📂 输出文件结构

```
checkpoints/GAT_LSTM_best/
└── explanations/
    ├── explanation_data.npz              # 原始数据(含注意力权重)
    ├── important_edges.txt               # Top-K重要边列表
    └── visualizations/
        ├── temporal_heatmap.png          # 时序特征热图
        ├── spatial_edges.png             # 空间边地理图(Top-K)
        ├── spatial_all_edges_overlay.png # 全边叠加图
        ├── spatial_all_edges_separate.png# 全边分离图
        ├── comparison_explainer_vs_attention.png # GNNExplainer vs GAT对比
        ├── edge_distribution.png         # 边重要性分布
        ├── time_importance.png           # 时间步柱状图
        ├── feature_importance.png        # 特征排名图
        ├── attention_matrix_heatmap.png  # ⭐全局注意力矩阵(28×28)
        ├── distance_vs_attention.png     # ⭐距离-注意力散点图
        └── correlation_vs_attention.png  # ⭐相关性-注意力散点图
```

---

## 🔧 配置选项

### ExplainerConfig类

```python
from myGNN.explainer import ExplainerConfig

exp_config = ExplainerConfig(
    # 采样配置
    num_samples=100,              # 分析样本数量
    season=None,                  # 季节筛选: 'spring', 'summer', 'autumn', 'winter', None

    # GNNExplainer配置
    epochs=200,                   # GNNExplainer训练轮数
    lr=0.01,                      # 学习率

    # Integrated Gradients配置
    ig_steps=50,                  # 积分步数

    # 输出配置
    top_k_edges=20,               # 保存Top-K重要边
    extract_attention=False,      # 是否提取GAT注意力(仅GAT模型)

    # 可视化配置
    use_basemap=True,             # 是否使用Mapbox地图底图
    viz_dpi=300                   # 图表分辨率
)
```

---

## 🐛 注意事项

### 1. 网络要求

- 空间边地理图需要访问Mapbox WMTS服务器
- 如果网络不可用,会自动降级为无底图版本或Natural Earth离线数据
- 建议使用稳定的网络连接

### 2. 性能优化

- 使用GPU可以显著加速分析(特别是Integrated Gradients)
- 建议`num_samples=100`,在精度和速度之间平衡
- GNNExplainer的`epochs`可以在100-300之间调整

### 3. 模型兼容性

- **TemporalAnalyzer**: 支持所有模型(LSTM, GAT_LSTM, GSAGE_LSTM等)
- **SpatialExplainer**: 仅支持包含GNN层的模型(GAT_LSTM, GSAGE_LSTM)
- **注意力提取**: 仅支持GAT模型(GAT_LSTM, GAT_SeparateEncoder)

### 4. 数据泄露避免

- 温度相关性计算**仅使用训练集**(2010-2015年,索引0-2190)
- 不包含验证集(2016年)和测试集(2017年)
- 确保分析的公平性和科学性

---

## 📚 常见问题 (FAQ)

### Q1: 如何获取气象站坐标?

**A**: 从原始数据集提取经纬度信息:
```python
import numpy as np
MetData = np.load('data/real_weather_data_2010_2017.npy')
station_coords = MetData[0, :, :2]  # [num_stations, 2] 经纬度
```

### Q2: 季节筛选不生效?

**A**: 确保数据集包含时间戳信息,或修改`utils.py`中的`extract_month_from_index()`函数:
```python
def extract_month_from_index(idx, start_year=2010):
    """
    根据索引提取月份

    假设数据从2010-01-01开始,索引0对应2010-01-01
    """
    from datetime import datetime, timedelta
    base_date = datetime(start_year, 1, 1)
    target_date = base_date + timedelta(days=int(idx))
    return target_date.month
```

### Q3: GNNExplainer收敛慢?

**A**: 尝试以下方法:
1. 调整`epochs` (100-300)
2. 调整学习率`lr` (0.001-0.1)
3. 增加`num_samples`以提高统计稳定性

### Q4: 如何理解注意力-距离的负相关?

**A**: 负相关(r<0)表示**距离越远,注意力越小**,这是符合物理规律的:
- 气象站之间的空间影响随距离衰减
- GAT模型学习到了这种空间依赖模式
- 这验证了模型学习的正确性

### Q5: 为什么要提取注意力而不只用GNNExplainer?

**A**: 两者互补:
- **GNNExplainer**: 事后解释,针对特定预测任务优化边重要性
- **GAT注意力**: 模型原生权重,反映训练过程中学到的全局空间依赖
- **对比分析**: 验证两种方法的一致性,增强可信度

### Q6: 如何处理"WMTS底图加载失败"?

**A**: 有3种方案:
1. **检查网络**: 确保能访问`api.mapbox.com`
2. **使用代理**: 配置HTTP代理访问Mapbox服务
3. **降级方案**: 自动使用Natural Earth离线数据(无需底图)

详细配置: [MAPBOX_WMTS_GUIDE.md](../../MAPBOX_WMTS_GUIDE.md)

### Q7: 特征索引如何对应特征名?

**A**: 特征索引对应关系(处理后的30维输入):
```python
# 基础特征 (0-25)
feature_names = [
    'x', 'y', 'height',                    # 0-2: 空间
    'tmin', 'tmax', 'tave',                # 3-5: 温度
    'pre', 'prs', 'rh', 'win',             # 6-9: 气象
    'BH', 'BHstd', 'SCD', 'PLA',           # 10-13: 城市环境
    'λp', 'λb', 'POI', 'POW', 'POV',       # 14-18: 城市环境
    'NDVI',                                # 19: 植被
    'surface_pressure', 'surface_solar_radiation',  # 20-21: ERA5
    'u_wind', 'v_wind',                    # 22-23: 风速
    'VegHeight_mean', 'VegHeight_std'      # 24-25: 植被高度
]

# 时间编码 (26-29)
# 26: doy_sin
# 27: doy_cos
# 28: month_sin
# 29: month_cos
```

详细说明: [DATA_FORMAT.md](../../DATA_FORMAT.md)

---

## 🔗 相关文档

- **详细使用指南**: [EXPLAINER_USAGE.md](EXPLAINER_USAGE.md) - 完整的使用教程
- **项目架构**: [CLAUDE.md](../../CLAUDE.md) - 项目架构详细说明 ⭐
- **myGNN框架**: [myGNN/README.md](../README.md) - 框架总览
- **数据格式**: [DATA_FORMAT.md](../../DATA_FORMAT.md) - 数据格式详解
- **地图可视化**: [MAPBOX_WMTS_GUIDE.md](../../MAPBOX_WMTS_GUIDE.md) - Mapbox配置指南

---

## 📄 技术细节

### 两阶段分层解释

1. **时序阶段**: Integrated Gradients分析完整模型的输入特征重要性
   - 分析对象: 完整LSTM-GNN模型
   - 输入: `[nodes, hist_len, in_dim]` 原始输入
   - 输出: 归因值 `[nodes, hist_len, in_dim]`

2. **空间阶段**: GNNExplainer仅分析GNN层的边重要性
   - 分析对象: 提取的GNN层(通过Wrapper)
   - 输入: `[nodes, hid_dim]` 固定维度特征
   - 输出: 边掩码 `[num_edges]`

### Wrapper模式原理

- 从完整模型提取GAT/SAGE层
- 共享原模型权重,无需重新训练
- 输入是LSTM输出的固定维度特征(`hid_dim`)
- 保证GNNExplainer的输入输出维度一致性

### 批量统计聚合

- 分析多个样本(通常100个)
- 计算边重要性的均值和标准差
- 提高解释的鲁棒性和泛化能力
- 减少单样本的随机性影响

---

## 📖 引用

如果您在学术研究中使用本可解释性分析模块,请引用:

```bibtex
@article{gnn_explainer_2025,
  title={Hybrid Explainability Framework for Graph Neural Networks in Weather Forecasting},
  author={...},
  note={Available at: https://github.com/...},
  year={2025}
}
```

**相关论文**:
- **GNNExplainer**: Ying et al. "GNNExplainer: Generating Explanations for Graph Neural Networks." NeurIPS 2019.
- **Integrated Gradients**: Sundararajan et al. "Axiomatic Attribution for Deep Networks." ICML 2017.

---

## 📜 许可证

本模块遵循主项目的许可证。

---

## 📧 联系方式

如有问题或建议,请联系项目维护者或提交Issue。

---

<div align="center">

**版本**: v2.1.0
**最后更新**: 2025-12-16
**维护者**: GNN气温预测项目组

</div>
