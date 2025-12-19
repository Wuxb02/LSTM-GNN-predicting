# myGNN - 基于图神经网络的气温预测框架

这是一个完全重构的GNN气温预测框架,提供统一的配置管理和训练接口。

**最新更新 (2025-12-16):**
- ✨ **超参数自动调优模块** - 基于Optuna的贝叶斯优化框架
- 🔑 **GAT_SeparateEncoder v3.0** - 分离式编码器,特征级交叉注意力+节点嵌入
- 🎯 **可解释性分析增强** - 11种可视化,GAT注意力深度分析
- 📊 **加权趋势损失** - 基于论文的自适应损失函数
- ⚙️ **统一配置管理** - 所有参数集中管理,无需命令行参数

---

## 📂 目录结构

```
myGNN/
├── # === 核心模块 ===
├── config.py                    # 配置管理模块（核心）⭐
├── dataset.py                   # 数据加载模块（4维时间编码）
├── network_GNN.py               # 训练核心模块
├── losses.py                    # 自适应损失函数
├── train.py                     # 主训练脚本（统一入口）⭐
├── train_enhanced.py            # 增强训练脚本
├── visualize_results.py         # 结果可视化
├── explain_model.py             # 可解释性分析入口
├── tune.py                      # 超参数调优入口 ⭐新增
├── feature_encoder.py           # 静态特征编码器
├── plot_graph_structure.py      # 图结构可视化
├── feature_correlation_analysis.py  # 特征相关性分析
│
├── # === models/ 模型子包 ===
├── models/
│   ├── __init__.py              # 模型包初始化
│   ├── LSTM.py                  # LSTM基线模型
│   ├── GAT.py                   # GAT + LSTM
│   ├── GSAGE.py                 # GraphSAGE + LSTM
│   ├── GAT_SeparateEncoder.py   # GAT + 分离式编码器 ⭐新增v3.0
│   └── GSAGE_SeparateEncoder.py # GSAGE + 分离式编码器
│
├── # === graph/ 图结构子包 ===
├── graph/
│   ├── __init__.py              # 图结构包初始化
│   └── distance_graph.py        # 基于距离的图构建模块
│
├── # === explainer/ 可解释性分析子包 === ⭐⭐⭐
├── explainer/
│   ├── __init__.py              # 解释器包初始化
│   ├── explainer_config.py      # 解释器配置
│   ├── hybrid_explainer.py      # 混合解释器(时序+空间)
│   ├── temporal_analyzer.py     # 时序特征分析(Integrated Gradients)
│   ├── spatial_explainer.py     # 空间关系分析(GNNExplainer)
│   ├── gnn_wrapper.py           # GNN层提取器
│   ├── visualize_explainer.py   # 可解释性可视化(11种图表)
│   ├── utils.py                 # 工具函数(季节筛选+注意力分析⭐)
│   ├── README.md                # 可解释性模块概览
│   └── EXPLAINER_USAGE.md       # 详细使用指南
│
├── # === tuner/ 超参数调优子包 === ⭐新增
├── tuner/
│   ├── __init__.py              # 调优包初始化
│   ├── search_space.py          # 搜索空间定义(3种预设)
│   ├── trial_runner.py          # 单次试验执行器
│   └── visualize_tuning.py      # 调优结果可视化
│
├── # === utils/ 工具子包 ===
├── utils/
│   ├── __init__.py              # 工具包初始化
│   └── cartopy_helpers.py       # Cartopy地理绘图辅助
│
├── # === checkpoints/ 训练结果 ===
├── checkpoints/                 # 训练结果保存目录
│   └── {模型名}_{时间戳}/
│       ├── config.txt           # 训练配置
│       ├── metrics.txt          # 评估指标
│       ├── best_model.pth       # 最佳模型权重
│       ├── train_losses.npy     # 训练损失
│       ├── val_losses.npy       # 验证损失
│       ├── loss_curves.png      # 损失曲线图
│       ├── test_predict.npy     # 测试集预测
│       └── test_label.npy       # 测试集标签
│
└── README.md                    # 本文档
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖:**
```
torch>=2.0.0
torch-geometric>=2.3.0
captum>=0.6.0              # 可解释性分析
optuna>=3.0.0              # 超参数调优
cartopy>=0.21.0            # 地理可视化
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
```

### 2. 默认配置训练

```bash
python train.py
```

这将使用默认配置训练模型:
- 模型: GAT_LSTM
- 历史窗口: 7天
- 预测长度: 3天
- 预测目标: 最高气温(tmax)
- 数据集: 2010-2017年真实气象数据(28个站点)

### 3. 修改配置

直接编辑`config.py`文件:

```python
from config import Config, ArchConfig, LossConfig

# 创建配置
config = Config()
arch_config = ArchConfig()
loss_config = LossConfig()

# 修改模型
config.exp_model = 'GAT_SeparateEncoder'  # 使用分离式编码器

# 修改时间窗口
config.hist_len = 14       # 历史窗口14天
config.pred_len = 3        # 预测未来3天

# 修改损失函数
loss_config.loss_type = 'WeightedTrend'  # 加权趋势损失

# 运行: python train.py
```

---

## 🔑 核心特性

### 1. 统一配置管理 (config.py)

所有参数在`config.py`中集中管理,无需命令行参数:

```python
from config import create_config, print_config

# 创建默认配置
config, arch_config, loss_config = create_config()

# 创建自定义配置
config, arch_config, loss_config = create_config(
    loss_type='WeightedTrend',
    batch_size=64,
    lr=0.001
)

# 打印配置信息
print_config(config, arch_config, loss_config)
```

**配置类说明:**

**Config类** - 数据和训练配置:
- 数据路径和划分
- 时间窗口(hist_len, pred_len)
- 特征选择和编码
- 图结构配置
- 训练超参数(batch_size, lr, epochs等)

**ArchConfig类** - 模型架构配置:
- 通用参数(hid_dim, MLP_layer, dropout等)
- GAT参数(GAT_layer, heads等)
- SAGE参数(SAGE_layer, aggr等)
- LSTM参数(lstm_num_layers等)
- 分离式编码器参数 ⭐新增

**LossConfig类** - 损失函数配置:
- 损失函数类型选择
- 加权趋势损失参数
- 多阈值加权参数

### 2. 4维时间周期编码 (dataset.py)

自动将时间特征(doy, month)转换为周期性编码:

```python
原始时间特征:
  - doy (1-366)       # 年内日序数
  - month (1-12)      # 月份

转换为4维sin/cos编码:
  - doy_sin, doy_cos      # 年周期
  - month_sin, month_cos  # 月周期

最终输入维度: 26 (基础特征) + 4 (时间编码) = 30
```

**数据加载器创建:**

```python
from dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    config, graph, MetData,
    batch_size=32,
    shuffle_train=True
)
```

### 3. 灵活的特征选择

```python
# 使用所有基础特征（默认）
config.feature_indices = None  # 自动使用0-25（移除doy和month）

# 自定义特征选择（例如：仅使用核心气象特征）
config.feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 4. 多种模型支持

| 模型类型 | exp_model 值 | 说明 |
|---------|-------------|------|
| **基础模型** | | |
| LSTM基线 | `'LSTM'` | 纯LSTM模型（无图结构） |
| GAT + LSTM | `'GAT_LSTM'` | 图注意力网络 + LSTM时序建模 |
| GraphSAGE + LSTM | `'GSAGE_LSTM'` | GraphSAGE + LSTM时序建模 |
| **进阶模型** ⭐ | | |
| GAT + 分离式编码器 | `'GAT_SeparateEncoder'` | 特征级交叉注意力+节点嵌入 v3.0 |
| GSAGE + 分离式编码器 | `'GSAGE_SeparateEncoder'` | SAGE版分离式编码器 |

**GAT_SeparateEncoder v3.0 核心创新:**
- 静态/动态特征分离编码
- 可学习节点嵌入(捕获气象站固有特性)
- 特征级交叉注意力融合
- 残差连接增强信息流

**使用方法:**

```python
# 编辑 config.py
config.exp_model = 'GAT_SeparateEncoder'

# 配置分离式编码器
arch_config.use_separate_encoder = True
arch_config.static_feature_indices = [0, 1, 2, 10, 11, 12, 13]  # 静态特征索引
arch_config.use_node_embedding = True
arch_config.embedding_dim = 16
arch_config.use_cross_attention = True
arch_config.num_cross_attention_heads = 4
arch_config.use_feature_residual = True
```

### 5. 自适应损失函数 (losses.py) ⭐

**WeightedTrendMSELoss** - 加权趋势损失(推荐):

基于论文《基于注意力机制与加权趋势损失的风速订正方法》(刘旭等, 2025)。

```python
损失函数: L = (1-α) × WeightedMSE + α × TrendMSE

WeightedMSE: 对高温样本加权的均方误差
  - 漏报高温(y>T, ŷ<T): 权重 c_under (最大)
  - 误报高温(y<T, ŷ>T): 权重 c_over (中等)
  - 正确预报高温(y>T, ŷ>T): 权重 c_default_high (基准)

TrendMSE: 趋势一致性约束

使用方法:
loss_config.loss_type = 'WeightedTrend'
loss_config.alert_temp = 35.0      # 高温警戒阈值(°C)
loss_config.c_under = 4            # 漏报权重系数
loss_config.c_over = 1.5           # 误报权重系数
loss_config.trend_weight = 0.5     # 趋势权重
```

**其他损失函数:**
- `'MSE'` - 标准均方误差

### 6. 灵活的图构建 (graph/distance_graph.py)

支持4种图构建方法:

| 图类型 | graph_type | 特点 | 适用场景 |
|--------|-----------|------|---------|
| **K近邻逆距离图** (推荐) | `'inv_dis'` | 边权重=1/distance | 通用,默认推荐 |
| **空间相似性图** | `'spatial_similarity'` | 综合距离和特征相似性 | 特征相似性重要 |
| **K近邻图** | `'knn'` | 无边权重 | 简单快速 |
| **全连接图** | `'full'` | 所有节点连接 | 小规模节点(<50) |

**使用方法:**

```python
from graph.distance_graph import create_graph_from_config

# 配置图类型
config.graph_type = 'inv_dis'
config.top_neighbors = 10
config.use_edge_attr = True

# 创建图结构
graph = create_graph_from_config(config, station_info)
```

### 7. 可解释性分析 (explainer/) ⭐⭐⭐

**完整的模型可解释性分析框架,包含时序和空间两个维度。**

**核心组件:**
- `HybridExplainer` - 混合解释器,整合时序和空间分析
- `TemporalAnalyzer` - 时序特征重要性分析(Integrated Gradients)
- `SpatialExplainer` - 空间关系重要性分析(GNNExplainer)
- `GNNWrapper` - 从完整模型提取纯GNN部分

**生成11种专业可视化图表:**
1. 时序特征热图
2. 空间边地理图(Top-K)
3. 全边叠加图
4. 全边分离图
5. GNNExplainer vs GAT注意力对比
6. 边重要性分布
7. 时间步重要性
8. 特征重要性排名
9. 全局注意力矩阵热力图 ⭐
10. 距离-注意力散点图 ⭐
11. 温度相关性-注意力散点图 ⭐

**快速使用:**

```python
from explainer import HybridExplainer, ExplainerConfig

# 配置解释器
exp_config = ExplainerConfig(
    num_samples=100,
    season='summer',           # 季节筛选
    extract_attention=True,    # 提取GAT注意力
)

# 运行完整分析
explainer = HybridExplainer(model, config, exp_config)
explanation = explainer.explain_full(
    test_loader,
    save_path='checkpoints/model/explanations/'
)
```

**命令行使用:**

```bash
# 基本分析
python explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --visualize

# 夏季分析
python explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --season summer \
    --visualize
```

详细文档: [explainer/README.md](explainer/README.md)

### 8. 超参数自动调优 (tune.py + tuner/) ⭐新增

**基于Optuna的贝叶斯优化框架,自动搜索最优超参数组合。**

**核心特性:**
- 3种预设搜索空间(quick/default/comprehensive)
- TPE (Tree-structured Parzen Estimator) 采样器
- Median Pruner早停剪枝
- 结果可视化(优化历史、参数重要性、并行坐标图)
- SQLite数据库持久化存储

**快速使用:**

```bash
# 快速模式(10次试验)
python tune.py --mode quick --n_trials 10

# 标准模式(50次试验)
python tune.py --mode default --n_trials 50

# 综合模式(100次试验)
python tune.py --mode comprehensive --n_trials 100

# 查看最佳结果
cat ../tuning_results/best_config.json
```

**包含的超参数:**
- 数据参数: hist_len, pred_len
- 训练参数: batch_size, lr, weight_decay, optimizer
- 模型架构: hid_dim, MLP_layer, GAT_layer/SAGE_layer, heads, dropout
- 图结构: graph_type, top_neighbors
- 损失函数: loss_type, alert_temp, c_under, c_over

**输出结果:**
```
tuning_results/
├── optuna_study.db           # Optuna数据库
├── best_config.json          # 最佳配置
├── trials_dataframe.csv      # 所有试验记录
└── visualizations/
    ├── optimization_history.png
    ├── param_importances.png
    └── parallel_coordinate.png
```

---

## 📊 配置说明

### Config类（数据和训练配置）

#### 数据配置
```python
config.MetData_fp           # 数据文件路径
config.station_info_fp      # 气象站信息路径
config.dataset_num          # 数据集标识
```

#### 时间窗口配置
```python
config.hist_len = 7         # 历史窗口长度（天）
config.pred_len = 1         # 预测长度（天）
```

#### 特征配置
```python
config.target_feature_idx = 4        # 预测目标索引（4=tmax最高气温）
config.feature_indices = None        # 特征选择（None=使用0-25）
config.add_temporal_encoding = True  # 是否添加时间编码
```

#### 数据集划分（按年份）
```python
config.train_start = 0      # 训练集起始（2010年）
config.train_end = 2191     # 训练集结束（2015年）
config.val_start = 2191     # 验证集起始（2016年）
config.val_end = 2557       # 验证集结束
config.test_start = 2557    # 测试集起始（2017年）
config.test_end = 2922      # 测试集结束
```

#### 图结构配置
```python
# 图类型选择
config.graph_type = 'inv_dis'  # 'inv_dis', 'spatial_similarity', 'knn', 'full'

# K近邻参数
config.top_neighbors = 10      # K近邻数量
config.use_edge_attr = True   # 是否使用边权重

# 空间相似性图参数
config.spatial_sim_top_k = 15              # 选择最相似的K个邻居
config.spatial_sim_alpha = 1.0             # 邻域相似性权重系数
```

#### 训练配置
```python
config.batch_size = 32      # 批次大小
config.epochs = 500         # 最大训练轮数
config.lr = 0.001           # 学习率
config.weight_decay = 1e-3  # 权重衰减
config.early_stop = 30      # 早停耐心值

# 优化器选择
config.optimizer = 'Adam'   # 'Adam', 'AdamW', 'SGD', 'RMSprop'

# 学习率调度器
config.scheduler = 'CosineAnnealingLR'  # 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'None'
config.T_max = 50           # CosineAnnealingLR周期
config.eta_min = 1e-6       # 最小学习率
```

### ArchConfig类（模型架构配置）

#### 通用参数
```python
arch_config.hid_dim = 16             # 隐藏层维度
arch_config.MLP_layer = 1            # MLP层数
arch_config.AF = 'ReLU'              # 激活函数
arch_config.norm_type = 'LayerNorm'  # 规范化类型
arch_config.dropout = True           # 是否使用Dropout
```

#### GAT特定参数
```python
arch_config.GAT_layer = 1            # GAT层数
arch_config.heads = 2                # 注意力头数
arch_config.intra_drop = 0.1         # 层内Dropout
arch_config.inter_drop = 0.1         # 层间Dropout
```

#### SAGE特定参数
```python
arch_config.SAGE_layer = 3           # SAGE层数
arch_config.aggr = 'mean'            # 聚合方式（'mean', 'max', 'add'）
```

#### LSTM特定参数
```python
arch_config.lstm_num_layers = 2      # LSTM层数
arch_config.lstm_dropout = 0.1       # LSTM Dropout
arch_config.lstm_bidirectional = False  # 是否双向
```

#### 分离式编码器参数 ⭐新增
```python
arch_config.use_separate_encoder = False        # 是否使用分离式编码器
arch_config.static_feature_indices = None       # 静态特征索引列表
arch_config.use_node_embedding = True           # 是否使用节点嵌入
arch_config.embedding_dim = 16                  # 节点嵌入维度
arch_config.use_cross_attention = True          # 是否使用交叉注意力
arch_config.num_cross_attention_heads = 4       # 交叉注意力头数
arch_config.use_feature_residual = True         # 是否使用特征残差连接
```

### LossConfig类（损失函数配置）

```python
# 损失函数类型(核心配置)
loss_config.loss_type = 'MSE'  # 'MSE', 'WeightedTrend'

# 加权趋势损失参数
loss_config.alert_temp = 35.0        # 高温警戒阈值(°C)
loss_config.c_under = 4              # 漏报权重系数(低估高温)
loss_config.c_over = 1.5             # 误报权重系数(高估)
loss_config.c_default_high = 1.0     # 正确预报高温权重
loss_config.trend_weight = 0.5       # 趋势权重
```

---

## 📝 使用示例

### 示例1: 修改时间窗口

```python
from config import Config, ArchConfig, LossConfig

config = Config()
arch_config = ArchConfig()
loss_config = LossConfig()

# 修改时间窗口
config.hist_len = 14  # 使用过去14天
config.pred_len = 3   # 预测未来3天

# 运行: python train.py
```

### 示例2: 预测最低气温

```python
config = Config()
arch_config = ArchConfig()

# 修改预测目标
config.target_feature_idx = 3  # 3=tmin（最低气温）

# 运行: python train.py
```

### 示例3: 特征选择

```python
config = Config()
arch_config = ArchConfig()

# 只使用核心气象特征（0-8）
config.feature_indices = list(range(9))
# 0-1: x, y (经纬度)
# 2: height (海拔)
# 3-5: tmin, tmax, tave
# 6-9: pre, prs, rh, win

# 输入维度自动计算: 9 (基础) + 4 (时间) = 13
```

### 示例4: 使用GAT模型

```python
config = Config()
arch_config = ArchConfig()

# 切换模型
config.exp_model = 'GAT_LSTM'

# GAT特定参数
arch_config.GAT_layer = 3
arch_config.heads = 8

# 运行: python train.py
```

### 示例5: 使用分离式编码器模型 ⭐

```python
config = Config()
arch_config = ArchConfig()

# 使用GAT_SeparateEncoder模型
config.exp_model = 'GAT_SeparateEncoder'

# 配置分离式编码器
arch_config.use_separate_encoder = True
arch_config.static_feature_indices = [0, 1, 2, 10, 11, 12, 13]  # 静态特征
arch_config.use_node_embedding = True
arch_config.embedding_dim = 16
arch_config.use_cross_attention = True
arch_config.num_cross_attention_heads = 4
arch_config.use_feature_residual = True

# 运行: python train.py
```

### 示例6: 使用加权趋势损失

```python
config = Config()
arch_config = ArchConfig()
loss_config = LossConfig()

# 启用加权趋势损失
loss_config.loss_type = 'WeightedTrend'
loss_config.alert_temp = 35.0
loss_config.c_under = 4
loss_config.c_over = 1.5
loss_config.trend_weight = 0.5

# 运行: python train.py
```

### 示例7: 可解释性分析

```python
from explainer import HybridExplainer, ExplainerConfig

# 配置解释器
exp_config = ExplainerConfig(
    num_samples=100,
    season='summer'  # 只分析夏季样本
)

explainer = HybridExplainer(model, config, exp_config)
explanation = explainer.explain_full(
    test_loader,
    save_path='explanations/summer/'
)

# 查看最重要的边
print("Top-5重要边:")
for src, dst, imp in explanation['spatial']['important_edges'][:5]:
    print(f"站点{src} → 站点{dst}: {imp:.4f}")
```

### 示例8: 超参数自动调优

```bash
# 运行超参数调优
python tune.py --mode default --n_trials 50

# 查看最佳配置
cat ../tuning_results/best_config.json

# 将最佳参数应用到config.py
# 然后运行训练
python train.py
```

---

## 📊 数据格式

### 输入数据（real_weather_data_2010_2017.npy）

**形状:** `[2922, 28, 28]`
- 维度0: 时间步（2922天）
- 维度1: 气象站（28个）
- 维度2: 特征（28个）

**特征列表（索引0-27）:**

| 索引 | 特征名 | 说明 | 单位 |
|:----:|--------|------|:----:|
| 0-1 | x, y | 经纬度 | ° |
| 2 | height | 海拔高度 | m |
| 3-5 | tmin, tmax, **tave** | 温度（**预测目标**） | °C |
| 6-9 | pre, prs, rh, win | 气象要素 | - |
| 10-17 | BH, BHstd, SCD, PLA, λp, λb, POI, POW, POV | 城市环境 | - |
| 18 | NDVI | 植被指数 | - |
| 19-20 | surface_pressure, surface_solar_radiation | ERA5数据 | - |
| 21-22 | u_wind, v_wind | 风速分量 | m/s |
| 23-24 | VegHeight_mean, VegHeight_std | 植被高度 | m |
| 25-26 | **doy, month** | **时间特征（转换为4维sin/cos）** | - |

### 气象站信息（station_info.npy）

**形状:** `[28, 4]`
- 列0: 站点ID
- 列1: 经度
- 列2: 纬度
- 列3: 海拔高度

---

## 🐛 修复的Bug

相比`origin_gnn`，本框架修复了以下问题:

### 1. whichAF函数参数不匹配
- **原问题**: `whichAF(AF, hid_dim)` 调用，但函数只接受1个参数
- **修复**: 移除多余的`hid_dim`参数

### 2. GAT模型返回值错误
- **原问题**: `return x, (edge_idx, attention)` 导致network_GNN无法处理
- **修复**: 只返回`x`

### 3. GAT循环索引错误
- **原问题**: 嵌套循环中索引计算错误
- **修复**: 使用`base_idx = i * self.element`统一计算

### 4. network_GNN中if/if结构
- **原问题**: 两个独立的`if`判断`use_edge_attr`
- **修复**: 改为`if/elif`结构

### 5. 未实现的模型引用
- **原问题**: 引用了不存在的`GC_LSTM`模型
- **修复**: 移除未实现的模型引用

---

## 🔍 常见问题

### Q1: 如何修改配置参数？

直接编辑`config.py`文件中的`Config`和`ArchConfig`类，或在训练脚本中创建配置对象后修改。

### Q2: 如何添加新的特征？

1. 更新`data/`目录下的数据文件
2. 修改`config.py`中的`base_feature_dim`
3. 在`dataset.py`中更新时间特征的索引位置

### Q3: 如何使用自己的数据？

1. 准备NPY格式数据：`[time_steps, num_stations, features]`
2. 准备气象站信息：`[num_stations, 4]` (ID, 经度, 纬度, 海拔)
3. 修改`config.py`中的文件路径和参数

### Q4: 训练过程中显存不足怎么办？

减小以下参数：
- `config.batch_size`
- `arch_config.hid_dim`
- `arch_config.GAT_layer` 或 `arch_config.SAGE_layer`
- `config.hist_len`

### Q5: 如何实现模型集成？

训练多个模型后进行集成预测:

```python
# 训练多个模型
models = ['GAT_LSTM', 'GSAGE_LSTM', 'GAT_SeparateEncoder']
predictions = []

for model_name in models:
    # 修改config.exp_model并训练
    # 保存预测结果
    predictions.append(test_predict)

# 集成预测（简单平均）
ensemble_pred = np.mean(predictions, axis=0)
```

### Q6: 如何查看模型的注意力权重？

使用可解释性分析模块:

```bash
python explain_model.py \
    --model_path checkpoints/model/best_model.pth \
    --extract_attention \
    --visualize
```

### Q7: 超参数调优需要多长时间？

- **quick模式** (10次试验): 约30分钟 - 1小时
- **default模式** (50次试验): 约3-5小时
- **comprehensive模式** (100次试验): 约6-10小时

实际时间取决于硬件配置和数据集大小。

---

## 📚 相关文档

- **项目总览**: [../README.md](../README.md)
- **项目架构**: [../CLAUDE.md](../CLAUDE.md) ⭐ 最详细的架构说明
- **可解释性分析**: [explainer/README.md](explainer/README.md)
- **可解释性使用指南**: [explainer/EXPLAINER_USAGE.md](explainer/EXPLAINER_USAGE.md)
- **数据格式**: [../DATA_FORMAT.md](../DATA_FORMAT.md)

---

## 📄 许可证

本框架遵循原项目的许可协议。

---

## 📋 更新日志

### v3.0 (2025-12-16)
- ✨ 新增超参数自动调优模块(Optuna)
- 🔑 新增GAT_SeparateEncoder v3.0模型
- 🎯 可解释性分析增强(11种可视化)
- 📊 新增加权趋势损失函数
- ⚙️ 统一配置管理优化

### v2.0 (2025-11)
- 新增可解释性分析模块
- 新增自适应损失函数
- 新增空间相似性图构建
- 优化配置管理
- 更新默认参数

### v1.0 (初始版本)
- 基于origin_gnn重构
- 修复已知Bug
- 统一配置管理
- 4维时间编码

---

**最后更新:** 2025-12-16
**维护者:** GNN气温预测项目组
