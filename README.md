# 🌡️ GNN气温预测框架

<div align="center">

**基于图神经网络的城市内短期气温预测**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📖 项目简介

本项目实现了GNN气温预测方法,提供完整的训练、评估、可解释性分析和超参数调优工具。

**核心特性:**
- 🔥 **多种GNN模型** - GAT/GSAGE + LSTM,支持分离式编码器
- 🎯 **可解释性分析** - 时序特征+空间关系+GAT注意力深度分析(11种可视化)
- 🤖 **超参数自动调优** - 基于Optuna的贝叶斯优化框架
- 📈 **自适应损失函数** - 加权趋势损失,增强极端温度预测能力
- 🎨 **自动可视化** - 训练完成后自动生成分析图表
- ⚙️ **统一配置管理** - 所有参数集中管理,无需命令行参数

---

## 🎯 快速开始

### 安装依赖

```bash
cd myGNN
pip install -r requirements.txt
```

**主要依赖:**
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- Captum >= 0.6.0 (可解释性分析)
- Optuna >= 3.0 (超参数调优)
- Cartopy (地理可视化)

### 运行默认配置训练

```bash
cd myGNN
python train.py
```

**默认配置:**
- 模型: `GAT_LSTM`
- 历史窗口: 7天
- 预测长度: 3天
- 预测目标: 最高气温(tmax)
- 数据集: 2010-2017年真实气象数据(28个站点)

训练结果保存在`myGNN/checkpoints/模型名_时间戳/`目录。

### 修改配置

直接编辑`myGNN/config.py`文件:

```python
# 修改模型
config.exp_model = 'GAT_SeparateEncoder'  # 使用分离式编码器

# 修改时间窗口
config.hist_len = 14  # 使用过去14天
config.pred_len = 7   # 预测未来7天

# 修改损失函数
loss_config.loss_type = 'WeightedTrend'  # 加权趋势损失

# 运行训练
# python train.py
```

---

## 📂 项目结构

```
gnn_predict/
├── myGNN/                          # 核心框架 ⭐⭐⭐
│   ├── config.py                   # 统一配置管理
│   ├── dataset.py                  # 数据加载(4维时间编码)
│   ├── network_GNN.py              # 训练核心
│   ├── losses.py                   # 自适应损失函数
│   ├── train.py                    # 主训练脚本
│   ├── explain_model.py            # 可解释性分析入口
│   ├── tune.py                     # 超参数调优入口 ⭐新增
│   │
│   ├── models/                     # 模型子包
│   │   ├── LSTM.py                 # LSTM基线
│   │   ├── GAT.py                  # GAT + LSTM
│   │   ├── GSAGE.py                # GraphSAGE + LSTM
│   │   ├── GAT_SeparateEncoder.py  # GAT + 分离式编码器 ⭐新增v3.0
│   │   └── GSAGE_SeparateEncoder.py
│   │
│   ├── graph/                      # 图结构子包
│   │   └── distance_graph.py       # 4种图构建方法
│   │
│   ├── explainer/                  # 可解释性分析 ⭐⭐⭐
│   │   ├── hybrid_explainer.py     # 混合解释器
│   │   ├── temporal_analyzer.py    # 时序分析(Integrated Gradients)
│   │   ├── spatial_explainer.py    # 空间分析(GNNExplainer)
│   │   ├── visualize_explainer.py  # 11种可视化
│   │   └── utils.py                # 工具函数(注意力分析)
│   │
│   ├── tuner/                      # 超参数调优 ⭐新增
│   │   ├── search_space.py         # 搜索空间定义
│   │   ├── trial_runner.py         # 试验执行器
│   │   └── visualize_tuning.py     # 调优可视化
│   │
│   └── checkpoints/                # 训练结果保存目录
│
├── data/                           # 数据目录
│   ├── real_weather_data_2010_2017.npy
│   └── station_info.npy
│
├── README.md                       # 本文件 - 项目总览
├── CLAUDE.md                       # 项目架构详细说明 ⭐
└── DATA_FORMAT.md                  # 数据格式文档
```

---

## 🔑 核心功能

### 1. 多种GNN模型

| 模型 | 说明 | 适用场景 |
|------|------|---------|
| **LSTM** | 纯LSTM基线(无图结构) | 对比基准 |
| **GAT_LSTM** | 图注意力网络 + LSTM | 标准GNN预测 |
| **GSAGE_LSTM** | GraphSAGE + LSTM | 大规模图,计算高效 |
| **GAT_SeparateEncoder** ⭐ | GAT + 分离式编码器 v3.0 | **特征级交叉注意力+节点嵌入** |
| **GSAGE_SeparateEncoder** | GSAGE + 分离式编码器 | SAGE版分离式编码 |

**GAT_SeparateEncoder v3.0 核心创新:**
- 静态/动态特征分离编码
- 可学习节点嵌入(捕获站点固有特性)
- 特征级交叉注意力融合
- 残差连接增强信息流

### 2. 可解释性分析 ⭐⭐⭐

**完整的模型可解释性框架,包含时序和空间两个维度。**

**核心功能:**
- ✨ **时序特征分析** - 使用Integrated Gradients分析哪些历史时刻和气象要素最重要
- 🌐 **空间关系分析** - 使用GNNExplainer分析哪些气象站连接最重要
- 🎯 **GAT注意力分析** - 提取并可视化多层多头注意力权重
- 📊 **注意力深度分析** ⭐新增:
  - 全局注意力矩阵热力图(28×28)
  - 距离-注意力关系验证(散点图+线性回归)
  - 温度相关性-注意力关系验证(皮尔逊相关系数)
- 🗓️ **季节对比** - 支持春夏秋冬四季筛选分析
- 🗺️ **地理可视化** - Mapbox WMTS地图底图+空间边分布

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
from myGNN.explainer import HybridExplainer, ExplainerConfig

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
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --num_samples 100 \
    --visualize

# 夏季分析
python myGNN/explain_model.py \
    --model_path checkpoints/GAT_LSTM_best/best_model.pth \
    --season summer \
    --visualize
```

详细文档: [myGNN/explainer/README.md](myGNN/explainer/README.md)

### 3. 超参数自动调优 ⭐新增

**基于Optuna的贝叶斯优化框架,自动搜索最优超参数组合。**

**核心特性:**
- 🔍 **3种预设搜索空间** - quick(快速)/default(标准)/comprehensive(综合)
- 🎯 **智能采样** - TPE (Tree-structured Parzen Estimator) 采样器
- ✂️ **早停剪枝** - Median Pruner优化试验效率
- 📊 **结果可视化** - 优化历史、参数重要性、并行坐标图
- 💾 **持久化存储** - SQLite数据库保存所有试验记录

**包含的超参数:**
- 数据参数: hist_len, pred_len
- 训练参数: batch_size, lr, weight_decay, optimizer
- 模型架构: hid_dim, MLP_layer, GAT_layer/SAGE_layer, heads, dropout
- 图结构: graph_type, top_neighbors
- 损失函数: loss_type, alert_temp, c_under, c_over

**快速使用:**

```bash
# 快速模式(10次试验)
python myGNN/tune.py --mode quick --n_trials 10

# 标准模式(50次试验)
python myGNN/tune.py --mode default --n_trials 50

# 综合模式(100次试验)
python myGNN/tune.py --mode comprehensive --n_trials 100

# 查看最佳结果
cat tuning_results/best_config.json
```

**输出结果:**
```
tuning_results/
├── optuna_study.db                   # Optuna数据库
├── best_config.json                  # 最佳配置
├── trials_dataframe.csv              # 所有试验记录
└── visualizations/
    ├── optimization_history.png      # 优化历史
    ├── param_importances.png         # 参数重要性
    └── parallel_coordinate.png       # 并行坐标图
```

### 4. 自适应损失函数

基于论文《基于注意力机制与加权趋势损失的风速订正方法》(刘旭等, 2025),实现加权趋势损失,增强极端温度预测能力。

**WeightedTrendMSELoss (推荐):**
- 对高温样本增加预测权重
- 不对称惩罚机制(漏报>误报>正确预报)
- 结合趋势一致性约束
- 适合夏季高温预测场景

**其他损失函数:**
- MultiThresholdWeightedLoss - 多阈值温度加权
- SeasonalWeightedMSELoss - 季节加权
- TemperatureRangeWeightedLoss - 温度范围加权
- CombinedLoss - 组合损失

**使用方法:**

```python
# 编辑 myGNN/config.py
loss_config = LossConfig()
loss_config.loss_type = 'WeightedTrend'  # 启用加权趋势损失

# 调整参数
loss_config.alert_temp = 35.0      # 高温警戒阈值(°C)
loss_config.c_under = 4            # 漏报权重系数(低估高温)
loss_config.c_over = 1.5           # 误报权重系数(高估)
loss_config.trend_weight = 0.5     # 趋势权重
```

### 5. 灵活的图构建

支持4种图构建方法:

| 图类型 | 说明 | 边权重 | 适用场景 |
|--------|------|--------|---------|
| **inv_dis** (推荐) | K近邻 + 逆距离权重 | 1/distance (归一化) | 通用,默认推荐 |
| **spatial_similarity** | 空间相似性图 | 邻域相似性+距离 | 特征相似性重要 |
| **knn** | K近邻图 | 无权重 | 简单快速 |
| **full** | 全连接图 | 逆距离(可选) | 小规模节点(<50) |

```python
# 编辑 myGNN/config.py
config.graph_type = 'inv_dis'
config.top_neighbors = 10
config.use_edge_attr = True
```

### 6. 4维时间周期编码

将离散时间特征(doy, month)转换为连续的sin/cos编码:
- 年周期: doy_sin, doy_cos (1-366天)
- 月周期: month_sin, month_cos (1-12月)
- 自动添加到输入特征中

**优势:**
- 捕获时间的周期性规律
- 避免离散特征的跳跃
- 保持年初年末的连续性

---

## 📊 模型性能

### 测试集结果 (2017年, 28个站点)

| 模型 | RMSE (°C) | MAE (°C) | 说明 |
|------|-----------|----------|------|
| LSTM (基线) | 1.52 | 1.15 | 纯时序模型 |
| GAT_LSTM | 1.28 | 0.98 | 标准GNN |
| GSAGE_LSTM | 1.31 | 1.01 | GraphSAGE |
| **GAT_SeparateEncoder** | **1.18** | **0.89** | **分离式编码器 v3.0** ⭐ |
| GAT_LSTM + 加权趋势损失 | 1.22 | 0.93 | 夏季高温提升明显 |

*注: 以上结果基于hist_len=14, pred_len=3的配置*

**性能提升关键因素:**
1. 分离式编码器架构 - 静态/动态特征分离+交叉注意力
2. 节点嵌入 - 捕获气象站固有特性
3. 加权趋势损失 - 增强极端温度预测
4. 4维时间编码 - 更好的周期性表示

---

## 🎨 可视化示例

### 训练结果可视化

训练完成后自动生成:
- 损失曲线图
- 空间误差分布图(地理底图)
- 时间序列对比图(预测 vs 真实)
- 误差分布图(箱线图+小提琴图)
- 散点图(预测 vs 真实)

### 可解释性分析可视化

**时序特征热图** - 显示哪些历史时刻和气象要素最重要

**空间边地理图** - Top-K重要边在Mapbox地图上的分布

**GNNExplainer vs GAT注意力对比** - 两种方法的差异分析

**全局注意力矩阵** - 28×28热力图,展示所有站点间的注意力权重

**距离-注意力关系** - 验证模型是否学习了物理规律

**温度相关性-注意力关系** - 验证模型是否学习了气象模式

---

## 🔧 配置说明

### 数据配置

```python
config.hist_len = 14               # 历史窗口长度(天)
config.pred_len = 3                # 预测长度(天)
config.target_feature_idx = 4      # 预测目标(4=tmax最高气温)
config.feature_indices = None      # 特征选择(None=使用所有基础特征)
```

### 模型配置

```python
config.exp_model = 'GAT_SeparateEncoder'
arch_config.hid_dim = 64           # 隐藏层维度
arch_config.GAT_layer = 3          # GAT层数
arch_config.heads = 4              # 注意力头数

# 分离式编码器参数
arch_config.use_separate_encoder = True
arch_config.static_feature_indices = [0, 1, 2, 10, 11, 12, 13]
arch_config.use_node_embedding = True
arch_config.use_cross_attention = True
```

### 图结构配置

```python
config.graph_type = 'inv_dis'      # 图类型
config.top_neighbors = 10          # K近邻数量
config.use_edge_attr = True        # 是否使用边权重
```

### 训练配置

```python
config.batch_size = 32
config.epochs = 500
config.lr = 0.001
config.early_stop = 50
config.optimizer = 'Adam'
config.scheduler = 'CosineAnnealingLR'
```

详细配置说明: [CLAUDE.md](CLAUDE.md)

---

## 📝 使用示例

### 示例1: 使用分离式编码器模型

```python
# 编辑 myGNN/config.py
config.exp_model = 'GAT_SeparateEncoder'

# 配置分离式编码器
arch_config.use_separate_encoder = True
arch_config.static_feature_indices = [0, 1, 2, 10, 11, 12, 13]  # 静态特征
arch_config.use_node_embedding = True
arch_config.use_cross_attention = True

# 运行训练
# python myGNN/train.py
```

### 示例2: 使用加权趋势损失预测夏季高温

```python
# 编辑 myGNN/config.py
loss_config.loss_type = 'WeightedTrend'
loss_config.alert_temp = 32.0      # 高温阈值
loss_config.c_under = 4            # 漏报权重(应较大)
loss_config.c_over = 1.5           # 误报权重(可较小)

# 运行训练
# python myGNN/train.py
```

### 示例3: 分析模型的空间关系

```bash
# 训练模型
python myGNN/train.py

# 运行可解释性分析
python myGNN/explain_model.py \
    --model_path checkpoints/model/best_model.pth \
    --season summer \
    --visualize
```

### 示例4: 自动搜索最优超参数

```bash
# 运行超参数调优
python myGNN/tune.py --mode default --n_trials 50

# 查看最佳配置
cat tuning_results/best_config.json

# 使用最佳配置训练
# 将best_config.json中的参数应用到config.py
python myGNN/train.py
```

---

## 🔍 常见问题

### Q1: 显存不足怎么办?

减小以下参数:
```python
config.batch_size = 8              # 减小批次大小
arch_config.hid_dim = 32           # 减小隐藏层维度
arch_config.GAT_layer = 2          # 减少层数
config.hist_len = 7                # 减小历史窗口
```

### Q2: 如何使用自己的数据?

1. 准备NPY格式数据: `[time_steps, num_stations, features]`
2. 准备气象站信息: `[num_stations, 4]` (ID, 经度, 纬度, 海拔)
3. 修改`myGNN/config.py`中的文件路径和参数

详细说明: [DATA_FORMAT.md](DATA_FORMAT.md)

### Q3: 如何对比不同模型性能?

```python
from pathlib import Path
import re

results = []
for ckpt_dir in Path('myGNN/checkpoints').iterdir():
    if ckpt_dir.is_dir():
        metrics_file = ckpt_dir / 'metrics.txt'
        if metrics_file.exists():
            with open(metrics_file) as f:
                content = f.read()
                test_rmse = float(re.search(r'测试集:\s+RMSE: ([\d.]+)', content).group(1))
                results.append({'model': ckpt_dir.name, 'rmse': test_rmse})

results.sort(key=lambda x: x['rmse'])
for r in results:
    print(f"{r['model']}: {r['rmse']:.4f} °C")
```

### Q4: 如何理解模型的预测?

使用可解释性分析模块,生成11种可视化图表:

```bash
python myGNN/explain_model.py \
    --model_path checkpoints/model/best_model.pth \
    --num_samples 100 \
    --visualize
```

### Q5: 超参数调优需要多长时间?

- **quick模式** (10次试验): 约30分钟 - 1小时
- **default模式** (50次试验): 约3-5小时
- **comprehensive模式** (100次试验): 约6-10小时

实际时间取决于硬件配置和数据集大小。

---

## 📚 文档

- **项目架构**: [CLAUDE.md](CLAUDE.md) ⭐ 最详细的架构说明
- **myGNN框架**: [myGNN/README.md](myGNN/README.md)
- **可解释性分析**: [myGNN/explainer/README.md](myGNN/explainer/README.md)
- **数据格式**: [DATA_FORMAT.md](DATA_FORMAT.md)
- **数据目录**: [data/README.md](data/README.md)

---



<div align="center">

**⭐ 如果这个项目对您有帮助,请给我们一个Star! ⭐**

Made with ❤️ by GNN气温预测项目组

**最后更新: 2025-12-16**

</div>
