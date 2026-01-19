# 🌡️ GNN气温预测框架

> 基于图神经网络(GNN)的短期气温预测系统 | Graph Neural Network Framework for Short-term Temperature Forecasting

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

一个用于城市尺度气温预测的图神经网络框架,专门针对中国华南地区28个气象站点的短期气温预测任务设计,结合真实气象观测数据(2010-2017年)和空间图结构建模,实现多步气温预测。

---

## ✨ 核心特性

- 🎯 **统一配置管理** - 所有参数集中在 [config.py](myGNN/config.py) 管理,无需命令行参数
- 🔄 **4维时间周期编码** - 自动将时间特征转换为sin/cos周期性编码,更好地捕获季节性规律
- 🗺️ **多种图构建策略** - 支持K近邻、空间相似性、逆距离权重等多种图拓扑
- 🧠 **多样化模型架构** - GAT、GraphSAGE、分离式编码器等6种模型可选
- 📊 **加权趋势损失函数** - 针对高温预测场景设计的自适应损失函数
- 🔍 **可解释性分析** - 完整的时序+空间可解释性分析框架(11种可视化)
- ⚡ **超参数自动调优** - 基于Optuna的贝叶斯优化框架
- 📈 **自动结果可视化** - 训练完成后自动生成损失曲线、预测对比图等

---

## 📊 项目概览

### 数据特征

| 项目 | 说明 |
|------|------|
| **时间范围** | 2010-2017年 (8年完整数据) |
| **空间范围** | 中国华南地区 28个气象站 |
| **数据来源** | 真实气象观测 + ERA5再分析 + 植被数据 |
| **特征维度** | 28个原始特征 → 26个基础特征 + 4维时间编码 = 30维输入 |
| **预测目标** | 日最高气温(tmax) / 日平均气温(tave) |

### 数据集划分

| 数据集 | 年份 | 天数 | 用途 |
|-------|------|------|------|
| **训练集** | 2010-2015 | 2191天 | 模型训练 |
| **验证集** | 2016 | 366天 | 超参数调优 |
| **测试集** | 2017 | 365天 | 性能评估 |

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选,用于GPU加速)
- 8GB+ RAM (推荐16GB)

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/gnn_predict.git
cd gnn_predict

# 安装依赖
pip install -r myGNN/requirements.txt
```

**核心依赖:**
```
torch>=2.0.0                # PyTorch深度学习框架
torch-geometric>=2.3.0      # PyG图神经网络库
captum>=0.6.0               # 可解释性分析
numpy>=1.24.0               # 数值计算
matplotlib>=3.7.0           # 可视化
scipy>=1.10.0               # 科学计算
```

### 数据准备

```bash
# 进入数据目录
cd data

# 运行数据转换脚本(将CSV转为NPY格式)
python convert_real_data.py
```

**输出文件:**
- `real_weather_data_2010_2017.npy` - 主数据数组 [2922天, 28站点, 28特征]
- `station_info.npy` - 气象站信息 [28站点, 4属性]

### 开始训练

```bash
# 使用默认配置训练
cd myGNN
python train.py
```

**默认配置:**
- 模型: GAT_SeparateEncoder (分离式编码器)
- 历史窗口: 14天
- 预测长度: 5天
- 损失函数: 加权趋势损失(WeightedTrend)

**训练输出:**
```
myGNN/checkpoints/GAT_SeparateEncoder_20260119_172246/
├── config.txt              # 训练配置
├── metrics.txt             # 评估指标
├── best_model.pth          # 最佳模型权重
├── train_losses.npy        # 训练损失历史
├── val_losses.npy          # 验证损失历史
├── loss_curves.png         # 损失曲线图
├── test_predict.npy        # 测试集预测
└── test_label.npy          # 测试集标签
```

---

## 📁 项目结构

```
gnn_predict/
├── 📊 data/                           # 数据目录
│   ├── result/                        # CSV原始数据(2010-2017年)
│   ├── real_weather_data_2010_2017.npy  # 转换后的NPY数据 [2922,28,28]
│   ├── station_info.npy               # 气象站信息 [28,4]
│   └── convert_real_data.py           # 数据转换脚本
│
├── 🧠 myGNN/                          # 核心框架 ⭐
│   ├── config.py                      # 配置管理模块(统一入口)
│   ├── train.py                       # 主训练脚本
│   ├── dataset.py                     # 数据加载(4维时间编码)
│   ├── network_GNN.py                 # 训练核心模块
│   ├── losses.py                      # 损失函数
│   ├── visualize_results.py           # 结果可视化
│   ├── explain_model.py               # 可解释性分析入口
│   │
│   ├── 🤖 models/                     # 模型子包
│   │   ├── GAT.py                     # GAT + LSTM
│   │   ├── GAT_SeparateEncoder.py     # GAT分离式编码器 ⭐
│   │   ├── GSAGE.py                   # GraphSAGE + LSTM
│   │   ├── GSAGE_SeparateEncoder.py   # GSAGE分离式编码器
│   │   ├── LSTM.py                    # LSTM基线模型
│   │   └── GAT_Pure.py                # 纯GAT模型
│   │
│   ├── 🗺️ graph/                      # 图结构子包
│   │   └── distance_graph.py          # 图构建模块
│   │
│   ├── 🔍 explainer/                  # 可解释性分析子包
│   │   ├── hybrid_explainer.py        # 混合解释器
│   │   ├── temporal_analyzer.py       # 时序分析
│   │   ├── spatial_explainer.py       # 空间分析
│   │   └── visualize_explainer.py     # 可视化(11种图表)
│   │
│   └── checkpoints/                   # 训练结果保存目录
│
├── 📈 figdraw/                        # 绘图脚本
│   ├── plot_lead_time_comparison.py
│   └── compare_models.py
│
├── 📄 CLAUDE.md                       # 项目架构详细说明 ⭐
├── 📄 DATA_FORMAT.md                  # 数据格式文档
└── 📄 README.md                       # 本文档
```

---

## 🎯 支持的模型

| 模型名称 | 说明 | 推荐场景 |
|---------|------|---------|
| **GAT_SeparateEncoder** ⭐ | GAT + 分离式编码器 | 默认推荐,性能最佳 |
| **GSAGE_SeparateEncoder** | GraphSAGE + 分离式编码器 | 大规模图结构 |
| **GAT_LSTM** | GAT + LSTM | 传统时空建模 |
| **GSAGE_LSTM** | GraphSAGE + LSTM | 可扩展性强 |
| **LSTM** | 纯LSTM | 基线对比 |
| **GAT_Pure** | 纯GAT | 无时序依赖 |

**分离式编码器(SeparateEncoder)核心创新:**
- 静态特征(地理位置、城市形态)只编码一次
- 动态特征(气象要素)保留时序处理
- 交叉注意力融合机制
- GAT残差连接增强

---

## ⚙️ 配置说明

### 修改配置

所有参数在 [myGNN/config.py](myGNN/config.py) 中集中管理:

```python
from myGNN.config import Config, ArchConfig, LossConfig

# 创建配置
config = Config()
arch_config = ArchConfig()
loss_config = LossConfig()

# 修改模型
config.exp_model = 'GAT_SeparateEncoder'

# 修改时间窗口
config.hist_len = 14        # 历史窗口14天
config.pred_len = 5         # 预测未来5天

# 修改图结构
config.graph_type = 'inv_dis'        # K近邻逆距离图
config.top_neighbors = 5             # 每个节点连接5个邻居

# 修改损失函数
loss_config.loss_type = 'WeightedTrend'  # 加权趋势损失
loss_config.alert_temp = 35.0            # 高温警戒阈值(°C)
loss_config.c_under = 4                  # 漏报权重(低估高温的惩罚)
loss_config.c_over = 2                   # 误报权重(高估的惩罚)
```

### 关键配置参数

**数据配置:**
```python
config.hist_len = 14                    # 历史窗口长度(天)
config.pred_len = 5                     # 预测长度(天)
config.target_feature_idx = 4           # 预测目标(4=tmax最高气温)
config.use_feature_separation = True    # 启用特征分离
```

**模型架构:**
```python
arch_config.hid_dim = 16                # 隐藏层维度
arch_config.GAT_layer = 1               # GAT层数
arch_config.heads = 1                   # 注意力头数
arch_config.dropout = True              # 启用Dropout
```

**训练参数:**
```python
config.batch_size = 32                  # 批次大小
config.epochs = 500                     # 最大训练轮数
config.lr = 0.001                       # 学习率
config.weight_decay = 1e-3              # 权重衰减
config.early_stop = 50                  # 早停耐心值
```

**详细配置说明:** 参见 [myGNN/README.md](myGNN/README.md)

---

## 🔍 可解释性分析

完整的模型可解释性分析框架,生成11种专业可视化图表。

### 快速使用

```bash
# Windows环境
"D:\anaconda\python.exe" "c:\Users\wxb55\Desktop\gnn_predict\myGNN\explain_model.py" --model_path checkpoints/GAT_SeparateEncoder_xxx/best_model.pth --num_samples 100 --visualize

# 夏季高温分析
"D:\anaconda\python.exe" "c:\Users\wxb55\Desktop\gnn_predict\myGNN\explain_model.py" --model_path checkpoints/GAT_SeparateEncoder_xxx/best_model.pth --season summer --visualize
```

### 生成的可视化图表

1. **时序特征热图** - 各时间步特征重要性
2. **空间边地理图(Top-K)** - 最重要的K条边在地图上可视化
3. **全边叠加图** - 所有边的重要性叠加
4. **全边分离图** - 每条边单独展示
5. **GNNExplainer vs GAT注意力对比** - 两种解释方法的对比
6. **边重要性分布** - 重要性直方图
7. **时间步重要性** - 各时间步的贡献度
8. **特征重要性排名** - Top特征柱状图
9. **全局注意力矩阵热力图** - 站点间注意力模式
10. **距离-注意力散点图** - 距离与注意力的关系
11. **温度相关性-注意力散点图** - 温度相关性与注意力的关系

**详细文档:** [myGNN/explainer/README.md](myGNN/explainer/README.md)

---

## 📊 结果可视化

训练完成后自动生成可视化图表:

```bash
# Windows环境
"D:\anaconda\python.exe" "c:\Users\wxb55\Desktop\gnn_predict\myGNN\visualize_results.py" --checkpoint_dir checkpoints/GAT_SeparateEncoder_20260119_172246
```

**生成的图表:**
- 损失曲线图 (training/validation loss)
- 预测对比散点图 (按预测步长)
- 时间序列对比图 (所有28个站点)
- 残差分析图
- 误差分布直方图

---

## 🎨 示例用法

### 示例1: 训练默认模型

```bash
cd myGNN
"D:\anaconda\python.exe" train.py
```

### 示例2: 修改时间窗口

编辑 [myGNN/config.py](myGNN/config.py):
```python
config.hist_len = 7         # 使用过去7天
config.pred_len = 3         # 预测未来3天
```

运行训练:
```bash
"D:\anaconda\python.exe" train.py
```

### 示例3: 切换到LSTM基线模型

编辑 [myGNN/config.py](myGNN/config.py):
```python
config.exp_model = 'LSTM'
```

运行训练:
```bash
"D:\anaconda\python.exe" train.py
```

### 示例4: 使用标准MSE损失

编辑 [myGNN/config.py](myGNN/config.py):
```python
loss_config.loss_type = 'MSE'
```

运行训练:
```bash
"D:\anaconda\python.exe" train.py
```

### 示例5: 特征选择实验

编辑 [myGNN/config.py](myGNN/config.py):
```python
# 只使用核心气象特征
config.feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 0-2: x, y, height (空间特征)
# 3-5: tmin, tmax, tave (温度)
# 6-9: pre, prs, rh, win (气象要素)
# 最终输入: 10 (基础) + 4 (时间编码) = 14维
```

---

## 📈 性能指标

典型性能指标(GAT_SeparateEncoder模型,测试集):

| 预测步长 | RMSE (°C) | MAE (°C) | R² |
|---------|-----------|----------|-----|
| Day 1 | 1.2-1.5 | 0.9-1.2 | 0.95+ |
| Day 3 | 1.8-2.2 | 1.4-1.8 | 0.90+ |
| Day 5 | 2.3-2.8 | 1.8-2.3 | 0.85+ |

*注: 实际性能取决于超参数配置和数据特征*

---

## 🔧 常见问题

### Q1: 如何处理显存不足?

减小以下参数:
```python
config.batch_size = 16          # 从32减小到16
arch_config.hid_dim = 8         # 从16减小到8
config.hist_len = 7             # 从14减小到7
```

### Q2: 如何加快训练速度?

- 减小 `config.epochs` (如从500减到200)
- 增大 `config.batch_size` (如从32增到64)
- 减小 `config.hist_len` 和 `config.pred_len`
- 使用更简单的模型 (如 `LSTM` 代替 `GAT_SeparateEncoder`)

### Q3: 如何使用自己的数据?

1. 准备NPY格式数据: `[time_steps, num_stations, features]`
2. 准备气象站信息: `[num_stations, 4]` (ID, 经度, 纬度, 高度)
3. 修改 [myGNN/config.py](myGNN/config.py):
   ```python
   config.MetData_fp = 'data/my_weather_data.npy'
   config.station_info_fp = 'data/my_station_info.npy'
   config.node_num = 你的站点数量
   config.base_feature_dim = 你的特征数量
   ```

### Q4: 如何对比不同模型?

```bash
# 训练多个模型
for model in GAT_LSTM GSAGE_LSTM GAT_SeparateEncoder; do
    # 编辑config.py修改exp_model
    "D:\anaconda\python.exe" train.py
done

# 对比结果
"D:\anaconda\python.exe" figdraw/compare_models.py
```

### Q5: 训练过程中出现NaN怎么办?

- 减小学习率: `config.lr = 0.0001`
- 增大权重衰减: `config.weight_decay = 1e-2`
- 检查数据是否包含NaN值
- 尝试使用 `loss_config.loss_type = 'MSE'`

---

## 📚 相关文档

### 核心文档
- [CLAUDE.md](CLAUDE.md) - 项目架构详细说明 ⭐⭐⭐ (最详细)
- [DATA_FORMAT.md](DATA_FORMAT.md) - 数据格式文档
- [myGNN/README.md](myGNN/README.md) - myGNN框架文档

### 模块文档
- [myGNN/explainer/README.md](myGNN/explainer/README.md) - 可解释性分析模块



## 🌟 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络库
- [Captum](https://captum.ai/) - 可解释性分析工具
- [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) - 气象再分析数据




---

<p align="center">
  <b>⭐ 如果觉得项目有用,请给个Star! ⭐</b>
</p>


