# GNN 气温预测系统

基于图神经网络（GNN）的华南地区日气温预测系统，覆盖 28 个气象站，数据跨度 2010–2019 年。

## 核心特性

- **4 维时间周期编码**：sin/cos 变换解决年末跳变问题
- **静态/动态特征分离**：静态特征只编码一次，动态特征保留时序处理
- **特征级交叉注意力**：动态特征查询静态特征，支持可解释性分析
- **加权趋势损失函数**（WeightedTrendMSELoss）：针对高温漏报优化
- **多种 GNN 架构支持**：GAT、GraphSAGE、LSTM 及其组合变体
- **K 近邻图结构**：逆距离权重 / 空间相似性 / 简单 KNN

## 快速开始

### 1. 环境准备

```bash
conda create -n gnn_predict python=3.10
conda activate gnn_predict
pip install torch>=2.0.0 torch-geometric>=2.3.0
pip install numpy pandas scipy matplotlib seaborn captum tqdm
```

完整依赖见 [myGNN/requirements.txt](myGNN/requirements.txt)。

### 2. 数据转换（仅首次）

将 CSV 原始数据转换为 NPY 格式：

```bash
python data/convert_real_data.py
```

### 3. 开始训练

修改 `myGNN/config.py` 中的参数后运行：

```bash
python myGNN/train.py
```

训练完成后自动输出：
- 最佳模型权重 `myGNN/checkpoints/<模型>_<时间戳>/best_model.pth`
- 训练配置快照 `config.json`
- 评估指标 `metrics.json`
- 损失曲线 `loss_curve.png`

## 常用命令

```bash
# 主训练
python myGNN/train.py

# 增强训练
python myGNN/train_enhanced.py

# 模型对比实验
python myGNN/experiments/compare_models.py

# 可解释性分析
python myGNN/explain_model.py

# 可视化（输出到 figdraw/result/）
python figdraw/spatialtemperal.py
python figdraw/plot_gat_attention.py
python figdraw/compare_rmse_spatial.py
```

## 项目结构

```
gnn_predict/
├── data/
│   ├── result/                          # CSV 原始数据（2010-2019 年）
│   ├── real_weather_data_2010_2019.npy  # 主数据 [3652, 28, 29]
│   ├── station_info.npy                 # 气象站信息 [28, 4]
│   └── convert_real_data.py             # CSV → NPY 转换脚本
│
├── myGNN/
│   ├── config.py                        # ★ 统一配置管理
│   ├── dataset.py                       # 数据加载 + 4维时间编码 + 滑动窗口
│   ├── network_GNN.py                   # 训练/验证/测试核心逻辑
│   ├── losses.py                        # WeightedTrendMSELoss
│   ├── feature_encoder.py               # 静态特征编码器
│   ├── train.py                         # 主训练脚本
│   ├── train_enhanced.py                # 增强训练脚本
│   ├── explain_model.py                 # 可解释性分析
│   ├── models/
│   │   ├── GAT_SeparateEncoder.py       # ★ 推荐模型
│   │   ├── GSAGE_SeparateEncoder.py
│   │   ├── GAT.py
│   │   ├── GSAGE.py
│   │   ├── GAT_Pure.py
│   │   └── LSTM.py
│   ├── graph/
│   │   └── distance_graph.py            # K近邻图构建
│   ├── explainer/                       # 可解释性分析模块
│   └── experiments/
│       └── compare_models.py            # 模型对比实验
│
└── figdraw/
    ├── spatialtemperal.py               # 时空分析图
    ├── plot_graph_structure.py          # 图结构可视化
    ├── plot_gat_attention.py            # GAT 注意力热力图
    ├── compare_rmse_spatial.py          # RMSE 空间差值对比
    └── result/                          # 图表输出目录
```

## 数据说明

### 主数据格式

| 文件 | 形状 | 说明 |
|------|------|------|
| `real_weather_data_2010_2019.npy` | `[3652, 28, 29]` | 3652 天 × 28 站 × 29 特征 |
| `station_info.npy` | `[28, 4]` | 站点 ID、经度、纬度、高度 |

### 数据集划分

| 集合 | 年份 | 索引 | 天数 |
|------|------|------|------|
| 训练 | 2010–2017 | 0–2921 | 2922 |
| 验证 | 2018 | 2922–3286 | 365 |
| 测试 | 2019 | 3287–3651 | 365 |

### 29 个特征

| 索引 | 特征 | 类型 | 索引 | 特征 | 类型 |
|------|------|------|------|------|------|
| 0–2 | x, y, height | 空间 | 10–18 | BH, BHstd, SCD, PLA, λp, λb, POI, POW, POV | 城市环境 |
| 3–5 | tmin, **tmax**, tave | 温度 | 19 | NDVI | 植被 |
| 6–9 | pre, prs, rh, win | 气象 | 20–24 | ERA5 再分析数据 | 气象 |
| | | | 25–26 | VegHeight_mean, VegHeight_std | 植被 |
| | | | 27–28 | doy, month | 时间（自动转 sin/cos） |

> 详细数据格式见 [DATA_FORMAT.md](DATA_FORMAT.md)

## 模型说明

| 模型 | 架构 | 特点 |
|------|------|------|
| **GAT_SeparateEncoder** | GAT + 特征分离 + 交叉注意力 | ★ 推荐，v3.0 |
| GSAGE_SeparateEncoder | GraphSAGE + 特征分离 | 轻量替代 |
| GAT | GAT + LSTM | 经典组合 |
| GSAGE | GraphSAGE + LSTM | 经典组合 |
| GAT_Pure | 纯 GAT | 无时序建模 |
| LSTM | 纯 LSTM | 基线模型 |

## 配置说明

**所有参数在 `myGNN/config.py` 中修改，无需命令行参数。**

三个配置类：
- `Config` — 数据路径、模型选择、训练参数、图结构
- `ArchConfig` — GAT/LSTM 层数、注意力头数、隐藏维度等
- `LossConfig` — 损失函数类型和高温阈值参数

### 关键配置项

```python
# 模型选择
config.exp_model = "GAT_SeparateEncoder"

# 时间窗口
config.hist_len = 14      # 历史窗口（天）
config.pred_len = 5       # 预测步长（天）

# 图结构
config.graph_type = "inv_dis"
config.top_neighbors = 5

# 训练参数
config.batch_size = 32
config.epochs = 500
config.lr = 0.001
config.optimizer = 'AdamW'

# 损失函数
loss_config.loss_type = 'WeightedTrend'
loss_config.use_station_day_threshold = True  # 站点-日内动态阈值
```

## 输出文件

| 路径 | 内容 |
|------|------|
| `myGNN/checkpoints/<模型>_<时间戳>/best_model.pth` | 最佳模型权重 |
| `myGNN/checkpoints/<模型>_<时间戳>/config.json` | 训练配置快照 |
| `myGNN/checkpoints/<模型>_<时间戳>/metrics.json` | RMSE/MAE 评估指标 |
| `myGNN/checkpoints/<模型>_<时间戳>/loss_curve.png` | 训练/验证损失曲线 |
| `figdraw/result/` | 所有可视化图表 |
| `myGNN/analysis_results/` | 可解释性分析结果 |

## 参考文档

- [CLAUDE.md](CLAUDE.md) — 详细架构说明、可视化脚本说明
- [DATA_FORMAT.md](DATA_FORMAT.md) — 数据格式详解、4维时间编码原理、使用示例

## 许可证

本项目仅供学术研究使用。
