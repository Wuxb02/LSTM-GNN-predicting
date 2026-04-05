# AGENTS.md

GNN气温预测系统 — 华南28站日最高气温预测（2010–2019），PyTorch + PyG。

## 执行命令

**Conda 环境**: `gnn_predict`

```bash
D:\anaconda\Scripts\activate.bat && conda activate gnn_predict && C:\Users\wxb55\.conda\envs\urban_climate\gnn_predict\python.exe <脚本路径>
```

所有脚本路径使用正斜杠 `/`。

## 关键入口

| 脚本 | 用途 |
|------|------|
| `myGNN/train.py` | 主训练（唯一入口，无需命令行参数） |
| `myGNN/train_enhanced.py` | 增强训练 |
| `myGNN/experiments/compare_models.py` | 模型对比实验 |
| `myGNN/explain_model.py` | 可解释性分析 |
| `data/convert_real_data.py` | CSV→NPY转换（仅首次） |
| `figdraw/*.py` | 各可视化脚本，输出到 `figdraw/result/` |

## 配置 — 唯一参数入口

**所有参数在 `myGNN/config.py` 中修改，无需命令行参数。**

三个配置类：
- `Config` — 数据路径、模型选择、训练参数、图结构
- `ArchConfig` — GAT/LSTM层数、注意力头数、隐藏维度等
- `LossConfig` — 损失函数类型和高温阈值参数

### 当前关键默认值（agent 容易猜错的）

- `config.exp_model = "GAT_SeparateEncoder"` — 推荐模型
- `config.use_feature_separation = True` — **默认启用**特征分离模式
- `config.target_feature_idx = 5` — 预测目标是 **tave**（日均温），不是 tmax
- `config.hist_len = 14`, `config.pred_len = 5`
- `config.graph_type = "inv_dis"`, `config.top_neighbors = 5`
- `config.static_feature_indices = [0,1,2,10,11,12,16,17,18,25]` — 10个静态特征
- `config.dynamic_feature_indices = [5,8,21,22,23,24]` — 6个动态特征
- `config.loss_config.loss_type = "WeightedTrend"`
- `config.loss_config.use_station_day_threshold = True` — 站点-日内动态阈值（365×28表）
- `config.epochs = 5` — 当前设得很低，可能是调试值
- `config.optimizer = 'AdamW'`, `config.scheduler = "ReduceLROnPlateau"`

## 数据

- **主数据**: `data/real_weather_data_2010_2019.npy` — shape `[3652, 28, 29]`（config 指向此文件）
- **旧数据**: `data/real_weather_data_2010_2017.npy` — shape `[2922, 28, 29]`（仍存在但 config 不再使用）
- **站点信息**: `data/station_info.npy` — shape `[28, 4]`

### 数据集划分（当前 config）

| 集合 | 年份 | 索引 | 天数 |
|------|------|------|------|
| 训练 | 2010–2017 | 0–2921 | 2922 |
| 验证 | 2018 | 2922–3286 | 365 |
| 测试 | 2019 | 3287–3651 | 365 |

### 特征维度变化

```
原始 NPY: [time, stations, 29]
  → 移除 doy(27), month(28)
  → 特征分离模式: 静态10 + 动态6 + 时间编码4 = 20 维输入
  → 模型输入: [nodes, hist_len, 20]
```

## 模型列表

| 模型文件 | 说明 |
|----------|------|
| `GAT_SeparateEncoder.py` | ★ 推荐：GAT + 特征分离 + 交叉注意力 |
| `GSAGE_SeparateEncoder.py` | GraphSAGE + 特征分离 |
| `GAT.py` | GAT + LSTM |
| `GSAGE.py` | GraphSAGE + LSTM |
| `GAT_Pure.py` | 纯 GAT |
| `LSTM.py` | 基线 |

## 注意事项

- `.gitignore` 忽略 `*.npy`，数据文件不在 git 中但本地存在
- `dataset.py` 自动处理闰年（2月29日样本被过滤）
- 标准化在 `create_dataloaders()` 中自动完成（基于训练集统计量）
- 损失函数的权重基于**反标准化后的物理温度**计算，梯度在标准化空间传播
- 无 lint/test/CI 配置，代码遵循 PEP8
- 修改 `config.py` 后直接运行 `train.py`，无需重新构建

## 参考文档

- `CLAUDE.md` — 详细架构说明、29特征表、可视化脚本说明
- `DATA_FORMAT.md` — 数据格式详解、4维时间编码原理、使用示例
