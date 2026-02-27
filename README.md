# GNN气温预测系统

基于图神经网络（GNN）的华南地区日最高气温预测系统，覆盖28个气象站，数据跨度2010–2017年。

---

## 核心特性

- **4维时间周期编码**：将 `doy`/`month` 转为 `sin/cos` 表示，解决年末跳变问题
- **静态/动态特征分离编码**：城市环境等静态特征单次编码，气象动态特征保留时序处理
- **特征级交叉注意力融合**：动态特征自适应查询静态地理信息（GAT_SeparateEncoder v3.0）
- **加权趋势损失函数**：针对高温漏报的不对称惩罚 + 一阶差分趋势约束
- **多模型架构支持**：GAT、GraphSAGE、LSTM 及其组合变体

---

## 环境配置

**Conda 环境：** `gnn_predict`

```bash
# 激活环境
D:\anaconda\Scripts\activate.bat && conda activate gnn_predict

# 安装依赖
cd myGNN
pip install -r requirements.txt
```

**主要依赖：**
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- Captum（可解释性分析）
- Cartopy, Contextily（地图底图，可选）

---

## 快速开始

```bash
# 1. 数据转换（仅首次运行）
D:\anaconda\Scripts\activate.bat && conda activate gnn_predict && C:\Users\wxb55\.conda\envs\urban_climate\gnn_predict\python.exe c:/Users/wxb55/Desktop/gnn_predict/data/convert_real_data.py

# 2. 训练模型
D:\anaconda\Scripts\activate.bat && conda activate gnn_predict && C:\Users\wxb55\.conda\envs\urban_climate\gnn_predict\python.exe c:/Users/wxb55/Desktop/gnn_predict/myGNN/train.py

# 3. 查看结果：myGNN/checkpoints/GAT_SeparateEncoder_时间戳/
```

---

## 项目结构

```
gnn_predict/
├── data/
│   ├── result/                          # CSV原始数据（2010-2017年，8个文件）
│   ├── real_weather_data_2010_2017.npy  # 主数据 [2922, 28, 28]
│   ├── station_info.npy                 # 气象站信息 [28, 4]
│   ├── convert_real_data.py             # CSV → NPY 转换脚本
│   └── diagnose_nan.py                  # 数据质量诊断
│
├── myGNN/
│   ├── config.py                        # 统一配置管理（修改参数的主要入口）
│   ├── dataset.py                       # 数据加载 + 4维时间编码 + 滑动窗口
│   ├── network_GNN.py                   # 训练/验证/测试核心逻辑
│   ├── losses.py                        # WeightedTrendMSELoss 损失函数
│   ├── feature_encoder.py               # 静态特征编码器
│   ├── train.py                         # 主训练脚本
│   ├── train_enhanced.py                # 增强训练脚本（含更多分析）
│   ├── visualize_results.py             # 训练结果可视化
│   ├── explain_model.py                 # 可解释性分析入口
│   │
│   ├── models/
│   │   ├── GAT_SeparateEncoder.py       # ★ 推荐：GAT + 特征分离 + 交叉注意力 v3.0
│   │   ├── GSAGE_SeparateEncoder.py     # GraphSAGE + 特征分离
│   │   ├── GAT.py                       # GAT + LSTM
│   │   ├── GSAGE.py                     # GraphSAGE + LSTM
│   │   ├── GAT_Pure.py                  # 纯GAT（无LSTM）
│   │   └── LSTM.py                      # 基线模型（无图结构）
│   │
│   ├── graph/
│   │   └── distance_graph.py            # K近邻图构建（逆距离权重/特征相似性/KNN）
│   │
│   ├── explainer/
│   │   ├── hybrid_explainer.py          # 混合解释器（时空联合分析）
│   │   ├── spatial_explainer.py         # 空间关系分析
│   │   ├── temporal_analyzer.py         # 时序特征分析
│   │   ├── visualize_explainer.py       # 可解释性可视化
│   │   └── utils.py                     # 工具函数
│   │
│   ├── experiments/
│   │   └── compare_models.py            # 多模型架构对比实验
│   │
│   ├── utils/
│   │   └── cartopy_helpers.py           # 地图绘制辅助函数
│   │
│   ├── checkpoints/                     # 训练检查点（自动生成）
│   └── analysis_results/               # 可解释性分析结果
│
├── figdraw/
│   ├── spatialtemperal.py               # 时空分析图
│   ├── plot_graph_structure.py          # 图结构可视化
│   ├── plot_gat_attention.py            # GAT注意力热力图
│   ├── cross-attn_weight.py             # 交叉注意力权重分析
│   ├── compare_rmse_spatial.py          # 两模型RMSE空间差值对比
│   ├── compare_LossFunction.py          # 损失函数对比
│   ├── compare_model_seasonal_errors.py # 季节性误差对比
│   ├── compare_validation_scatter.py    # 验证集散点图对比
│   ├── plot_lead_time_comparison.py     # 不同预测步长对比
│   ├── plot_temperature_spatial_mean.py # 气温空间均值图
│   ├── analyze_extreme_errors.py        # 极端误差分析
│   ├── analyze_station_errors.py        # 站点误差分析
│   └── result/                          # 生成的图表输出目录
│
├── DATA_FORMAT.md                       # 详细数据格式说明
├── CLAUDE.md                            # Claude Code 项目指南
└── README.md
```

---

## 数据说明

### 主数据 `real_weather_data_2010_2017.npy`

形状：`[2922, 28, 28]`（天数 × 站点数 × 特征数）

| 索引 | 特征 | 类型 |
|------|------|------|
| 0–2 | x, y, height | 空间位置 |
| 3–5 | tmin, **tmax**, tave | 温度（tmax为预测目标） |
| 6–9 | pre, prs, rh, win | 气象观测 |
| 10–18 | BH, BHstd, SCD, PLA, λp, λb, POI, POW, POV | 城市环境 |
| 19 | NDVI | 植被指数 |
| 20–23 | surface_pressure, solar_radiation, u_wind, v_wind | ERA5再分析 |
| 24–25 | VegHeight_mean, VegHeight_std | 植被高度 |
| 26–27 | doy, month | 时间（自动转为4维sin/cos编码） |

### 数据集划分

| 集合 | 年份 | 索引范围 | 天数 |
|------|------|----------|------|
| 训练集 | 2010–2015 | 0–2190 | 2191 |
| 验证集 | 2016 | 2191–2556 | 366 |
| 测试集 | 2017 | 2557–2921 | 365 |

---

## 模型架构

### 推荐模型：GAT_SeparateEncoder（v3.0）

```
输入特征
  ├── 静态特征（位置、城市环境、植被）→ LightweightStaticEncoder → [N, 12, dim]
  └── 动态特征（气象、ERA5、温度）+ 时间编码 → LSTM → [N, dim]
          ↓
    特征级交叉注意力（Cross-Attention）
          ↓
    GATv2Conv（多头注意力 + Skip Connection）
          ↓
    MLP解码器 → 预测未来5天tmax
```

**关键设计：**
- `LightweightStaticEncoder`：特征值 × 可学习基向量，参数量极低
- 可学习节点嵌入（Node Embedding）：捕获隐式微气候效应
- GATv2 残差连接：防止图卷积过度平滑

### 其他模型

| 模型 | 特点 |
|------|------|
| `GAT_LSTM` | GAT + LSTM，无特征分离 |
| `GSAGE_SeparateEncoder` | GraphSAGE + 特征分离 |
| `GSAGE_LSTM` | GraphSAGE + LSTM |
| `GAT_Pure` | 纯GAT，无时序建模 |
| `LSTM` | 基线模型，无图结构 |

---

## 配置说明

所有参数集中在 `myGNN/config.py`，直接编辑即可，无需命令行参数。

### 关键配置项

```python
# 模型选择
config.exp_model = 'GAT_SeparateEncoder'  # 推荐

# 时间窗口
config.hist_len = 14   # 历史窗口（天）
config.pred_len = 5    # 预测步长（天）

# 图结构
config.graph_type = 'inv_dis'   # 逆距离加权K近邻（推荐）
config.top_neighbors = 5        # K近邻数量
config.use_edge_attr = False    # 是否使用边权重

# 训练参数
config.batch_size = 32
config.epochs = 500
config.lr = 0.001
config.optimizer = 'Adam'       # Adam / AdamW / SGD / RMSprop

# 损失函数（LossConfig）
loss_config.loss_type = 'WeightedTrend'   # 或 'MSE'
loss_config.use_dynamic_threshold = True  # True=90分位数阈值，False=固定阈值
loss_config.alert_temp = 35.0             # 固定高温阈值（°C）
loss_config.c_under = 4                   # 漏报惩罚系数
loss_config.c_over = 2                    # 误报惩罚系数
loss_config.trend_weight = 0              # 趋势约束权重
```

---

## 损失函数

### WeightedTrendMSELoss

针对高温预测的自适应加权损失：

```
L_total = L_weighted_mse + α × L_trend

L_weighted_mse = mean(w_i × (pred_i - label_i)²)

权重规则：
  漏报高温（实际≥阈值，预测<实际）：w += c_under × (T_actual - T_alert + δ)
  误报高温（实际<阈值，预测≥阈值）：w += c_over  × (T_pred  - T_alert + δ)
  正确命中高温：                      w += 1 × (T_actual - T_alert + δ)

L_trend = MSE(Δpred, Δlabel)  # 一阶差分趋势约束
```

权重基于反标准化后的物理温度计算，梯度在标准化空间中传播，兼顾物理意义与数值稳定性。

---

## 常用命令

```bash
PYTHON=C:\Users\wxb55\.conda\envs\urban_climate\gnn_predict\python.exe
ACT=D:\anaconda\Scripts\activate.bat && conda activate gnn_predict

# 训练
$ACT && $PYTHON c:/Users/wxb55/Desktop/gnn_predict/myGNN/train.py

# 增强训练
$ACT && $PYTHON c:/Users/wxb55/Desktop/gnn_predict/myGNN/train_enhanced.py

# 模型对比实验
$ACT && $PYTHON c:/Users/wxb55/Desktop/gnn_predict/myGNN/experiments/compare_models.py

# 可解释性分析
$ACT && $PYTHON c:/Users/wxb55/Desktop/gnn_predict/myGNN/explain_model.py

# 可视化（figdraw/下各脚本）
$ACT && $PYTHON c:/Users/wxb55/Desktop/gnn_predict/figdraw/spatialtemperal.py
$ACT && $PYTHON c:/Users/wxb55/Desktop/gnn_predict/figdraw/compare_rmse_spatial.py
$ACT && $PYTHON c:/Users/wxb55/Desktop/gnn_predict/figdraw/plot_gat_attention.py
```

---

## 输出文件

| 路径 | 内容 |
|------|------|
| `myGNN/checkpoints/<模型名>_<时间戳>/best_model.pth` | 最佳模型权重 |
| `myGNN/checkpoints/<模型名>_<时间戳>/config.json` | 训练配置快照 |
| `myGNN/checkpoints/<模型名>_<时间戳>/metrics.json` | RMSE/MAE评估指标 |
| `myGNN/checkpoints/<模型名>_<时间戳>/loss_curve.png` | 训练/验证损失曲线 |
| `figdraw/result/` | 所有可视化图表 |
| `myGNN/analysis_results/` | 可解释性分析结果 |

---

## 参考文献

刘旭, 杨昊, 梁潇云, 等. 基于注意力机制与加权趋势损失的风速订正方法. 应用气象学报, 2025, 36(3): 316–327.
