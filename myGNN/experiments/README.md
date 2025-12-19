# 模型对比实验模块

## 📋 功能概述

本模块提供多模型性能对比分析功能，用于评估不同模型在城市气温预测任务中的表现。

## 📂 文件说明

- `compare_models.py` - 多模型对比脚本
- `model_comparison.csv` - 对比结果表格（运行后生成）
- `model_comparison.png` - 对比可视化图表（运行后生成）

## 🚀 使用方法

### 1. 训练模型

首先需要训练要对比的模型：

```bash
# 训练GNN模型（修改config.py中的exp_model）
python myGNN/train.py

# 训练XGBoost基线
python myGNN/baselines/train_xgboost.py
```

### 2. 运行对比分析

```bash
python myGNN/experiments/compare_models.py
```

### 3. 查看结果

对比脚本会自动：
1. 扫描 `myGNN/checkpoints/` 目录下的所有训练结果
2. 解析每个模型的 `metrics.txt` 文件
3. 生成对比CSV文件
4. 打印对比摘要
5. 生成可视化图表

## 📊 输出格式

### model_comparison.csv

包含以下列：
- `Model` - 模型名称
- `Checkpoint` - checkpoint目录名
- `Test_RMSE` - 测试集RMSE (°C)
- `Test_MAE` - 测试集MAE (°C)
- `Test_R2` - 测试集R²
- `Test_Bias` - 测试集Bias (°C)
- `Val_RMSE` - 验证集RMSE (°C)
- `Val_MAE` - 验证集MAE (°C)
- `Val_R2` - 验证集R²
- `Val_Bias` - 验证集Bias (°C)

### 控制台输出示例

```
================================================================================
模型性能对比 (按测试集RMSE排序)
================================================================================
      Model  Test_RMSE  Test_MAE  Test_R2  Test_Bias
  GAT_LSTM     1.2345    0.9876   0.8765    +0.0123
GSAGE_LSTM     1.3456    1.0234   0.8543    -0.0234
  GAT_Pure     1.4567    1.1234   0.8321    +0.0345
   XGBoost     1.6789    1.2345   0.7654    -0.0456
      LSTM     1.7890    1.3456   0.7432    +0.0567
================================================================================

🏆 最佳模型 (RMSE最低): GAT_LSTM
   测试集RMSE: 1.2345 °C
   测试集MAE:  0.9876 °C
   测试集R²:   0.8765
   测试集Bias: +0.0123 °C
```

### model_comparison.png

4个子图对比：
1. **测试集RMSE对比** - 柱状图
2. **测试集MAE对比** - 柱状图
3. **测试集R²对比** - 柱状图
4. **测试集Bias对比** - 柱状图（包含零线）

## 🔧 自定义

### 修改对比指标

编辑 `compare_models.py` 中的 `print_comparison_summary()` 函数：

```python
display_cols = ['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_Bias']
```

### 修改排序方式

编辑 `generate_comparison_table()` 函数：

```python
# 按测试集RMSE排序
df = df.sort_values('Test_RMSE')

# 或按MAE排序
df = df.sort_values('Test_MAE')
```

### 添加更多可视化

在 `plot_comparison()` 函数中添加新的子图。

## 📝 注意事项

1. **checkpoints目录结构**：脚本假设checkpoint目录格式为 `{模型名}_{时间戳}/`
2. **metrics.txt格式**：必须包含标准的评估指标格式（由train.py自动生成）
3. **中文显示**：需要系统安装SimHei字体，否则可视化图表中文显示异常

## 🎯 支持的模型

当前框架支持对比以下模型：
- **LSTM** - 纯LSTM基线（无图结构）
- **GAT_LSTM** - 图注意力网络 + LSTM
- **GSAGE_LSTM** - GraphSAGE + LSTM
- **GAT_Pure** - 纯GAT（无LSTM） ⭐新增
- **GAT_SeparateEncoder** - GAT + 分离式编码器
- **GSAGE_SeparateEncoder** - GSAGE + 分离式编码器
- **XGBoost** - 传统梯度提升树基线 ⭐新增

## 🔍 故障排除

### 问题1: 未找到任何训练结果

**原因**：checkpoints目录为空或metrics.txt不存在

**解决**：先运行训练脚本生成结果

### 问题2: 解析metrics.txt失败

**原因**：metrics.txt格式不符合预期

**解决**：确保使用train.py或train_xgboost.py生成的标准格式

### 问题3: 可视化图表中文乱码

**原因**：系统未安装SimHei字体

**解决**：
```python
# 修改compare_models.py中的字体设置
matplotlib.rcParams['font.sans-serif'] = ['Arial']  # 使用英文字体
```

## 📚 参考

- 主训练脚本: [myGNN/train.py](../train.py)
- XGBoost训练: [myGNN/baselines/train_xgboost.py](../baselines/train_xgboost.py)
- 项目文档: [CLAUDE.md](../../CLAUDE.md)
