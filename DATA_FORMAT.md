# 数据格式说明文档

本文档详细说明了GNN气温预测项目中使用的数据格式,包括原始CSV数据和转换后的NPY数组格式。

---

## 目录

- [1. 数据文件概览](#1-数据文件概览)
- [2. NPY数组格式](#2-npy数组格式)
- [3. CSV原始数据格式](#3-csv原始数据格式)
- [4. 4维时间周期编码](#4-4维时间周期编码)
- [5. 使用示例](#5-使用示例)
- [6. 数据处理流程](#6-数据处理流程)
- [7. 常见问题](#7-常见问题)

---

## 1. 数据文件概览

### 1.1 目录结构

```
gnn_predict/
├── data/
│   ├── result/                                    # 原始CSV数据
│   │   ├── merged_data_2010_2000m.csv            # 2010年数据 (365天 × 28站 = 10,220行)
│   │   ├── merged_data_2011_2000m.csv            # 2011年数据 (365天 × 28站 = 10,220行)
│   │   ├── merged_data_2012_2000m.csv            # 2012年数据 (366天 × 28站 = 10,248行, 闰年)
│   │   ├── merged_data_2013_2000m.csv            # 2013年数据 (365天 × 28站 = 10,220行)
│   │   ├── merged_data_2014_2000m.csv            # 2014年数据 (365天 × 28站 = 10,220行)
│   │   ├── merged_data_2015_2000m.csv            # 2015年数据 (365天 × 28站 = 10,220行)
│   │   ├── merged_data_2016_2000m.csv            # 2016年数据 (366天 × 28站 = 10,248行, 闰年)
│   │   └── merged_data_2017_2000m.csv            # 2017年数据 (365天 × 28站 = 10,220行)
│   │
│   ├── real_weather_data_2010_2017.npy           # 转换后的真实数据 [2922, 28, 29] ⭐
│   ├── station_info.npy                          # 气象站信息 [28, 4] ⭐
│   ├── convert_real_data.py                      # 数据转换脚本
│   └── README.md                                 # 数据目录说明
│
├── myGNN/                                        # 核心框架
│   ├── dataset.py                                # 数据加载(4维时间编码)
│   └── ...
│
└── DATA_FORMAT.md                                # 本文档
```

### 1.2 数据覆盖范围

| 项目 | 说明 |
|------|------|
| **时间范围** | 2010年1月1日 - 2017年12月31日（8年完整数据） |
| **气象站数量** | 28个 |
| **地理位置** | 中国华南地区（经度: 111.0°-114.0°E, 纬度: 21.4°-23.4°N） |
| **时间分辨率** | 日数据（每天一个观测值） |
| **总数据量** | 81,816条记录（约2922天 × 28站，考虑闰年） |
| **数据来源** | 真实气象观测 + ERA5再分析数据 + 植被数据 |

### 1.3 数据集划分

| 数据集 | 时间范围 | 索引范围 | 天数 | 说明 |
|-------|---------|---------|------|------|
| **训练集** | 2010-2015年 | 0-2190 | 2191天 | 6年完整数据 |
| **验证集** | 2016年 | 2191-2556 | 366天 | 闰年,用于超参数调优 |
| **测试集** | 2017年 | 2557-2921 | 365天 | 独立测试,评估泛化性能 |

**划分原则:**
- 按年份划分,避免时间泄露
- 训练集占比约75%(6年/8年)
- 验证集和测试集各占约12.5%
- 保持时间连续性,符合实际预测场景

---

## 2. NPY数组格式

### 2.1 主数据数组 (`real_weather_data_2010_2017.npy`) ⭐⭐⭐

#### 2.1.1 数组维度

```python
shape: [时间步, 气象站数, 特征数] = [2922, 28, 29]
```

- **维度0 (时间步)**: 2922天 (2010-01-01 至 2017-12-31, 包含2个闰年)
- **维度1 (气象站)**: 28个气象站，按站点ID升序排序
- **���度2 (特征)**: 29个特征（完整特征集）

#### 2.1.2 特征顺序与索引（29个特征）

| 索引 | 特征名 | 类型 | 单位 | 典型范围 | 说明 |
|:----:|--------|------|:----:|---------|------|
| **空间特征（静态）** |
| 0 | `x` | 空间 | ° | 111.0-114.0 | 经度（东经） |
| 1 | `y` | 空间 | ° | 21.4-23.4 | 纬度（北纬） |
| 2 | `height` | 空间 | m | 变化范围 | 海拔高度 |
| **温度特征（动态）** ⭐ |
| 3 | `tmin` | 气象 | °C | 2.4-32.1 | 日最低气温 |
| 4 | `tmax` | 气象 | °C | 12.1-38.4 | **日最高气温（主要预测目标）** |
| 5 | `tave` | 气象 | °C | 6.7-28.7 | 日平均气温 |
| **气象要素（动态）** |
| 6 | `pre` | 气象 | mm | 0.0-200.0 | 日降水量 |
| 7 | `prs` | 气象 | hPa | 997.5-1018.2 | 气压（百帕） |
| 8 | `rh` | 气象 | % | 24.0-95.0 | 相对湿度 |
| 9 | `win` | 气象 | m/s | 1.0-5.8 | 风速 |
| **城市环境特征（静态）** |
| 10 | `BH` | 环境 | m | 12.8-19.6 | 建筑平均高度（1km缓冲区内） |
| 11 | `BHstd` | 环境 | m | 9.8-18.8 | 建筑高度标准差 |
| 12 | `SCD` | 环境 | - | 0.075-0.562 | 地表覆盖密度（Surface Cover Density） |
| 13 | `PLA` | 环境 | - | 0.051-0.482 | 路面铺装率（Paved Land Area ratio） |
| 14 | `λp` | 环境 | - | - | 天空开阔度参数λp |
| 15 | `λb` | 环境 | - | - | 天空开阔度参数λb |
| 16 | `POI` | 环境 | - | 0.108-0.118 | 兴趣点密度（Point of Interest density） |
| 17 | `POW` | 环境 | - | 0.0044-0.0046 | 工人口密度相关特征 |
| 18 | `POV` | 环境 | - | 0.25-0.888 | 车辆密度相关特征 |
| **植被特征** |
| 19 | `NDVI` | 环境 | - | 0.25-0.49 | 归一化植被指数（Normalized Difference Vegetation Index） |
| **ERA5再分析数据** |
| 20 | `surface_pressure` | 气象 | Pa | 99500-101100 | ERA5地表气压（帕斯卡，绝对值） |
| 21 | `surface_solar_radiation` | 气象 | J/m² | 1.5M-20M | ERA5地表太阳辐射下行通量（日累积） |
| 22 | `u_wind` | 气象 | m/s | - | 10m高度U分量风速（东西方向） |
| 23 | `v_wind` | 气象 | m/s | - | 10m高度V分量风速（南北方向） |
| 24 | `total_precipitation_sum` | 气象 | m | - | ERA5累计降水量（日累积） |
| **植被高度特征** ⭐新增 |
| 25 | `VegHeight_mean` | 环境 | m | - | 植被高度均值 |
| 26 | `VegHeight_std` | 环境 | m | - | 植被高度标准差 |
| **时间特征（转换为4维sin/cos编码）** ⭐⭐⭐ |
| 27 | `doy` | 时间 | - | 1-366 | 年内日序数（Day of Year） |
| 28 | `month` | 时间 | - | 1-12 | 月份 |

#### 2.1.3 特征分类汇总

**静态特征（站点固定，不随时间变化）:**
- 空间: `x, y, height` (0-2)
- 城市环境: `BH, BHstd, SCD, PLA, λp, λb, POI, POW, POV` (10-18)
- 植被: `NDVI, VegHeight_mean, VegHeight_std` (19, 25-26)

**动态特征（每天变化）:**
- 温度: `tmin, tmax, tave` (3-5)
- 气象要素: `pre, prs, rh, win` (6-9)
- ERA5数据: `surface_pressure, surface_solar_radiation, u_wind, v_wind, total_precipitation_sum` (20-24)

**时间特征（周期性）:**
- `doy, month` (27-28) → 在dataset.py中自动转换为4维sin/cos编码

#### 2.1.4 数据类型与存储

```python
dtype: np.float32  # 单精度浮点数，节省存储空间
size: ~9.6 MB      # 2922 × 28 × 29 × 4 bytes = 9,512,256 bytes
```

#### 2.1.5 访问示例

```python
import numpy as np

# 加载数据
data = np.load('data/real_weather_data_2010_2017.npy')
print(f"数据形状: {data.shape}")  # (2922, 28, 29)

# 获取第10天、所有气象站的最高气温
tmax_day10 = data[9, :, 4]  # shape: (28,) - 注意tmax索引为4

# 获取59264号气象站（索引0）2010年全年的温度序列
station_0_tmax = data[:365, 0, 4]  # shape: (365,)

# 获取所有站点、所有时间的经纬度（用于构建图）
lon = data[0, :, 0]  # shape: (28,) - 经度在时间维度上恒定
lat = data[0, :, 1]  # shape: (28,)

# 获取时间特征（doy和month）
doy_all = data[:, :, 27]   # shape: (2922, 28) - 年内日序数
month_all = data[:, :, 28] # shape: (2922, 28) - 月份

# 提取训练集数据（2010-2015年）
train_data = data[0:2191, :, :]  # shape: (2191, 28, 28)

# 提取验证集数据（2016年）
val_data = data[2191:2557, :, :]  # shape: (366, 28, 28)

# 提取测试集数据（2017年）
test_data = data[2557:2922, :, :]  # shape: (365, 28, 28)
```

### 2.2 气象站信息数组 (`station_info.npy`)

#### 2.2.1 数组维度

```python
shape: [气象站数, 信息维度] = [28, 4]
```

- **维度0**: 28个气象站，按站点ID升序排列
- **维度1**: 4个信息字段（ID, 经度, 纬度, 高度）

#### 2.2.2 字段说明

| 索引 | 字段 | 说明 |
|:----:|------|------|
| 0 | Station ID | WMO站点编码 (59264-59493) |
| 1 | Longitude | 经度 (°E, 111.0-114.0) |
| 2 | Latitude | 纬度 (°N, 21.4-23.4) |
| 3 | Height | 海拔高度 (m) |

#### 2.2.3 访问示例

```python
import numpy as np

# 加载气象站信息
station_info = np.load('data/station_info.npy')
print(f"气象站信息形状: {station_info.shape}")  # (28, 4)

# 获取所有站点ID
station_ids = station_info[:, 0].astype(int)
print(f"站点数量: {len(station_ids)}")  # 28

# 获取所有站点的经纬度
coords = station_info[:, 1:3]  # shape: (28, 2)
lon = station_info[:, 1]       # shape: (28,)
lat = station_info[:, 2]       # shape: (28,)

# 查找特定站点的信息
target_id = 59264
idx = np.where(station_info[:, 0] == target_id)[0][0]
station_info_59264 = station_info[idx]
print(f"站点{target_id}: 经度={station_info_59264[1]:.2f}°, "
      f"纬度={station_info_59264[2]:.2f}°, "
      f"高度={station_info_59264[3]:.2f}m")

# 计算站点间距离（用于构建图）
from scipy.spatial.distance import cdist
distances = cdist(coords, coords, metric='euclidean')  # shape: (28, 28)
```

---

## 3. CSV原始数据格式

### 3.1 文件基本信息

- **编码**: UTF-8 (带BOM)
- **分隔符**: 逗号 (,)
- **每行含义**: 一个气象站在某一天的观测数据
- **数据组织**: 按日期和气象站ID排列

### 3.2 CSV字段定义（30个列）

CSV文件包含30个列（包括id, doy, month等元数据），NPY数组保留29个特征列（排除id）。

| CSV列 | NPY索引 | 字段名 | 数据类型 | 单位 | 说明 |
|:----:|:------:|--------|---------|------|------|
| 1 | - | `id` | int | - | 气象站WMO编码（转换时排除） |
| 2 | 0 | `x` | float | ° | 经度 |
| 3 | 1 | `y` | float | ° | 纬度 |
| 4 | 2 | `height` | float | m | 海拔高度 |
| 5 | 3 | `tmin` | float | °C | 日最低气温 |
| 6 | 4 | `tmax` | float | °C | 日最高气温 |
| 7 | 5 | `tave` | float | °C | 日平均气温 |
| 8 | 6 | `pre` | float | mm | 日降水量 |
| 9 | 7 | `prs` | float | hPa | 气压 |
| 10 | 8 | `rh` | float | % | 相对湿度 |
| 11 | 9 | `win` | float | m/s | 风速 |
| 12 | 10 | `BH` | float | m | 建筑平均高度 |
| 13 | 11 | `BHstd` | float | m | 建筑高度标准差 |
| 14 | 12 | `SCD` | float | - | 地表覆盖密度 |
| 15 | 13 | `PLA` | float | - | 路��铺装率 |
| 16 | 14 | `λp` | float | - | 天空开阔度参数λp |
| 17 | 15 | `λb` | float | - | 天空开阔度参数λb |
| 18 | 16 | `POI` | float | - | 兴趣点密度 |
| 19 | 17 | `POW` | float | - | 工人口密度 |
| 20 | 18 | `POV` | float | - | 车辆密度 |
| 21 | 19 | `NDVI` | float | - | 归一化植被指数 |
| 22 | 20 | `surface_pressure` | float | Pa | ERA5地表气压 |
| 23 | 21 | `surface_solar_radiation` | float | J/m² | ERA5太阳辐射 |
| 24 | 22 | `u_component_of_wind_10m` | float | m/s | U分量风速 |
| 25 | 23 | `v_component_of_wind_10m` | float | m/s | V分量风速 |
| 26 | 24 | `total_precipitation_sum` | float | m | ERA5累计降水量 |
| 27 | 25 | `VegHeight_mean` | float | m | 植被高度均值 |
| 28 | 26 | `VegHeight_std` | float | m | 植被高度标准差 |
| 29 | 27 | `doy` | int | - | 年内日序数 |
| 30 | 28 | `month` | int | - | 月份 |

### 3.3 数据示例

```csv
id,x,y,height,tmin,tmax,tave,pre,prs,rh,win,BH,BHstd,SCD,PLA,λp,λb,POI,POW,POV,NDVI,surface_pressure,surface_solar_radiation,u_wind,v_wind,total_precipitation_sum,VegHeight_mean,VegHeight_std,doy,month
59264,111.51,23.4,45.3,11.2,18.3,14.5,3.4,1008.5,92.0,1.2,12.81,9.87,0.075,0.051,0.15,0.12,0.108,0.0044,0.888,0.459,99701.5,5246494.0,2.3,1.1,8.5,2.1,1,1
59264,111.51,23.4,45.3,12.1,19.9,15.8,7.8,1007.3,94.0,2.3,12.81,9.87,0.075,0.051,0.15,0.12,0.108,0.0044,0.888,0.457,99749.5,1576552.0,3.1,0.9,8.5,2.1,2,1
59269,111.82,23.21,28.6,10.8,19.2,14.9,2.1,1009.1,88.0,1.5,15.34,12.45,0.124,0.089,0.18,0.14,0.112,0.0045,0.756,0.412,99823.2,5389721.0,1.8,1.3,9.2,2.3,1,1
...
```

---

## 4. 4维时间周期编码

### 4.1 为什么需要时间周期编码?

原始时间特征`doy`(1-366)和`month`(1-12)是离散数值,存在以下问题:
1. **不连续性**: 1月(1)和12月(12)数值差异大,但实际上相邻
2. **线性假设**: 神经网络会假设doy=2比doy=1高一倍的关系,这是错误的
3. **周期性丢失**: 无法体现年度周期和季节周期

### 4.2 编码原理

使用**sin/cos变换**将离散时间特征转换为连续周期性编码:

```python
# 年周期编码（doy: 1-366 → doy_sin, doy_cos）
year_phase = 2π × (doy - 1) / days_in_year  # 闰年366,平年365
doy_sin = sin(year_phase)
doy_cos = cos(year_phase)

# 月周期编码（month: 1-12 → month_sin, month_cos）
month_phase = 2π × (month - 1) / 12
month_sin = sin(month_phase)
month_cos = cos(month_phase)
```

**优势:**
- ✅ **周期连续性**: 12月31日和1月1日的编码值相近
- ✅ **唯一性**: sin/cos组合可唯一确定时间点
- ✅ **平滑性**: 相邻时间点的编码值平滑变化
- ✅ **模型友好**: 更适合神经网络学习周期性规律

### 4.3 编码流程

在`myGNN/dataset.py`中自动完成:

```python
# 步骤1: 提取原始时间特征
doy = MetData[:, :, 27]    # shape: (2922, 28)
month = MetData[:, :, 28]  # shape: (2922, 28)

# 步骤2: 计算周期相位
T, N, _ = MetData.shape
days_in_year = np.array([365 if year % 4 != 0 else 366
                         for year in range(2010, 2018)])  # 处理闰年
year_phase = 2 * np.pi * (doy - 1) / days_in_year[:, None]
month_phase = 2 * np.pi * (month - 1) / 12

# 步骤3: sin/cos变换
doy_sin = np.sin(year_phase).astype(np.float32)
doy_cos = np.cos(year_phase).astype(np.float32)
month_sin = np.sin(month_phase).astype(np.float32)
month_cos = np.cos(month_phase).astype(np.float32)

# 步骤4: 移除原始时间特征（索引27-28）
base_features = MetData[:, :, :27]  # shape: (2922, 28, 27)

# 步骤5: 拼接4维时间编码
temporal_encoding = np.stack([doy_sin, doy_cos, month_sin, month_cos],
                             axis=-1)  # shape: (2922, 28, 4)
final_data = np.concatenate([base_features, temporal_encoding],
                            axis=-1)  # shape: (2922, 28, 31)

# 最终输入维度: 27 (基础特征) + 4 (时间编码) = 31
```

### 4.4 特征维度变化

```
原始数据: [2922, 28, 29]
     │
     ├─ 移除doy和month (索引27-28)
     │   → [2922, 28, 27] 基础特征
     │
     ├─ 生成4维时间编码
     │   - doy_sin, doy_cos (年周期)
     │   - month_sin, month_cos (月周期)
     │   → [2922, 28, 4] 时间编码
     │
     └─ 拼接
         → [2922, 28, 31] 最终输入
```

### 4.5 代码示例

```python
from myGNN.dataset import WeatherGraphDataset, create_dataloaders

# 加载数据
MetData = np.load('data/real_weather_data_2010_2017.npy')  # [2922, 28, 29]

# 创建数据集（自动进行4维时间编码）
train_loader, val_loader, test_loader = create_dataloaders(
    config, graph, MetData,
    batch_size=32, shuffle_train=True
)

# 查看数据维度
for batch in train_loader:
    print(f"节点特征形状: {batch.x.shape}")
    # 输出: torch.Size([28, 7, 31]) - [nodes, hist_len, features]
    # 31 = 27 (基础特征) + 4 (时间编码)
    break
```

---

## 5. 使用示例

### 5.1 基础数据加载

```python
import numpy as np

# 加载数据
data = np.load('data/real_weather_data_2010_2017.npy')
station_info = np.load('data/station_info.npy')

print(f"数据形状: {data.shape}")           # (2922, 28, 29)
print(f"气象站信息: {station_info.shape}") # (28, 4)
print(f"特征数量: {data.shape[2]}")        # 29
```

### 5.2 数据探索

```python
# 1. 获取特定特征的时间序列
tmax_all = data[:, :, 4]  # 所有站点的最高气温，shape: (2922, 28)

# 2. 计算全局统计量
print(f"最高气温范围: {tmax_all.min():.2f}°C - {tmax_all.max():.2f}°C")
print(f"最高气温均值: {tmax_all.mean():.2f}°C")
print(f"最高气温标准差: {tmax_all.std():.2f}°C")

# 3. 提取单个气象站的数据
station_0_data = data[:, 0, :]  # 第一个气象站，shape: (2922, 28)

# 4. 提取某一天的所有站点数据
day_100_data = data[99, :, :]  # 第100天，shape: (28, 28)

# 5. 分析季节性规律
import matplotlib.pyplot as plt

# 绘制2017年（测试集）的温度曲线
test_tmax = data[2557:2922, :, 4]  # 测试集最高气温
plt.figure(figsize=(12, 4))
plt.plot(test_tmax.mean(axis=1))
plt.title('2017年平均最高气温')
plt.xlabel('天数')
plt.ylabel('温度 (°C)')
plt.grid(True)
plt.show()
```

### 5.3 在myGNN框架中使用

```python
from myGNN.config import create_config, print_config
from myGNN.dataset import create_dataloaders
from myGNN.graph.distance_graph import create_graph_from_config
import numpy as np

# 1. 创建配置
config, arch_config, loss_config = create_config()

# 2. 加载数据
MetData = np.load(config.MetData_fp)  # [2922, 28, 28]
station_info = np.load(config.station_info_fp)  # [28, 4]

# 3. 构建图结构
graph = create_graph_from_config(config, station_info)

# 4. 创建数据加载器（自动进行4维时间编码）
train_loader, val_loader, test_loader = create_dataloaders(
    config, graph, MetData,
    batch_size=config.batch_size,
    shuffle_train=True
)

# 5. 查看数据维度
for batch in train_loader:
    print(f"批次大小: {batch.num_graphs}")
    print(f"节点特征: {batch.x.shape}")  # [nodes, hist_len, 31]
    print(f"边索引: {batch.edge_index.shape}")  # [2, num_edges]
    print(f"标签: {batch.y.shape}")  # [nodes, pred_len]
    break

# 最终输入维度说明:
# batch.x.shape[-1] = 31
# = 27 (基础特征: x,y,height,tmin,tmax,tave,pre,prs,rh,win,
#               BH,BHstd,SCD,PLA,λp,λb,POI,POW,POV,NDVI,
#               surface_pressure,surface_solar_radiation,u_wind,v_wind,
#               total_precipitation_sum,VegHeight_mean,VegHeight_std)
# + 4 (时间编码: doy_sin,doy_cos,month_sin,month_cos)
```

### 5.4 数据标准化

```python
# 计算训练集的均值和标准差
train_data = data[config.train_start:config.train_end]  # 2010-2015年

# 对每个特征分别标准化
feature_means = train_data.mean(axis=(0, 1))  # shape: (28,)
feature_stds = train_data.std(axis=(0, 1))    # shape: (28,)

# 标准化所有数据
data_normalized = (data - feature_means) / (feature_stds + 1e-8)

# 仅标准化目标变量（tmax）
tmax_mean = train_data[:, :, 4].mean()
tmax_std = train_data[:, :, 4].std()
print(f"tmax均值: {tmax_mean:.2f}°C, 标准差: {tmax_std:.2f}°C")

# 保存标准化参数（用于反标准化）
np.savez('data/normalization_params.npz',
         feature_means=feature_means,
         feature_stds=feature_stds,
         tmax_mean=tmax_mean,
         tmax_std=tmax_std)

# 反标准化
def denormalize(normalized, mean, std):
    return normalized * std + mean

# 使用示例
predictions_normalized = model(test_data)  # 模型输出
predictions = denormalize(predictions_normalized, tmax_mean, tmax_std)
```

### 5.5 特征选择

```python
from myGNN.config import Config

config = Config()

# 方法1: 使用所有基础特征（默认）
config.feature_indices = None
# 自动使用索引0-26（移除doy和month）
# 加上4维时间编码，最终维度=31

# 方法2: 仅使用核心气象特征
config.feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 0-2: x, y, height (空间特征)
# 3-5: tmin, tmax, tave (温度)
# 6-9: pre, prs, rh, win (气象要素)
# 加上4维时间编码，最终维度=14

# 方法3: 使用所有特征（包括城市环境）
config.feature_indices = list(range(27))  # 0-26
# 加上4维时间编码，最终维度=31

# 方法4: 自定义特征组合
config.feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 25, 26]
# 核心气象 + NDVI + 植被高度
# 加上4维时间编码，最终维度=17
```

### 5.6 构建空间图

```python
from myGNN.graph.distance_graph import create_graph_from_config

# 方法1: 使用配置自动构建（推荐）
graph = create_graph_from_config(config, station_info)

# 方法2: 手动构建K近邻逆距离图
from myGNN.graph.distance_graph import Graph_inv_dis

lon = station_info[:, 1]
lat = station_info[:, 2]

graph = Graph_inv_dis(
    lon=lon,
    lat=lat,
    top_neighbors=10,      # K近邻数量
    use_edge_attr=True     # 使用逆距离作为边权重
)

# 图对象属性
print(f"边索引: {graph.edge_index.shape}")  # [2, num_edges]
print(f"边权重: {graph.edge_attr.shape}")   # [num_edges, 1]
print(f"节点数: {graph.num_nodes}")         # 28
print(f"边数: {graph.num_edges}")           # 约280 (28×10)
```

---

## 6. 数据处理流程

### 6.1 完整数据流

```
CSV原始数据 (8个文件, 2010-2017年)
     │
     ├─ [data/convert_real_data.py]
     │   ├─ 读取并合并所有CSV
     │   ├─ 提取气象站信息
     │   ├─ 转换为NPY数组 [2922, 28, 29]
     │   └─ 保存NPY文件
     │
     ↓
NPY数组文件
├── real_weather_data_2010_2017.npy [2922, 28, 29]
└── station_info.npy [28, 4]
     │
     ├─ [myGNN/dataset.py]
     │   ├─ 移除doy和month (索引27-28)
     │   │   → [2922, 28, 27] 基础特征
     │   │
     │   ├─ 生成4维时间编码
     │   │   - doy → doy_sin, doy_cos (年周期)
     │   │   - month → month_sin, month_cos (月周期)
     │   │   → [2922, 28, 4] 时间编码
     │   │
     │   ├─ 拼接特征
     │   │   → [2922, 28, 31] 完整特征
     │   │
     │   ├─ 滑动窗口切分
     │   │   hist_len=7, pred_len=3
     │   │   → 输入: [hist_len, 30]
     │   │   → 输出: [pred_len]
     │   │
     │   ├─ 数据集划分
     │   │   - 训练集: 0-2190 (2010-2015)
     │   │   - 验证集: 2191-2556 (2016)
     │   │   - 测试集: 2557-2921 (2017)
     │   │
     │   └─ 标准化（基于训练集统计量）
     │
     ↓
DataLoader批次
├── batch.x: [nodes, hist_len, 31] 节点特征
├── batch.y: [nodes, pred_len] 目标标签
├── batch.edge_index: [2, num_edges] 边索引
└── batch.edge_attr: [num_edges, 1] 边权重
     │
     ├─ [myGNN/models/]
     │   - GAT_LSTM / GSAGE_LSTM / GAT_SeparateEncoder
     │
     ↓
模型预测输出 [batch×nodes, pred_len]
```

### 6.2 数据转换脚本使用

```bash
# 运行数据转换脚本
cd data
python convert_real_data.py
```

**脚本输出示例:**
```
================================================================================
真实气象数据转换脚本
================================================================================

数据目录: C:\Users\wxb55\Desktop\gnn_predict\data\result
输出目录: C:\Users\wxb55\Desktop\gnn_predict\data

--------------------------------------------------------------------------------
步骤 1/4: 加载CSV文件
--------------------------------------------------------------------------------
找到 8 个CSV文件:
  - merged_data_2010_2000m.csv
  - merged_data_2011_2000m.csv
  - merged_data_2012_2000m.csv
  - merged_data_2013_2000m.csv
  - merged_data_2014_2000m.csv
  - merged_data_2015_2000m.csv
  - merged_data_2016_2000m.csv
  - merged_data_2017_2000m.csv

合并后数据: 81816行
时间范围: 2010-01-01 至 2017-12-31
气象站数量: 28

--------------------------------------------------------------------------------
步骤 2/4: 提取气象站信息
--------------------------------------------------------------------------------
气象站信息:
  数量: 28
  ID范围: 59264 - 59493
  经度范围: 111.00° - 114.00°
  纬度范围: 21.40° - 23.40°

--------------------------------------------------------------------------------
步骤 3/4: 转换为NPY数组格式
--------------------------------------------------------------------------------
数组维度:
  时间步数: 2922
  气象站数: 28
  特征数: 29

数据统计:
  形状: (2922, 28, 29)
  数据类型: float32
  最小值: 0.00
  最大值: 20000000.00

--------------------------------------------------------------------------------
步骤 4/4: 保存文件
--------------------------------------------------------------------------------
✓ 保存数据数组: data/real_weather_data_2010_2017.npy
  形状: (2922, 28, 29)
  大小: 9.17 MB

✓ 保存气象站信息: data/station_info.npy
  形状: (28, 4)
  大小: 0.44 KB

================================================================================
转换完成!
================================================================================
```

### 6.3 数据验证

```python
import numpy as np

# 1. 加载数据
data = np.load('data/real_weather_data_2010_2017.npy')
station_info = np.load('data/station_info.npy')

# 2. 检查形状
assert data.shape == (2922, 28, 29), f"数据形状错误: {data.shape}"
assert station_info.shape == (28, 4), f"气象站信息形状错误: {station_info.shape}"

# 3. 检查缺失值
assert not np.isnan(data).any(), "存在缺失值"
assert not np.isnan(station_info).any(), "气象站信息存在缺失值"

# 4. 检查数值范围
tmax = data[:, :, 4]
assert tmax.min() > -20 and tmax.max() < 50, f"温度范围异常: {tmax.min():.2f} - {tmax.max():.2f}"

# 5. 检查时间连续性
doy = data[:, 0, 27]  # 第一个站点的doy序列
print(f"DOY范围: {doy.min():.0f} - {doy.max():.0f}")
assert doy.min() >= 1 and doy.max() <= 366, "DOY范围异常"

# 6. 检查站点ID
station_ids = station_info[:, 0].astype(int)
assert len(np.unique(station_ids)) == 28, "站点ID不唯一"
assert station_ids.min() >= 59000 and station_ids.max() < 60000, "站点ID范围异常"

print("✓ 所有验证通过!")
```

---

## 7. 常见问题

### Q1: 为什么NPY文件比CSV小?

**A:** NPY使用二进制存储,而CSV是文本格式。二进制存储更紧凑,且NPY使用float32(4字节)而非文本字符串。

**对比:**
- CSV: ~20-30 MB (8个文件合计, 文本格式)
- NPY: ~9.2 MB (单个文件, 二进制格式)

### Q2: 如何处理闰年数据?

**A:** 转换脚本和dataset.py自动处理闰年:
- 2012年和2016年有366天
- 其他年份365天
- 总计2922天 = 365×6 + 366×2

在4维时间编码时,使用动态days_in_year:
```python
days_in_year = 366 if year % 4 == 0 else 365
year_phase = 2 * np.pi * (doy - 1) / days_in_year
```

### Q3: 特征索引为什么与CSV列不完全对应?

**A:** NPY格式移除了元数据列:
- CSV包含29列（含id, doy, month等）
- NPY保留28列（排除id）
- dataset.py进一步处理时移除doy和month(索引27-28),转换为4维sin/cos编码
- 最终输入维度: 27(基础特征) + 4(时间编码) = 31

### Q4: 如何选择预测目标?

**A:** 框架支持多种预测目标:
- **tmax**(索引4): 日最高气温（默认推荐）
- tmin(索引3): 日最低气温
- tave(索引5): 日平均气温

配置方法:
```python
config.target_feature_idx = 4  # tmax
```

### Q5: 数据需要标准化吗?

**A:** 强烈推荐标准化,原因:
- 特征量纲差异大（温度°C vs 太阳辐射J/m²）
- 加速模型收敛
- 提高数值稳定性

**标准化方法:**
```python
# 基于训练集统计量
normalized = (value - train_mean) / train_std

# 预测后反标准化
actual = normalized * train_std + train_mean
```

### Q6: 为什么使用4维时间编码而不是直接使用doy和month?

**A:** 4维sin/cos编码的优势:
- ✅ 捕获周期性规律（年周期、月周期）
- ✅ 避免离散特征的跳跃（12月→1月连续）
- ✅ 保持相邻时间点的连续性
- ✅ 更适合神经网络学习

**对比:**
```python
# 原始编码问题
doy = [1, 2, ..., 365, 366, 1, 2, ...]  # 365→1跳跃
month = [12, 1, 2, ...]  # 12→1跳跃

# sin/cos编码
doy_sin, doy_cos  # 连续周期函数,365和1的编码相近
month_sin, month_cos  # 12月和1月的编码相近
```

### Q7: 如何使用自己的数据?

**A:** 准备数据:
1. **格式要求**: NPY数组 `[time_steps, num_stations, features]`
2. **气象站信息**: NPY数组 `[num_stations, 4]` (ID, 经度, 纬度, 高度)
3. **最少特征**: 至少包含预测目标和基本空间特征(x, y)

**配置修改:**
```python
# myGNN/config.py
config.MetData_fp = 'data/my_weather_data.npy'
config.station_info_fp = 'data/my_station_info.npy'
config.node_num = 你的站点数量
config.base_feature_dim = 你的特征数量 - 2  # 排除doy和month
config.target_feature_idx = 你的预测目标索引
```

### Q8: 如何查看NPY文件内容?

**A:** 使用Python查看:
```python
import numpy as np

# 加载数据
data = np.load('data/real_weather_data_2010_2017.npy')

# 查看形状和统计信息
print(f"形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"最小值: {data.min()}")
print(f"最大值: {data.max()}")
print(f"均值: {data.mean()}")

# 查看第一个时间步、第一个站点的数据
print(data[0, 0, :])

# 查看特定特征
tmax_all = data[:, :, 4]
print(f"tmax形状: {tmax_all.shape}")
print(f"tmax范围: {tmax_all.min():.2f}°C - {tmax_all.max():.2f}°C")
```

### Q9: 训练/验证/测试集如何划分?

**A:** 按年份划分,避免时间泄露:

| 数据集 | 年份 | 索引范围 | 天数 | 占比 |
|-------|------|---------|------|------|
| 训练集 | 2010-2015 | 0-2190 | 2191 | 75% |
| 验证集 | 2016 | 2191-2556 | 366 | 12.5% |
| 测试集 | 2017 | 2557-2921 | 365 | 12.5% |

**配置:**
```python
config.train_start = 0
config.train_end = 2191
config.val_start = 2191
config.val_end = 2557
config.test_start = 2557
config.test_end = 2922
```

### Q10: 如何对比不同特征组合的效果?

**A:** 修改`config.feature_indices`进行消融实验:

```python
# 实验1: 仅核心气象特征
config.feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10特征
# 训练并记录结果: RMSE_1, MAE_1

# 实验2: 核心气象 + 城市环境
config.feature_indices = list(range(19))  # 19特征
# 训练并记录结果: RMSE_2, MAE_2

# 实验3: 所有特征
config.feature_indices = None  # 26特征
# 训练并记录结果: RMSE_3, MAE_3

# 对比分析
# 如果RMSE_2 < RMSE_1,说明城市环境特征有效
# 如果RMSE_3 ≈ RMSE_2,说明ERA5和植被特征贡献较小
```

---

## 附录

### A. 特征完整列表（29个）

| 索引 | 特征名 | 类型 | 单位 | 是否静态 | 说明 |
|:----:|--------|------|:----:|:--------:|------|
| 0 | x | 空间 | ° | ✓ | 经度 |
| 1 | y | 空间 | ° | ✓ | 纬度 |
| 2 | height | 空间 | m | ✓ | 海拔高度 |
| 3 | tmin | 气象 | °C | ✗ | 日最低气温 |
| 4 | tmax | 气象 | °C | ✗ | **日最高气温（主要预测目标）** |
| 5 | tave | 气象 | °C | ✗ | 日平均气温 |
| 6 | pre | 气象 | mm | ✗ | 日降水量 |
| 7 | prs | 气象 | hPa | ✗ | 气压 |
| 8 | rh | 气象 | % | ✗ | 相对湿度 |
| 9 | win | 气象 | m/s | ✗ | 风速 |
| 10 | BH | 环境 | m | ✓ | ���筑平均高度 |
| 11 | BHstd | 环境 | m | ✓ | 建筑高度标准差 |
| 12 | SCD | 环境 | - | ✓ | 地表覆盖密度 |
| 13 | PLA | 环境 | - | ✓ | 路面铺装率 |
| 14 | λp | 环境 | - | ✓ | 天空开阔度参数λp |
| 15 | λb | 环境 | - | ✓ | 天空开阔度参数λb |
| 16 | POI | 环境 | - | ✓ | 兴趣点密度 |
| 17 | POW | 环境 | - | ✓ | 工人口密度 |
| 18 | POV | 环境 | - | ✓ | 车辆密度 |
| 19 | NDVI | 环境 | - | ✓ | 归一化植被指数 |
| 20 | surface_pressure | 气象 | Pa | ✗ | ERA5地表气压 |
| 21 | surface_solar_radiation | 气象 | J/m² | ✗ | ERA5太阳辐射 |
| 22 | u_wind | 气象 | m/s | ✗ | 10m高度U分量风速 |
| 23 | v_wind | 气象 | m/s | ✗ | 10m高度V分量风速 |
| 24 | total_precipitation_sum | 气象 | m | ✗ | ERA5累计降水量 |
| 25 | VegHeight_mean | 环境 | m | ✓ | 植被高度均值 |
| 26 | VegHeight_std | 环境 | m | ✓ | 植被高度标准差 |
| 27 | doy | 时间 | - | ✗ | 年内日序数（转换为sin/cos） |
| 28 | month | 时间 | - | ✗ | 月份（转换为sin/cos） |

**说明:**
- ✓ 静态特征: 每个站点固定,不随时间变化
- ✗ 动态特征: 每天变化

### B. 相关文档

- [README.md](README.md) - 项目总览
- [CLAUDE.md](CLAUDE.md) - 项目架构详细说明 ⭐
- [myGNN/README.md](myGNN/README.md) - myGNN框架文档
- [data/README.md](data/README.md) - 数据目录说明
- [myGNN/config.py](myGNN/config.py) - 配置文件
- [myGNN/dataset.py](myGNN/dataset.py) - 数据集类（4维时间编码实现）
- [data/convert_real_data.py](data/convert_real_data.py) - 数据转换脚本

### C. 参考资料

- **ERA5-Land文档**: https://confluence.ecmwf.int/display/CKB/ERA5-Land
- **PyTorch Geometric数据格式**: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
- **论文**: *Intra-city Scale Graph Neural Networks Enhance Short-term Air Temperature Forecasting*
- **时间编码**: Vaswani et al. "Attention is All You Need" (Positional Encoding)

### D. 更新日志

#### v3.1 (2026-03-28)
- ✨ 更新为29特征格式 [2922, 28, 29]
- ✨ 新增 total_precipitation_sum（ERA5累计降水，索引24）
- 📝 VegHeight 索引调整为 25-26，doy/month 调整为 27-28

#### v3.0 (2025-12-16)
- ✨ 更新为28特征格式 [2922, 28, 28]
- ✨ 新增植被高度特征 (VegHeight_mean, VegHeight_std)
- ✨ 详细说明4维时间周期编码原理和实现
- ✨ 新增数据集划分说明（训练/验证/测试）
- ✨ 完善使用示例和FAQ
- 📝 重构文档结构,更清晰的目录组织

#### v2.0 (2025-11)
- 更新为26特征格式 [2922, 28, 26]
- 添加ERA5再分析数据
- 完善数据验证流程

#### v1.0 (初始版本)
- 基础数据格式说明
- CSV到NPY转换流程

---

**文档版本**: v3.0
**最后更新**: 2025-12-16
**维护者**: GNN气温预测项目组
