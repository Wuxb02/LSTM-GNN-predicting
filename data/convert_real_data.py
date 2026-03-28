"""
数据转换脚本: 将CSV格式的真实气象数据转换为模型支持的NPY格式

输入: data/result/merged_data_YYYY_2000m.csv (2010-2017年, 8个文件)
输出:
    - data/real_weather_data_2010_2017.npy: [时间步, 气象站数, 28]
    - data/station_info.npy: [气象站数, 4] (ID, 经度, 纬度, 高度)

特征维度说明 (共29个特征):
    0-1: x, y (经纬度)
    2: height (海拔高度)
    3-5: tmin, tmax, tave (温度)
    6-9: pre, prs, rh, win (气象要素)
    10-11: BH, BHstd (建筑高度特征)
    12-13: SCD, PLA (地表覆盖)
    14-15: λp, λb (天空开阔度参数)
    16-18: POI, POW, POV (兴趣点/人口密度)
    19: NDVI (植被指数)
    20-21: surface_pressure, surface_solar_radiation (ERA5)
    22-23: u_component_of_wind_10m, v_component_of_wind_10m (风速分量)
    24: total_precipitation_sum (ERA5累计降水)
    25-26: VegHeight_mean, VegHeight_std (植被高度特征)
    27-28: doy, month (时间特征)

作者: GNN气温预测项目
日期: 2025
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def load_and_merge_csv_files(data_dir: str) -> pd.DataFrame:
    """
    加载并合并所有年份的CSV文件

    Args:
        data_dir: 数据目录路径

    Returns:
        合并后的DataFrame，按日期和气象站ID排序
    """
    csv_pattern = os.path.join(data_dir, "merged_data_*_2000m.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        raise FileNotFoundError(f"未找到CSV文件: {csv_pattern}")

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")

    # 读取所有CSV文件
    df_list = []
    for file in csv_files:
        year = int(os.path.basename(file).split('_')[2])
        df = pd.read_csv(file, encoding='utf-8-sig')

        # 使用year和doy构建日期
        df['year'] = year
        df['date'] = pd.to_datetime(year * 1000 + df['doy'], format='%Y%j')

        df_list.append(df)
        print(f"  {year}年: {len(df)}行, {df['id'].nunique()}个气象站")

    # 合并所有数据
    df_merged = pd.concat(df_list, ignore_index=True)

    # 按日期和气象站ID排序
    df_merged = df_merged.sort_values(['date', 'id']).reset_index(drop=True)

    print(f"\n合并后数据: {len(df_merged)}行")
    print(f"时间范围: {df_merged['date'].min()} 至 {df_merged['date'].max()}")
    print(f"气象站数量: {df_merged['id'].nunique()}")

    return df_merged


def extract_station_info(df: pd.DataFrame) -> np.ndarray:
    """
    提取气象站信息

    Args:
        df: 原始DataFrame

    Returns:
        station_info: [气象站数, 4] (ID, 经度, 纬度, 高度)
    """
    stations = df[['id', 'x', 'y', 'height']].drop_duplicates().sort_values('id')
    station_info = stations.values

    print(f"\n气象站信息:")
    print(f"  数量: {len(station_info)}")
    print(f"  ID范围: {station_info[:, 0].min():.0f} - {station_info[:, 0].max():.0f}")
    print(f"  经度范围: {station_info[:, 1].min():.2f}° - {station_info[:, 1].max():.2f}°")
    print(f"  纬度范围: {station_info[:, 2].min():.2f}° - {station_info[:, 2].max():.2f}°")
    print(f"  高度范围: {station_info[:, 3].min():.2f}m - {station_info[:, 3].max():.2f}m")

    return station_info


def convert_to_npy_format(df: pd.DataFrame, station_info: np.ndarray) -> np.ndarray:
    """
    将DataFrame转换为NPY数组格式 [时间步, 气象站数, 特征数]

    Args:
        df: 原始DataFrame (已排序)
        station_info: 气象站信息

    Returns:
        data_array: [时间步, 气象站数, 29]
    """
    # 特征列（按顺序）
    feature_columns = [
        'x', 'y',           # 0-1: 经纬度
        'height',           # 2: 海拔高度
        'tmin', 'tmax', 'tave',  # 3-5: 温度
        'pre',              # 6: 降水
        'prs',              # 7: 气压
        'rh',               # 8: 相对湿度
        'win',              # 9: 风速
        'BH', 'BHstd',      # 10-11: 建筑高度特征
        'SCD', 'PLA',       # 12-13: 地表覆盖
        'λp', 'λb',         # 14-15: 天空开阔度参数
        'POI', 'POW', 'POV',  # 16-18: 兴趣点/人口密度
        'NDVI',             # 19: 植被指数
        'surface_pressure',  # 20: 地表气压
        'surface_solar_radiation_downwards',  # 21: 太阳辐射
        'u_component_of_wind_10m',  # 22: 10m高���U分量风速
        'v_component_of_wind_10m',  # 23: 10m高度V分量风速
        'total_precipitation_sum',  # 24: ERA5累计降水
        'VegHeight_mean',   # 25: 植被高度均值
        'VegHeight_std',    # 26: 植被高度标准差
        'doy', 'month'      # 27-28: 时间特征
    ]

    # 获取唯一日期和气象站
    unique_dates = sorted(df['date'].unique())
    station_ids = station_info[:, 0]

    n_timesteps = len(unique_dates)
    n_stations = len(station_ids)
    n_features = len(feature_columns)

    print(f"\n数组维度:")
    print(f"  时间步数: {n_timesteps}")
    print(f"  气象站数: {n_stations}")
    print(f"  特征数: {n_features}")

    # 初始化数组
    data_array = np.zeros((n_timesteps, n_stations, n_features), dtype=np.float32)

    # 构建站点ID到索引的映射
    station_id_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}

    # 填充数据
    for t, date in enumerate(unique_dates):
        date_data = df[df['date'] == date]

        for _, row in date_data.iterrows():
            station_idx = station_id_to_idx[row['id']]
            for f, col in enumerate(feature_columns):
                data_array[t, station_idx, f] = row[col]

    # 检查缺失值
    n_missing = np.isnan(data_array).sum()
    if n_missing > 0:
        print(f"\n警告: 发现 {n_missing} 个缺失值 ({n_missing / data_array.size * 100:.2f}%)")
        print("  使用前向填充处理缺失值...")

        # 沿时间轴填充缺失值
        for s in range(n_stations):
            for f in range(n_features):
                series = data_array[:, s, f]
                mask = np.isnan(series)
                if mask.any():
                    idx = np.where(~mask)[0]
                    if len(idx) > 0:
                        # 使用插值填充中间的缺失值
                        series[mask] = np.interp(
                            np.flatnonzero(mask),
                            idx,
                            series[idx]
                        )
                        # 处理开头的缺失值（后向填充）
                        first_valid = idx[0]
                        if first_valid > 0:
                            series[:first_valid] = series[first_valid]
                        # 处理结尾的缺失值（前向填充）
                        last_valid = idx[-1]
                        if last_valid < len(series) - 1:
                            series[last_valid + 1:] = series[last_valid]
                    else:
                        # 该特征在该站点全部为NaN，记录警告
                        print(f"    警告: 站点{s} 特征{f} 全部为NaN")
                    data_array[:, s, f] = series

        # 检查是否还有剩余的NaN
        remaining_nan = np.isnan(data_array).sum()
        if remaining_nan > 0:
            print(f"\n  填充后仍有 {remaining_nan} 个NaN值")

    # 数据统计
    print(f"\n数据统计:")
    print(f"  形状: {data_array.shape}")
    print(f"  数据类型: {data_array.dtype}")
    print(f"  最小值: {data_array.min():.2f}")
    print(f"  最大值: {data_array.max():.2f}")
    print(f"  平均值: {data_array.mean():.2f}")
    print(f"  标准差: {data_array.std():.2f}")

    # 各特征统计
    print(f"\n各特征统计:")
    for i, col in enumerate(feature_columns):
        feat_data = data_array[:, :, i]
        print(f"  {i:2d}. {col:40s} 范围: [{feat_data.min():10.2f}, {feat_data.max():10.2f}]  "
              f"均值: {feat_data.mean():10.2f}  标准差: {feat_data.std():10.2f}")

    return data_array


def main():
    """主函数"""
    # 当前脚本在data文件夹下
    script_dir = Path(__file__).parent
    data_dir = script_dir / "result"
    output_dir = script_dir

    print("=" * 80)
    print("真实气象数据转换脚本")
    print("=" * 80)
    print(f"\n数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 1. 加载并合并CSV文件
    print("\n" + "-" * 80)
    print("步骤 1/4: 加载CSV文件")
    print("-" * 80)
    df = load_and_merge_csv_files(str(data_dir))

    # 2. 提取气象站信息
    print("\n" + "-" * 80)
    print("步骤 2/4: 提取气象站信息")
    print("-" * 80)
    station_info = extract_station_info(df)

    # 3. 转换为NPY格式
    print("\n" + "-" * 80)
    print("步骤 3/4: 转换为NPY数组格式")
    print("-" * 80)
    data_array = convert_to_npy_format(df, station_info)

    # 4. 保存文件
    print("\n" + "-" * 80)
    print("步骤 4/4: 保存文件")
    print("-" * 80)

    output_data_path = output_dir / "real_weather_data_2010_2017.npy"
    output_station_path = output_dir / "station_info.npy"

    np.save(output_data_path, data_array)
    print(f"✓ 保存数据数组: {output_data_path}")
    print(f"  形状: {data_array.shape}")
    print(f"  大小: {output_data_path.stat().st_size / 1024 / 1024:.2f} MB")

    np.save(output_station_path, station_info)
    print(f"✓ 保存气象站信息: {output_station_path}")
    print(f"  形状: {station_info.shape}")

    print("\n" + "=" * 80)
    print("转换完成!")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  1. {output_data_path.name}")
    print(f"     - 形状: [时间步={data_array.shape[0]}, 气象站={data_array.shape[1]}, 特征={data_array.shape[2]}] (含total_precipitation_sum)")
    print(f"     - 时间范围: 2010-01-01 至 2017-12-31 ({data_array.shape[0]} 天)")
    print(f"  2. {output_station_path.name}")
    print(f"     - 形状: [气象站={station_info.shape[0]}, 信息={station_info.shape[1]}] (ID, 经度, 纬度, 高度)")

    print(f"\n使用示例:")
    print(f"  import numpy as np")
    print(f"  data = np.load('data/{output_data_path.name}')")
    print(f"  station_info = np.load('data/{output_station_path.name}')")
    print(f"  print(f'数据形状: {{data.shape}}')")


if __name__ == "__main__":
    main()
