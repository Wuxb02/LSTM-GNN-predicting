# -*- coding: utf-8 -*-
"""
温度统计量空间分布图绘制脚本

完全照搬 visualize_results.py 中的 plot_rmse_spatial_map_with_basemap 函数
只替换数据为2010-2017年温度统计量（平均值或标准差）

使用方法:
    1. 修改配置区域的 STAT_TYPE 参数 ('mean' 或 'std')
    2. 修改配置区域的 TEMP_TYPE 参数 ('tmin', 'tmax', 'tave')
    3. 运行脚本: python plot_temperature_spatial_mean.py

输出:
    - figdraw/result/temperature_spatial_{stat_type}_{temp_type}.png
    - 示例: temperature_spatial_mean_tmax.png
    - 示例: temperature_spatial_std_tmin.png

作者: GNN气温预测项目
日期: 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 地理底图相关导入
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib_scalebar.scalebar import ScaleBar
    BASEMAP_AVAILABLE = True
except ImportError as e:
    BASEMAP_AVAILABLE = False
    print(f"[WARNING] 地理底图功能不可用: {e}")
    print(f"  如需使用底图,请安装: pip install cartopy matplotlib-scalebar")


# ==================== 配置区域 ====================
# 统计类型选择: 'mean' (平均值), 'std' (标准差)
STAT_TYPE = 'mean'

# 温度类型选择: 'tmin', 'tmax', 'tave'
TEMP_TYPE = 'tmax'

# 数据路径
DATA_PATH = project_root / 'data' / 'real_weather_data_2010_2017.npy'
STATION_INFO_PATH = project_root / 'data' / 'station_info.npy'

# 输出路径
OUTPUT_DIR = Path(__file__).parent / 'result'

# 地理底图配置（与visualize_results.py保持一致）
USE_BASEMAP = True
USE_WMTS_BASEMAP = False  # True: Mapbox在线底图, False: Natural Earth离线底图
ADD_SCALEBAR = False
ADD_NORTH_ARROW = False

# 图表配置（与visualize_results.py保持一致）
DPI = 300
# ==================================================


# 温度特征索引映射
TEMP_FEATURE_MAP = {
    'tmin': 3,
    'tmax': 4,
    'tave': 5
}

# 温度类型显示名称
TEMP_DISPLAY_NAME = {
    'tmin': 'Tmin',
    'tmax': 'Tmax',
    'tave': 'Tave'
}


# 统计类型函数映射
STAT_FUNCTIONS = {
    'mean': np.mean,
    'std': np.std
}

# 统计类型显示名称
STAT_DISPLAY_NAME = {
    'mean': '',
    'std': ' '
}

# 统计类型单位后缀
STAT_UNIT_SUFFIX = {
    'mean': '',
    'std': ''
}


def load_data(temp_type, stat_type):
    """
    加载数据并计算温度统计量

    Args:
        temp_type: 温度类型 ('tmin', 'tmax', 'tave')
        stat_type: 统计类型 ('mean', 'std')

    Returns:
        tuple: (temp_stat, lon, lat, num_stations)
    """
    # 验证温度类型
    if temp_type not in TEMP_FEATURE_MAP:
        raise ValueError(
            f"无效的温度类型: {temp_type}\n"
            f"支持的类型: {list(TEMP_FEATURE_MAP.keys())}"
        )

    # 验证统计类型
    if stat_type not in STAT_FUNCTIONS:
        raise ValueError(
            f"无效的统计类型: {stat_type}\n"
            f"支持的类型: {list(STAT_FUNCTIONS.keys())}"
        )

    # 加载主数据
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

    print(f"正在加载数据: {DATA_PATH}")
    data = np.load(DATA_PATH)
    print(f"  数据形状: {data.shape}")

    # 验证数据形状
    if data.shape != (2922, 28, 28):
        raise ValueError(
            f"数据形状不正确: {data.shape}\n"
            f"期望形状: (2922, 28, 28)"
        )

    # 获取温度特征索引
    temp_idx = TEMP_FEATURE_MAP[temp_type]
    print(f"  温度类型: {temp_type} (索引: {temp_idx})")
    print(f"  统计类型: {stat_type}")

    # 提取温度数据
    temp_data = data[:, :, temp_idx]  # [2922, 28]

    # 计算统计量
    stat_func = STAT_FUNCTIONS[stat_type]
    temp_stat = stat_func(temp_data, axis=0)  # [28]
    print(f"  温度{STAT_DISPLAY_NAME[stat_type]}范围: "
          f"{temp_stat.min():.2f} - {temp_stat.max():.2f} C")

    # 加载站点信息
    if not STATION_INFO_PATH.exists():
        raise FileNotFoundError(f"站点信息文件不存在: {STATION_INFO_PATH}")

    print(f"正在加载站点信息: {STATION_INFO_PATH}")
    station_info = np.load(STATION_INFO_PATH)
    print(f"  站点信息形状: {station_info.shape}")

    # 验证站点数量
    if station_info.shape[0] != 28:
        raise ValueError(
            f"站点数量不正确: {station_info.shape[0]}\n"
            f"期望数量: 28"
        )

    # 提取经纬度
    lon = station_info[:, 1]
    lat = station_info[:, 2]
    num_stations = len(lon)

    print(f"  经度范围: {lon.min():.2f} - {lon.max():.2f} E")
    print(f"  纬度范围: {lat.min():.2f} - {lat.max():.2f} N")

    return temp_stat, lon, lat, num_stations


def plot_temperature_spatial_map_with_basemap(temp_stat, lon, lat,
                                               num_stations,
                                               temp_type, stat_type, save_path,
                                               annotation_type='ids'):
    """
    绘制带专业地理底图的温度统计量空间分布图 (使用cartopy)
    完全照搬 visualize_results.py 中的 plot_rmse_spatial_map_with_basemap 函数

    Args:
        temp_stat: 温度统计量数组 [num_stations]
        lon: 经度数组 [num_stations]
        lat: 纬度数组 [num_stations]
        num_stations: 站点数量
        temp_type: 温度类型 ('tmin', 'tmax', 'tave')
        stat_type: 统计类型 ('mean', 'std')
        save_path: 保存路径
        annotation_type: 标注类型 ('ids'=站点下标, 'values'=温度数值)
    """
    if not BASEMAP_AVAILABLE:
        print(f"  [WARNING] 跳过底图版本(依赖未安装): {save_path.name}")
        return

    fontsize = 16
    # 创建cartopy地理坐标轴
    fig = plt.figure(figsize=(10, 10), dpi=DPI)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 散点图 (直接使用经纬度)
    scatter = ax.scatter(
        lon, lat,
        transform=ccrs.PlateCarree(),  # 关键!指定数据投影
        c=temp_stat,
        s=500,
        cmap='RdYlGn_r',
        edgecolors='black',
        linewidth=1.5,
        # alpha=0.85,
        vmin=np.percentile(temp_stat, 5),
        vmax=np.percentile(temp_stat, 95),
        zorder=5  # 确保散点在底图之上
    )

    # 添加标注
    if annotation_type == 'ids':
        # 版本1: 在散点内部标注站点下标
        for i in range(num_stations):
            ax.text(lon[i], lat[i], str(i),
                    transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                    fontsize=fontsize, ha='center', va='center',
                    color='black', weight='bold', zorder=6)
    else:
        # 版本2: 在散点旁边标注温度数值
        for i in range(num_stations):
            temp_val = temp_stat[i]
            ax.text(lon[i], lat[i], f'{temp_val:.2f}',
                    transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                    fontsize=fontsize, ha='center', va='center',
                    color='black', weight='bold', zorder=6)

    # 添加底图
    basemap_loaded = False

    if USE_WMTS_BASEMAP:
        print(f"  尝试加载Mapbox WMTS底图...")
        from myGNN.utils.cartopy_helpers import add_mapbox_wmts

        # 使用标准OGC WMTS接口
        add_mapbox_wmts(ax, layer_name='cmit4xn41001v01s51jp2eq6p', alpha=0.6)
        basemap_loaded = True
        basemap_name = 'Mapbox WMTS'
        print(f"  [OK] Mapbox WMTS底图加载成功")

    else:
        # 直接使用Natural Earth
        print(f"  使用Natural Earth离线底图...")
        from myGNN.utils.cartopy_helpers import add_basemap_features
        add_basemap_features(ax, style='natural_earth', add_gridlines=False)
        basemap_loaded = True
        basemap_name = 'Natural Earth'
        print(f"  [OK] Natural Earth底图加载成功")

    # Colorbar - 放置在地图内部左下角
    # 创建一个内嵌的坐标轴用于colorbar
    cax = inset_axes(ax,
                     width="3%",      # colorbar宽度
                     height="30%",    # colorbar高度
                     loc='lower left',  # 位置：左下角
                     bbox_to_anchor=(0.05, 0.1, 1, 1),  # 相对于ax的位置
                     bbox_transform=ax.transAxes,
                     borderpad=0)

    cbar = plt.colorbar(scatter, cax=cax, orientation='vertical')
    temp_display = TEMP_DISPLAY_NAME[temp_type]
    stat_display = STAT_DISPLAY_NAME[stat_type]
    unit_suffix = STAT_UNIT_SUFFIX[stat_type]
    cbar.ax.set_title(f'{temp_display}(°C)',
                      fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # 给colorbar添加白色半透明背景
    cax.patch.set_facecolor('white')
    cax.patch.set_alpha(0.8)

    # 添加网格线
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                      y_inline=False, alpha=0.5, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # 控制刻度密度（设置经纬度刻度间隔）
    import matplotlib.ticker as mticker
    gl.xlocator = mticker.MaxNLocator(nbins=4)
    gl.ylocator = mticker.MaxNLocator(nbins=2)

    # 设置刻度标签字体大小
    gl.xlabel_style = {'size': fontsize}
    gl.ylabel_style = {'size': fontsize, 'rotation': 90}

    # 添加比例尺（如果配置启用）
    if ADD_SCALEBAR and basemap_loaded:
        try:
            # cartopy PlateCarree投影下,需要根据纬度计算度→米的换算
            avg_lat = np.mean(lat)
            meters_per_degree = 111320 * np.cos(np.radians(avg_lat))

            scalebar = ScaleBar(
                dx=meters_per_degree,  # 每度对应的米数
                units='m',
                dimension='si-length',
                length_fraction=0.25,
                width_fraction=0.01,
                location='lower left',
                box_alpha=0.7,
                box_color='white',
                color='black',
                font_properties={'size': 9, 'weight': 'bold'}
            )
            ax.add_artist(scalebar)
        except Exception as e:
            print(f"  [WARNING] 比例尺添加失败: {e}")

    # 添加指北针（如果配置启用）
    if ADD_NORTH_ARROW and basemap_loaded:
        try:
            arrow_x, arrow_y = 0.05, 0.92

            # 指北针文字
            ax.text(arrow_x, arrow_y + 0.05, 'N',
                    transform=ax.transAxes, fontsize=14, ha='center',
                    va='center',
                    weight='bold',
                    bbox=dict(boxstyle='circle', facecolor='white',
                              edgecolor='black', linewidth=1.5, alpha=0.8))

            # 指北箭头
            ax.annotate('', xy=(arrow_x, arrow_y + 0.04),
                        xytext=(arrow_x, arrow_y),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        except Exception as e:
            print(f"  [WARNING] 指北针添加失败: {e}")

    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] 图片已保存至: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("温度统计量空间分布图绘制脚本")
    print("=" * 60)

    # 加载数据
    temp_stat, lon, lat, num_stations = load_data(TEMP_TYPE, STAT_TYPE)

    # 生成保存路径
    save_path = OUTPUT_DIR / f'temperature_spatial_{STAT_TYPE}_{TEMP_TYPE}.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 绘制图片
    plot_temperature_spatial_map_with_basemap(
        temp_stat, lon, lat, num_stations,
        TEMP_TYPE, STAT_TYPE, save_path, annotation_type='ids'
    )

    print("=" * 60)
    print("绘图完成!")
    print(f"温度类型: {TEMP_TYPE}")
    print(f"统计类型: {STAT_TYPE}")
    print(f"输出文件: {save_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
