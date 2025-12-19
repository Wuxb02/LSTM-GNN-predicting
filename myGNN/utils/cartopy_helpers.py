"""
Cartopy地理绘图辅助工具

提供统一的cartopy地理可视化接口,包括:
- 地理坐标轴创建
- Mapbox WMTS在线底图 (标准OGC WMTS) ⭐
- Natural Earth离线矢量要素 (降级)
- 自动异常处理和降级机制

作者: GNN气温预测项目
日期: 2025
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings

# 禁用SSL证书验证(可选,用于解决某些网络环境的SSL问题)
# 注释: 如果遇到SSL证书错误,取消注释下一行
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


def get_mapbox_wmts_url():
    """
    获取Mapbox WMTS服务URL

    Returns:
        str: 完整的WMTS GetCapabilities URL
    """
    # Mapbox WMTS配置
    username = 'wuxb55'
    style_id = 'cmit4xn41001v01s51jp2eq6p'
    access_token = 'pk.eyJ1Ijoid3V4YjU1IiwiYSI6ImNtaWJsbmRwMDBtbG8yaXM2cTV1NTJ6N2MifQ.cO_6K7oGnqEsoaNky1FLHw'
    wmts_url = f"https://api.mapbox.com/styles/v1/{username}/{style_id}/wmts?access_token={access_token}"

    return wmts_url

def add_mapbox_wmts(ax, wmts_url=None, layer_name=r'cmit4xn41001v01s51jp2eq6p', alpha=1.0):
    """
    添加Mapbox WMTS底图到地理坐标轴 (标准OGC WMTS)

    Args:
        ax: Cartopy GeoAxes对象
        wmts_url (str): WMTS服务URL (可选，有默认值)
        layer_name (str): WMTS图层标识符 (可选，自动检测)
        alpha (float): 透明度 (0-1，默认1.0)

    Returns:
        ax: 添加了WMTS底图的GeoAxes对象

    Raises:
        Exception: WMTS服务不可用或网络错误

    Note:
        缩放级别由Cartopy根据地图范围自动选择
    """
    # 获取WMTS URL
    if wmts_url is None:
        wmts_url = get_mapbox_wmts_url()

    # 使用标准ax.add_wmts()方法
    # ax.add_wmts()接收URL字符串和layer_name，会自动创建WMTSRasterSource
    # 缩放级别由Cartopy根据地图范围自动选择
    ax.add_wmts(wmts_url, layer_name, alpha=alpha)

    return ax




def create_geo_axes(projection='PlateCarree', figsize=(15, 10), dpi=300):
    """
    创建带地理投影的matplotlib坐标轴

    Args:
        projection (str): 投影类型
            - 'PlateCarree': 等距圆柱投影(默认,适合经纬度数据)
            - 'Mercator': 墨卡托投影
            - 'LambertConformal': 兰伯特等角圆锥投影
        figsize (tuple): 图表大小 (width, height)
        dpi (int): 图表分辨率

    Returns:
        fig, ax: matplotlib Figure和GeoAxes对象
    """
    projection_map = {
        'PlateCarree': ccrs.PlateCarree(),
        'Mercator': ccrs.Mercator(),
        'LambertConformal': ccrs.LambertConformal()
    }

    proj = projection_map.get(projection, ccrs.PlateCarree())

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    return fig, ax


def add_basemap_features(ax, style='natural_earth', add_gridlines=True,
                         land_color='lightgray', ocean_color='lightblue'):
    """
    添加底图地理要素(离线矢量数据)

    Args:
        ax: GeoAxes对象
        style (str): 底图样式
            - 'natural_earth': Natural Earth完整要素(默认)
            - 'simple': 简化版(仅海岸线和国界)
            - 'minimal': 最简版(仅海岸线)
        add_gridlines (bool): 是否添加网格线
        land_color (str): 陆地颜色
        ocean_color (str): 海洋颜色

    Returns:
        ax: 添加了要素的GeoAxes对象
    """
    if style == 'natural_earth':
        # Natural Earth完整要素
        ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray', zorder=1)
        ax.add_feature(cfeature.LAKES, facecolor=ocean_color, edgecolor='black',
                       linewidth=0.3, alpha=0.5, zorder=1)
        ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3, alpha=0.5, zorder=1)

    elif style == 'simple':
        # 简化版(陆地、海岸线、国界)
        ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray', zorder=1)

    elif style == 'minimal':
        # 最简版(仅海岸线)
        ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', zorder=1)

    else:
        raise ValueError(f"不支持的底图样式: {style}. 支持: 'natural_earth', 'simple', 'minimal'")

    if add_gridlines:
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                          y_inline=False, alpha=0.5, linestyle='--', linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # 控制刻度密度和字体大小
        import matplotlib.ticker as mticker
        gl.xlocator = mticker.MaxNLocator(nbins=5)
        gl.ylocator = mticker.MaxNLocator(nbins=5)
        gl.xlabel_style = {'size': 11}
        gl.ylabel_style = {'size': 11}

    return ax


def set_geo_extent(ax, lon, lat, margin_percent=10):
    """
    设置地理范围(自动添加边距)

    Args:
        ax: GeoAxes对象
        lon (array): 经度数组
        lat (array): 纬度数组
        margin_percent (float): 边距百分比(默认10%)

    Returns:
        ax: 设置了范围的GeoAxes对象
    """
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    # 计算边距
    margin_lon = (lon_max - lon_min) * (margin_percent / 100.0)
    margin_lat = (lat_max - lat_min) * (margin_percent / 100.0)

    # 设置范围
    ax.set_extent([lon_min - margin_lon, lon_max + margin_lon,
                   lat_min - margin_lat, lat_max + margin_lat],
                  crs=ccrs.PlateCarree())

    return ax


def plot_geo_scatter(ax, lon, lat, values=None, cmap='RdYlGn_r',
                     s=300, edgecolors='black', linewidth=1.5, alpha=0.85,
                     vmin=None, vmax=None, zorder=3, **kwargs):
    """
    在地理坐标系上绘制散点图

    Args:
        ax: GeoAxes对象
        lon (array): 经度数组
        lat (array): 纬度数组
        values (array, optional): 散点颜色值
        cmap (str): 颜色映射
        s (float): 散点大小
        edgecolors (str): 边缘颜色
        linewidth (float): 边缘宽度
        alpha (float): 透明度
        vmin, vmax (float): 颜色范围
        zorder (int): 绘制层级
        **kwargs: 传递给ax.scatter的其他参数

    Returns:
        scatter: matplotlib PathCollection对象
    """
    scatter = ax.scatter(
        lon, lat,
        c=values,
        cmap=cmap,
        s=s,
        edgecolors=edgecolors,
        linewidth=linewidth,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),  # 关键!指定数据投影
        zorder=zorder,
        **kwargs
    )

    return scatter


def plot_geo_lines(ax, edge_index, station_coords, linewidths, alphas,
                   color='red', zorder=2):
    """
    在地理坐标系上绘制线条(用于边可视化)

    Args:
        ax: GeoAxes对象
        edge_index: [2, num_edges] 边索引数组
        station_coords: [num_stations, 2] 气象站坐标 [lon, lat]
        linewidths (array): 线条宽度数组
        alphas (array): 透明度数组
        color (str): 线条颜色
        zorder (int): 绘制层级

    Returns:
        ax: GeoAxes对象
    """
    for idx in range(edge_index.shape[1]):
        src, dst = edge_index[:, idx]
        ax.plot(
            [station_coords[src, 0], station_coords[dst, 0]],
            [station_coords[src, 1], station_coords[dst, 1]],
            color=color,
            linewidth=linewidths[idx] if hasattr(linewidths, '__len__') else linewidths,
            alpha=alphas[idx] if hasattr(alphas, '__len__') else alphas,
            transform=ccrs.PlateCarree(),  # 关键!指定数据投影
            zorder=zorder
        )

    return ax


def add_station_labels(ax, lon, lat, labels, fontsize=11, ha='center',
                       va='center', color='white', weight='bold', zorder=4):
    """
    在地理坐标系上添加站点标签

    Args:
        ax: GeoAxes对象
        lon (array): 经度数组
        lat (array): 纬度数组
        labels (list): 标签列表
        fontsize (int): 字体大小
        ha, va (str): 水平和垂直对齐方式
        color (str): 字体颜色
        weight (str): 字体粗细
        zorder (int): 绘制层级

    Returns:
        ax: GeoAxes对象
    """
    for i in range(len(lon)):
        ax.text(lon[i], lat[i], str(labels[i]),
                transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                fontsize=fontsize, ha=ha, va=va,
                color=color, weight=weight, zorder=zorder)

    return ax


def add_north_arrow(ax, x=0.05, y=0.92, arrow_length=0.04, fontsize=14):
    """
    添加指北针

    Args:
        ax: GeoAxes对象
        x, y (float): 指北针位置(相对坐标,0-1)
        arrow_length (float): 箭头长度(相对坐标)
        fontsize (int): 字体大小

    Returns:
        ax: GeoAxes对象
    """
    # 指北针文字
    ax.text(x, y + 0.05, 'N',
            transform=ax.transAxes, fontsize=fontsize, ha='center', va='center',
            weight='bold',
            bbox=dict(boxstyle='circle', facecolor='white',
                      edgecolor='black', linewidth=1.5, alpha=0.8))

    # 指北箭头
    ax.annotate('', xy=(x, y + 0.04), xytext=(x, y),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    return ax


def add_basemap_credit(ax, basemap_name, x=0.99, y=0.01):
    """
    添加底图来源说明

    Args:
        ax: GeoAxes对象
        basemap_name (str): 底图名称
        x, y (float): 文本位置(相对坐标,0-1)

    Returns:
        ax: GeoAxes对象
    """
    credit_text = f"Basemap source: {basemap_name}"
    if 'OpenStreetMap' in basemap_name or 'OSM' in basemap_name:
        credit_text += " contributors"

    ax.text(x, y, credit_text,
            transform=ax.transAxes, fontsize=7,
            ha='right', va='bottom', style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))

    return ax


def create_geo_subplot(fig, nrows, ncols, index, projection='PlateCarree'):
    """
    在现有figure中创建地理子图

    Args:
        fig: matplotlib Figure对象
        nrows, ncols (int): 子图网格行列数
        index (int): 子图索引(从1开始)
        projection (str): 投影类型

    Returns:
        ax: GeoAxes对象
    """
    projection_map = {
        'PlateCarree': ccrs.PlateCarree(),
        'Mercator': ccrs.Mercator(),
        'LambertConformal': ccrs.LambertConformal()
    }

    proj = projection_map.get(projection, ccrs.PlateCarree())
    ax = fig.add_subplot(nrows, ncols, index, projection=proj)

    return ax
