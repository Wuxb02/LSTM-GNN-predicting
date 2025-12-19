"""
可解释性分析可视化工具

生成时序和空间可解释性的各种可视化图表

作者: GNN气温预测项目
日期: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 设置matplotlib使用英文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_temporal_heatmap(temporal_heatmap, feature_names=None, save_path=None, hist_len=None):
    """
    绘制时序-特征重要性热图

    Args:
        temporal_heatmap: [hist_len, in_dim] 时空热图
        feature_names: List[str] 特征名称列表(可选)
        save_path: 保存路径
        hist_len: 历史时间窗口长度

    生成图表:
        X轴: 历史时间步 (T-14, T-13, ..., T-1)
        Y轴: 气象特征
        颜色: 重要性得分 (深色=重要)
    """
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    # 转置为 [in_dim, hist_len] 用于热图
    heatmap_data = temporal_heatmap.T if hasattr(temporal_heatmap, 'T') else temporal_heatmap.transpose()

    # 时间步标签
    if hist_len is None:
        hist_len = temporal_heatmap.shape[0]
    xticklabels = [f'T-{i}' for i in range(hist_len, 0, -1)]

    # Feature labels
    if feature_names is None:
        feature_names = [f'Feature{i}' for i in range(temporal_heatmap.shape[1])]

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        xticklabels=xticklabels,
        yticklabels=feature_names,
        cbar_kws={'label': 'Importance Score'},
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        fmt='.3f',
    )

    ax.set_xlabel('Historical Time Steps', fontsize=13, weight='bold')
    ax.set_ylabel('Meteorological Features', fontsize=13, weight='bold')
    ax.set_title('LSTM Temporal Feature Importance Analysis', fontsize=15, weight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Temporal heatmap saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_edges_on_axis(
    ax,
    edge_importance,
    edge_index,
    station_coords,
    top_k=20,
    color='red',
    title='',
    use_basemap=True,
    show_all_edges=False,
    all_edges_alpha=0.1
):
    """
    在给定的axis上绘制边（通用函数）

    Args:
        ax: matplotlib axis对象
        edge_importance: [num_edges] 边重要性/注意力权重
        edge_index: [2, num_edges]
        station_coords: [num_stations, 2] 经纬度坐标
        top_k: 显示Top-K重要边
        color: 边的颜色（'red', 'blue', 'green'等）
        title: 子图标题
        use_basemap: 是否使用地理底图
        show_all_edges: 是否显示所有边（灰色底层）
        all_edges_alpha: 所有边的透明度
    """
    # 转换为numpy
    if hasattr(edge_importance, 'numpy'):
        edge_importance = edge_importance.numpy()
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()

    # 坐标处理: cartopy直接使用经纬度,无需转换
    lon_coords = station_coords[:, 0]
    lat_coords = station_coords[:, 1]

    # 1. 绘制所有边（如果需要）
    if show_all_edges:
        for idx in range(len(edge_importance)):
            src, dst = edge_index[:, idx]
            ax.plot(
                [lon_coords[src], lon_coords[dst]],
                [lat_coords[src], lat_coords[dst]],
                transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                color='lightgray',
                linewidth=0.5,
                alpha=all_edges_alpha,
                zorder=1
            )

    # 2. 选择Top-K重要边
    top_k = min(top_k, len(edge_importance))
    top_indices = np.argsort(edge_importance)[-top_k:]

    # 3. 归一化重要性（基于Top-K边）
    top_importances = edge_importance[top_indices]
    norm_min = top_importances.min()
    norm_max = top_importances.max()

    # 4. 绘制Top-K重要边
    for idx in top_indices:
        src, dst = edge_index[:, idx]
        importance = edge_importance[idx]

        # 归一化到[0, 1]
        norm_importance = (importance - norm_min) / (norm_max - norm_min + 1e-10)

        # 线条宽度和透明度映射重要性
        linewidth = 0.5 + 4.0 * norm_importance  # 0.5-4.5
        alpha = 0.3 + 0.7 * norm_importance      # 0.3-1.0

        ax.plot(
            [lon_coords[src], lon_coords[dst]],
            [lat_coords[src], lat_coords[dst]],
            transform=ccrs.PlateCarree(),  # 关键!指定数据投影
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2
        )

    # 5. 绘制气象站
    ax.scatter(
        lon_coords, lat_coords,
        transform=ccrs.PlateCarree(),  # 关键!指定数据投影
        c='blue',
        s=300,
        edgecolors='black',
        linewidth=2,
        alpha=0.85,
        zorder=3
    )

    # 6. 添加站点标签
    for i in range(len(station_coords)):
        ax.text(lon_coords[i], lat_coords[i], str(i),
                transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                fontsize=11, ha='center', va='center',
                color='white', weight='bold', zorder=4)

    # 7. 添加地理底图 (使用Mapbox WMTS)
    if use_basemap:
        from myGNN.utils.cartopy_helpers import add_mapbox_wmts
        add_mapbox_wmts(ax, layer_name='cmit4xn41001v01s51jp2eq6p', alpha=0.5)
        print("  ✓ Mapbox WMTS basemap loaded successfully")


    # 8. 设置标题和网格线
    ax.set_title(title, fontsize=15, weight='bold')
    if use_basemap:
        # 添加网格线和标签
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
    else:
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.grid(True, alpha=0.3)


def plot_spatial_edges(edge_importance, edge_index, station_coords, save_path=None, top_k=20, use_basemap=True):
    """
    在地理底图上绘制重要边 (使用cartopy)

    Args:
        edge_importance: torch.Tensor [num_edges] 边重要性
        edge_index: torch.Tensor [2, num_edges]
        station_coords: np.ndarray [num_stations, 2] 经纬度坐标 [lon, lat]
        save_path: 保存路径
        top_k: 显示Top-K重要边
        use_basemap: 是否使用在线底图 (True=Mapbox WMTS, False=Natural Earth)

    可视化元素:
        - 气象站: 蓝色散点
        - 重要边: 红色线条(粗细=重要性)
        - 地理底图: Mapbox WMTS (use_basemap=True) 或 Natural Earth (use_basemap=False)
    """
    # 创建cartopy地理坐标轴
    fig = plt.figure(figsize=(15, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 转换为numpy
    if hasattr(edge_importance, 'numpy'):
        edge_importance = edge_importance.numpy()
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()

    # 提取经纬度
    lon = station_coords[:, 0]
    lat = station_coords[:, 1]

    # 选择top-k重要边
    top_k = min(top_k, len(edge_importance))
    top_indices = np.argsort(edge_importance)[-top_k:]

    # 绘制重要边
    for idx in top_indices:
        src, dst = edge_index[:, idx]
        importance = edge_importance[idx]

        # 归一化重要性到[0, 1]
        norm_importance = (importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min() + 1e-10)

        # 线条宽度和透明度映射重要性
        linewidth = 0.5 + 4.0 * norm_importance  # 0.5-4.5
        alpha = 0.3 + 0.7 * norm_importance      # 0.3-1.0

        ax.plot(
            [lon[src], lon[dst]],
            [lat[src], lat[dst]],
            transform=ccrs.PlateCarree(),  # 关键!指定数据投影
            color='red',
            linewidth=linewidth,
            alpha=alpha,
            zorder=2
        )

    # 绘制气象站
    scatter = ax.scatter(
        lon, lat,
        transform=ccrs.PlateCarree(),  # 关键!指定数据投影
        c='blue',
        s=300,
        edgecolors='black',
        linewidth=2,
        alpha=0.85,
        zorder=3
    )

    # 添加站点标签
    for i in range(len(station_coords)):
        ax.text(lon[i], lat[i], str(i),
                transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                fontsize=11, ha='center', va='center',
                color='white', weight='bold', zorder=4)

    # 添加地理底图
    if use_basemap:
        try:
            from myGNN.utils.cartopy_helpers import add_mapbox_wmts
            add_mapbox_wmts(ax, layer_name='cmit4xn41001v01s51jp2eq6p', alpha=0.5)
            print("  ✓ Mapbox WMTS basemap loaded successfully")
        except Exception as e:
            print(f"  ⚠ Mapbox WMTS加载失败: {e}")
            print("  ℹ 降级使用Natural Earth离线底图...")
            from myGNN.utils.cartopy_helpers import add_basemap_features
            add_basemap_features(ax, style='simple', add_gridlines=False)
    else:
        from myGNN.utils.cartopy_helpers import add_basemap_features
        add_basemap_features(ax, style='simple', add_gridlines=False)
        print("  ✓ Natural Earth basemap loaded")

    # 添加网格线
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

    ax.set_title(f'GAT Important Edges Visualization (Top-{top_k})', fontsize=15, weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Spatial edges plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_spatial_edges_no_basemap(edge_importance, edge_index, station_coords, save_path=None, top_k=20):
    """
    绘制重要边(无地理底图版本,使用cartopy GeoAxes)

    Args:
        edge_importance: [num_edges] 边重要性
        edge_index: [2, num_edges]
        station_coords: [num_stations, 2] 经纬度坐标 [lon, lat]
        save_path: 保存路径
        top_k: 显示Top-K重要边
    """
    # 创建cartopy地理坐标轴(不添加底图)
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 转换为numpy
    if hasattr(edge_importance, 'numpy'):
        edge_importance = edge_importance.numpy()
    if hasattr(edge_index, 'numpy'):
        edge_index = edge_index.numpy()

    # 提取经纬度
    lon = station_coords[:, 0]
    lat = station_coords[:, 1]

    # 选择top-k重要边
    top_k = min(top_k, len(edge_importance))
    top_indices = np.argsort(edge_importance)[-top_k:]

    # 绘制重要边
    for idx in top_indices:
        src, dst = edge_index[:, idx]
        importance = edge_importance[idx]

        # 归一化重要性
        norm_importance = (importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min() + 1e-10)

        linewidth = 0.5 + 3.0 * norm_importance
        alpha = 0.3 + 0.7 * norm_importance

        ax.plot(
            [lon[src], lon[dst]],
            [lat[src], lat[dst]],
            transform=ccrs.PlateCarree(),  # 关键!指定数据投影
            color='red',
            linewidth=linewidth,
            alpha=alpha,
            zorder=2
        )

    # 绘制气象站
    ax.scatter(
        lon, lat,
        transform=ccrs.PlateCarree(),  # 关键!指定数据投影
        c='blue',
        s=300,
        edgecolors='black',
        linewidth=2,
        alpha=0.85,
        zorder=3
    )

    # 添加站点标签
    for i in range(len(station_coords)):
        ax.text(lon[i], lat[i], str(i),
                transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                fontsize=11, ha='center', va='center',
                color='white', weight='bold', zorder=4)

    # 添加网格线
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False,
                      y_inline=False, alpha=0.3, linestyle='--', linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # 控制刻度密度和字体大小
    import matplotlib.ticker as mticker
    gl.xlocator = mticker.MaxNLocator(nbins=5)
    gl.ylocator = mticker.MaxNLocator(nbins=5)
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}

    ax.set_title(f'Important Edges Visualization (Top-{top_k})', fontsize=15, weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Spatial edges plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_edge_distribution(edge_importance, save_path=None):
    """
    绘制边重要性分布图

    Args:
        edge_importance: [num_edges] 边重要性
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 转换为numpy
    if hasattr(edge_importance, 'numpy'):
        edge_importance = edge_importance.numpy()

    # Histogram
    ax1.hist(edge_importance, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Importance Score', fontsize=12)
    ax1.set_ylabel('Number of Edges', fontsize=12)
    ax1.set_title('Edge Importance Distribution', fontsize=13, weight='bold')
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_importance = np.sort(edge_importance)[::-1]
    cumsum = np.cumsum(sorted_importance)
    cumsum_norm = cumsum / cumsum[-1] * 100

    ax2.plot(range(len(cumsum_norm)), cumsum_norm, color='darkred', linewidth=2)
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='80% Contribution Line')
    ax2.set_xlabel('Edge Rank', fontsize=12)
    ax2.set_ylabel('Cumulative Contribution (%)', fontsize=12)
    ax2.set_title('Cumulative Edge Importance Distribution', fontsize=13, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Edge distribution plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_spatial_edges_with_all(
    edge_importance,
    edge_index,
    station_coords,
    save_path=None,
    top_k=20,
    show_mode='overlay',
    use_basemap=True
):
    """
    绘制所有边+Top-K重要边可视化 (使用cartopy)

    Args:
        edge_importance: [num_edges] 边重要性
        edge_index: [2, num_edges]
        station_coords: [num_stations, 2] 经纬度坐标 [lon, lat]
        save_path: 保存路径
        top_k: 显示Top-K重要边
        show_mode: 显示模式
        use_basemap: 是否使用在线底图 (True=Mapbox WMTS, False=Natural Earth)
            - 'overlay': 叠加模式（所有边+Top-K边在同一图）
            - 'separate': 分离模式（两个子图分别显示）

    生成图表:
        - overlay模式: 底层灰色所有边 + 上层红色Top-K重要边
        - separate模式: 左图所有边 + 右图Top-K重要边
    """
    if show_mode == 'overlay':
        # 叠加模式：单图
        fig = plt.figure(figsize=(15, 10), dpi=300)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        plot_edges_on_axis(
            ax, edge_importance, edge_index, station_coords,
            top_k=top_k, color='red',
            title=f'All Edges + Top-{top_k} Important Edges (Overlay)',
            use_basemap=use_basemap,
            show_all_edges=True,
            all_edges_alpha=0.1
        )

    elif show_mode == 'separate':
        # Separate mode: two subplots with GeoAxes
        fig = plt.figure(figsize=(28, 10), dpi=300)
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

        # Left: all edges
        plot_edges_on_axis(
            ax1, edge_importance, edge_index, station_coords,
            top_k=len(edge_importance),  # Show all edges
            color='gray',
            title='All Edge Distribution',
            use_basemap=use_basemap,
            show_all_edges=False,  # No background layer needed
            all_edges_alpha=0.3
        )

        # Right: Top-K important edges
        plot_edges_on_axis(
            ax2, edge_importance, edge_index, station_coords,
            top_k=top_k,
            color='red',
            title=f'Top-{top_k} Important Edges',
            use_basemap=use_basemap,
            show_all_edges=False
        )

    else:
        raise ValueError(f"Invalid show_mode: {show_mode}. Supported: 'overlay', 'separate'")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ All edges visualization saved ({show_mode} mode): {save_path}")
    else:
        plt.show()

    plt.close()


def plot_gat_attention_comparison(
    explainer_importance,
    attention_weights,
    edge_index,
    station_coords,
    save_path=None,
    top_k=20,
    use_basemap=True
):
    """
    绘制GNNExplainer vs GAT注意力权重对比图 (使用cartopy)

    Args:
        explainer_importance: [num_edges] GNNExplainer边重要性
        attention_weights: [num_edges] GAT聚合注意力权重
        edge_index: [2, num_edges]
        station_coords: [num_stations, 2] 经纬度坐标 [lon, lat]
        save_path: 保存路径
        top_k: 显示Top-K边
        use_basemap: 是否使用在线底图 (True=Mapbox WMTS, False=Natural Earth)

    生成图表:
        左图: GNNExplainer边重要性 (红色)
        右图: GAT聚合注意力权重 (蓝色)
    """
    # 创建两个GeoAxes子图
    fig = plt.figure(figsize=(28, 10), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

    # Left: GNNExplainer edge importance
    plot_edges_on_axis(
        ax1, explainer_importance, edge_index, station_coords,
        top_k=top_k, color='red',
        title=f'GNNExplainer Edge Importance (Top-{top_k})',
        use_basemap=use_basemap
    )

    # Right: GAT attention weights
    plot_edges_on_axis(
        ax2, attention_weights, edge_index, station_coords,
        top_k=top_k, color='blue',
        title=f'GAT Aggregated Attention Weights (Top-{top_k})',
        use_basemap=use_basemap
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Comparison plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_time_importance(time_importance, save_path=None):
    """
    绘制时间步重要性柱状图

    Args:
        time_importance: [hist_len] 时间步重要性
            - 索引0对应最早的时间步(T-hist_len)
            - 索引hist_len-1对应最近的时间步(T-1)
        save_path: 保存路径

    注意: 数据索引与时间步的对应关系
        - time_importance[0] → T-14 (最早)
        - time_importance[13] → T-1 (最近)
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # 转换为numpy
    if hasattr(time_importance, 'numpy'):
        time_importance = time_importance.numpy().copy()

    hist_len = len(time_importance)

    # X-axis labels: T-14, T-13, ..., T-1
    # Corresponding data: time_importance[0], time_importance[1], ..., time_importance[13]
    # Data doesn't need to be flipped, keep consistent with heatmap
    x_labels = [f'T-{i}' for i in range(hist_len, 0, -1)]

    bars = ax.bar(range(hist_len), time_importance, color='teal', edgecolor='black', alpha=0.7)

    # Highlight most important time step
    max_idx = np.argmax(time_importance)
    bars[max_idx].set_color('orangered')

    ax.set_xlabel('Historical Time Steps', fontsize=12, weight='bold')
    ax.set_ylabel('Importance Score (Normalized)', fontsize=12, weight='bold')
    ax.set_title('Time Step Importance Distribution', fontsize=14, weight='bold')
    ax.set_xticks(range(hist_len))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Time importance plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(feature_importance, feature_names=None, save_path=None, top_k=15):
    """
    绘制特征重要性排名图

    Args:
        feature_importance: [in_dim] 特征重要性
        feature_names: List[str] 特征名称(可选)
        save_path: 保存路径
        top_k: 显示Top-K重要特征
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 转换为numpy
    if hasattr(feature_importance, 'numpy'):
        feature_importance = feature_importance.numpy()

    # Sort
    top_k = min(top_k, len(feature_importance))
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]

    top_values = feature_importance[top_indices]

    if feature_names is None:
        feature_names = [f'Feature{i}' for i in range(len(feature_importance))]

    top_names = [feature_names[i] for i in top_indices]

    # Horizontal bar chart
    bars = ax.barh(range(top_k), top_values, color='mediumseagreen', edgecolor='black', alpha=0.8)

    # Highlight most important feature
    bars[0].set_color('crimson')

    ax.set_xlabel('Importance Score (Normalized)', fontsize=12, weight='bold')
    ax.set_ylabel('Meteorological Features', fontsize=12, weight='bold')
    ax.set_title(f'Feature Importance Ranking (Top-{top_k})', fontsize=14, weight='bold')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Feature importance plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


# ============ GAT注意力权重深度分析图表 ============

def plot_attention_matrix_heatmap(attention_matrix, save_path=None):
    """
    绘制全局注意力矩阵热力图

    可视化所有气象站之间的注意力强度分布(N×N矩阵)。

    Args:
        attention_matrix (np.ndarray): [num_nodes, num_nodes] 注意力矩阵
        save_path (str, optional): 保存路径

    Output:
        - 热力图: 28×28注意力矩阵
        - 配色: YlOrRd (黄→橙→红)
        - 尺寸: 12×10英寸, DPI=300
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    # 绘制热力图
    sns.heatmap(
        attention_matrix,
        cmap='YlOrRd',
        xticklabels=np.arange(attention_matrix.shape[1]),
        yticklabels=np.arange(attention_matrix.shape[0]),
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        linecolor='white',
        ax=ax,
        vmin=0,
        square=True
    )

    ax.set_xlabel('Target Station ID', fontsize=13, weight='bold')
    ax.set_ylabel('Source Station ID', fontsize=13, weight='bold')
    ax.set_title('Global GAT Attention Matrix Heatmap',
                 fontsize=15, weight='bold', pad=15)
    ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Attention matrix heatmap saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_distance_vs_attention(edge_distances, attention_weights,
                               save_path=None):
    """
    绘制距离-注意力散点图

    分析物理距离与注意力权重的关系,验证空间局部性假设。

    Args:
        edge_distances (np.ndarray): [num_edges] 边的物理距离 (公里)
        attention_weights (np.ndarray): [num_edges] 边的注意力权重
        save_path (str, optional): 保存路径

    Output:
        - 散点图: 距离(X) vs 注意力(Y)
        - 趋势线: 线性回归拟合
        - 统计标注: Pearson r, p-value, R²
        - 尺寸: 10×8英寸, DPI=300

    Hypothesis:
        距离近的站点应获得更高注意力 (负相关)
    """
    from scipy.stats import linregress

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 过滤自环边 (距离≈0)
    mask = edge_distances > 1e-6
    edge_distances_filtered = edge_distances[mask]
    attention_weights_filtered = attention_weights[mask]

    # 散点图
    ax.scatter(edge_distances_filtered, attention_weights_filtered,
               alpha=0.5, s=30, c='steelblue',
               edgecolors='black', linewidth=0.5)

    # 线性回归趋势线
    slope, intercept, r_value, p_value, std_err = linregress(
        edge_distances_filtered, attention_weights_filtered
    )

    x_trend = np.linspace(edge_distances_filtered.min(),
                          edge_distances_filtered.max(), 100)
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, 'r--', linewidth=2,
            label=f'Trend Line (R²={r_value**2:.3f})')

    # 统计标注
    textstr = f'Pearson r = {r_value:.3f}\np-value = {p_value:.2e}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Physical Distance (km)', fontsize=12, weight='bold')
    ax.set_ylabel('GAT Attention Weight', fontsize=12, weight='bold')
    ax.set_title('Distance vs. Attention Analysis',
                 fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Distance-Attention plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_correlation_vs_attention(edge_correlations, attention_weights,
                                  save_path=None):
    """
    绘制温度相关性-注意力散点图

    分析温度相关性与注意力权重的关系,验证特征相似性假设。

    Args:
        edge_correlations (np.ndarray): [num_edges] 温度相关系数
        attention_weights (np.ndarray): [num_edges] 注意力权重
        save_path (str, optional): 保存路径

    Output:
        - 散点图: 相关系数(X) vs 注意力(Y)
        - 趋势线: 线性回归拟合
        - 统计标注: Pearson r, p-value, R²
        - 零相关参考线
        - 尺寸: 10×8英寸, DPI=300

    Hypothesis:
        温度模式相似的站点应获得更高注意力 (正相关)
    """
    from scipy.stats import linregress

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # 过滤自环边 (相关系数≈1)
    mask = np.abs(edge_correlations - 1.0) > 1e-6
    edge_correlations_filtered = edge_correlations[mask]
    attention_weights_filtered = attention_weights[mask]

    # 散点图
    ax.scatter(edge_correlations_filtered, attention_weights_filtered,
               alpha=0.5, s=30, c='mediumseagreen',
               edgecolors='black', linewidth=0.5)

    # 线性回归趋势线
    slope, intercept, r_value, p_value, std_err = linregress(
        edge_correlations_filtered, attention_weights_filtered
    )

    x_trend = np.linspace(edge_correlations_filtered.min(),
                          edge_correlations_filtered.max(), 100)
    y_trend = slope * x_trend + intercept
    ax.plot(x_trend, y_trend, 'r--', linewidth=2,
            label=f'Trend Line (R²={r_value**2:.3f})')

    # 统计标注
    textstr = f'Pearson r = {r_value:.3f}\np-value = {p_value:.2e}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.set_xlabel('Temperature Correlation Coefficient',
                  fontsize=12, weight='bold')
    ax.set_ylabel('GAT Attention Weight', fontsize=12, weight='bold')
    ax.set_title('Temperature Correlation vs. Attention Analysis',
                 fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)  # 零相关参考线

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Correlation-Attention plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_tsne_clusters(tsne_2d, altitudes, save_path=None,
                       station_ids=None, figsize=(12, 10)):
    """
    绘制节点嵌入的t-SNE聚类图 (颜色映射海拔)

    此函数将高维节点嵌入的t-SNE 2D投影可视化，通过海拔高度着色来验证
    模型是否学习到了有意义的地理特征聚类模式。

    Args:
        tsne_2d: [num_nodes, 2] t-SNE 2D投影坐标
            - 列0: t-SNE维度1
            - 列1: t-SNE维度2
        altitudes: [num_nodes] 海拔高度数组 (单位: m)
        save_path: str or Path, 保存路径 (默认: None, 显示但不保存)
        station_ids: [num_nodes] 气象站编号 (可选，用于标注)
        figsize: tuple, 图表尺寸 (默认: (12, 10))

    生成图表:
        - 散点图: x轴=t-SNE维度1, y轴=t-SNE维度2
        - 颜色: 映射海拔高度 (colorbar显示高度范围)
        - 标注: 每个点显示站点ID (如果提供station_ids)
        - 网格: 浅灰色网格 + 中心十字线

    可视化设计:
        - colormap: 'terrain' (地形配色: 蓝绿→黄褐→白灰)
        - marker size: 150
        - 边缘颜色: 黑色 (linewidth=1.5, 增强可见性)
        - 透明度: 0.8

    科学意义:
        如果海拔相近的站点在t-SNE图中聚类在一起,说明模型的节点嵌入
        成功捕获了地形对气温预测的隐式影响（如山地效应、热岛效应等）。

    示例:
        >>> tsne_2d = np.array([[1.2, 0.5], [-0.8, 1.1], ...])  # [28, 2]
        >>> altitudes = np.array([50, 120, 35, ...])            # [28]
        >>> station_ids = np.array([59264, 59265, ...])         # [28]
        >>> plot_tsne_clusters(tsne_2d, altitudes, station_ids=station_ids,
        ...                    save_path='tsne_clusters.png')
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # 创建散点图
    scatter = ax.scatter(
        tsne_2d[:, 0],
        tsne_2d[:, 1],
        c=altitudes,
        cmap='terrain',     # 地形配色: 低海拔→蓝绿, 高海拔→白灰
        s=150,              # 标记大小
        edgecolors='black', # 黑色边缘增强可见性
        linewidths=1.5,
        alpha=0.8
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Altitude (m)', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # 标注站点ID
    if station_ids is not None:
        for i, (x, y) in enumerate(tsne_2d):
            ax.text(
                x, y, str(int(station_ids[i])),
                fontsize=8,
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.7
                )
            )

    # 设置标题和标签
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13, weight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13, weight='bold')
    ax.set_title('Node Embedding Clustering Analysis (Colored by Altitude)',
                 fontsize=15, weight='bold', pad=15)

    # 网格和中心线
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # 添加说明文本
    textstr = (
        'High-altitude stations should cluster together if the model\n'
        'successfully learned terrain-related implicit features.'
    )
    ax.text(
        0.02, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ t-SNE clustering plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def generate_all_visualizations(
    explanation_data_path,
    save_dir,
    station_coords=None,
    feature_names=None,
    all_edges_mode='both',
    use_basemap=True,
    weather_data_path=None,
    train_indices=(0, 2191),
    generate_attention_analysis=True
):
    """
    生成所有可解释性可视化图表

    Args:
        explanation_data_path: explanation_data.npz文件路径
        save_dir: 可视化保存目录
        station_coords: [num_stations, 2] 气象站坐标(可选)
        feature_names: 特征名称列表(可选)
        all_edges_mode: 全边可视化模式('both', 'overlay', 'separate')
        use_basemap: 是否使用地理底图
        weather_data_path: 气象数据路径(用于相关性分析,可选)
        train_indices: 训练集索引范围(start, end)
        generate_attention_analysis: 是否生成注意力深度分析图表
        feature_names: List[str] 特征名称列表(可选)
        all_edges_mode: 全边可视化模式 ('overlay', 'separate', 'both', None)
            - 'overlay': 只生成叠加模式图
            - 'separate': 只生成分离模式图
            - 'both': 生成两种模式图
            - None: 不生成全边可视化图
        use_basemap: 是否使用在线底图 (True=Mapbox WMTS, False=Natural Earth)

    生成图表:
        1. temporal_heatmap.png - 时序特征热图
        2. spatial_edges.png - 空间边地理图 (Top-K)
        3. spatial_all_edges_overlay.png - 全边叠加图 (新增)
        4. spatial_all_edges_separate.png - 全边分离图 (新增)
        5. comparison_explainer_vs_attention.png - 对比图 (新增,需attention数据)
        6. edge_distribution.png - 边重要性分布
        7. time_importance.png - 时间步柱状图
        8. feature_importance.png - 特征排名图
    """
    import torch

    # 加载数据
    data = np.load(explanation_data_path)

    # 转换为torch tensor
    time_importance = torch.from_numpy(data['time_importance'])
    feature_importance = torch.from_numpy(data['feature_importance'])
    temporal_heatmap = torch.from_numpy(data['temporal_heatmap'])
    edge_importance_mean = torch.from_numpy(data['edge_importance_mean'])

    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nGenerating explainability visualizations...")
    print(f"Save directory: {save_dir}")

    # 1. Temporal heatmap
    plot_temporal_heatmap(
        temporal_heatmap,
        feature_names=feature_names,
        save_path=save_dir / 'temporal_heatmap.png',
        hist_len=len(time_importance)
    )

    # 2. Spatial edges plot (requires station_coords and edge_index)
    if station_coords is not None and 'edge_index' in data:
        edge_index = torch.from_numpy(data['edge_index'])
        plot_spatial_edges(
            edge_importance_mean,
            edge_index,
            station_coords,
            save_path=save_dir / 'spatial_edges.png',
            top_k=20,
            use_basemap=use_basemap
        )
    elif station_coords is None:
        print("  ⚠ Skipping spatial edges plot (requires station_coords)")
    else:
        print("  ⚠ Skipping spatial edges plot (edge_index missing in npz)")

    # 3. Edge distribution plot
    plot_edge_distribution(
        edge_importance_mean,
        save_path=save_dir / 'edge_distribution.png'
    )

    # 4. Time importance
    plot_time_importance(
        time_importance,
        save_path=save_dir / 'time_importance.png'
    )

    # 5. Feature importance
    plot_feature_importance(
        feature_importance,
        feature_names=feature_names,
        save_path=save_dir / 'feature_importance.png'
    )

    # 6. All edges visualization (new)
    if station_coords is not None and 'edge_index' in data and all_edges_mode:
        edge_index = torch.from_numpy(data['edge_index'])

        if all_edges_mode in ['overlay', 'both']:
            plot_spatial_edges_with_all(
                edge_importance_mean, edge_index, station_coords,
                save_path=save_dir / 'spatial_all_edges_overlay.png',
                top_k=20, show_mode='overlay', use_basemap=use_basemap
            )

        if all_edges_mode in ['separate', 'both']:
            plot_spatial_edges_with_all(
                edge_importance_mean, edge_index, station_coords,
                save_path=save_dir / 'spatial_all_edges_separate.png',
                top_k=20, show_mode='separate', use_basemap=use_basemap
            )
    elif all_edges_mode:
        print("  ⚠ Skipping all edges visualization (requires station_coords and edge_index)")

    # 7. GNNExplainer vs GAT attention comparison plot (new)
    if 'attention_mean' in data and station_coords is not None and 'edge_index' in data:
        edge_index = torch.from_numpy(data['edge_index'])
        attention_mean = torch.from_numpy(data['attention_mean'])

        plot_gat_attention_comparison(
            edge_importance_mean, attention_mean,
            edge_index, station_coords,
            save_path=save_dir / 'comparison_explainer_vs_attention.png',
            top_k=20, use_basemap=use_basemap
        )
    elif 'attention_mean' not in data:
        print("  ⚠ Skipping attention comparison plot (attention_mean missing in npz)")
    else:
        print("  ⚠ Skipping attention comparison plot (requires station_coords and edge_index)")

    # ========== GAT注意力权重深度分析 ========== #
    if generate_attention_analysis and 'attention_mean' in data:
        print(f"\n{'='*60}")
        print("Generating GAT Attention Analysis Plots...")
        print(f"{'='*60}")

        if station_coords is None or 'edge_index' not in data:
            print("  ⚠ Skipping attention analysis (requires station_coords and edge_index)")
        else:
            edge_index = torch.from_numpy(data['edge_index'])
            attention_mean = torch.from_numpy(data['attention_mean'])
            num_nodes = len(station_coords)

            # 图表A: 全局注意力矩阵热力图
            print("\n[A] Generating attention matrix heatmap...")
            try:
                from .utils import edge_attention_to_matrix
                attention_matrix = edge_attention_to_matrix(
                    edge_index.numpy(), attention_mean.numpy(),
                    num_nodes, aggregation='mean'
                )
                plot_attention_matrix_heatmap(
                    attention_matrix,
                    save_path=save_dir / 'attention_matrix_heatmap.png'
                )
            except Exception as e:
                print(f"  ⚠ Failed to generate attention matrix heatmap: {e}")

            # 图表B: 距离-注意力散点图
            print("\n[B] Generating distance-attention scatter plot...")
            try:
                from .utils import compute_edge_distances
                edge_distances = compute_edge_distances(
                    edge_index.numpy(), station_coords
                )
                plot_distance_vs_attention(
                    edge_distances, attention_mean.numpy(),
                    save_path=save_dir / 'distance_vs_attention.png'
                )
            except Exception as e:
                print(f"  ⚠ Failed to generate distance-attention plot: {e}")

            # 图表C: 相关性-注意力散点图
            if weather_data_path:
                print("\n[C] Generating correlation-attention scatter plot...")
                try:
                    from .utils import (
                        compute_temperature_correlation,
                        extract_edge_correlations
                    )
                    weather_data = np.load(weather_data_path)
                    corr_matrix = compute_temperature_correlation(
                        weather_data, train_indices, target_feature_idx=4
                    )
                    edge_corrs = extract_edge_correlations(
                        edge_index.numpy(), corr_matrix
                    )
                    plot_correlation_vs_attention(
                        edge_corrs, attention_mean.numpy(),
                        save_path=save_dir / 'correlation_vs_attention.png'
                    )
                except Exception as e:
                    print(f"  ⚠ Failed to generate correlation-attention plot: {e}")
            else:
                print("\n[C] Skipping correlation-attention plot (weather_data_path not provided)")

        print(f"\n{'='*60}")
        print("Attention analysis completed!")
        print(f"{'='*60}")

    # ==================== 节点嵌入t-SNE可视化 (新增) ====================
    if 'has_node_embeddings' in data and data['has_node_embeddings']:
        print("\n  [12/12] Generating node embedding t-SNE clustering plot...")

        try:
            # 加载站点信息
            station_info_path = Path(__file__).parent.parent.parent / 'data' / 'station_info.npy'
            if station_info_path.exists():
                station_info = np.load(station_info_path)
                station_ids = station_info[:, 0]  # 列0: WMO编码
                altitudes = station_info[:, 3]    # 列3: 海拔高度

                plot_tsne_clusters(
                    tsne_2d=data['tsne_2d'],
                    altitudes=altitudes,
                    station_ids=station_ids,
                    save_path=save_dir / 'node_embedding_tsne.png'
                )
            else:
                print(f"    ⚠ Station info not found, skipping t-SNE visualization")
        except Exception as e:
            print(f"    ⚠ t-SNE visualization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n  Skipping node embedding t-SNE (model does not support node embeddings)")

    print(f"\n✓ Visualization generation completed!")


# ==============================================================================
# Cross-Attention Feature-Level Analysis Visualization (v3.0)
# ==============================================================================

def plot_cross_attention_by_dynamic_feature(
    attention_weights,
    dynamic_feature_values,
    static_feature_names,
    dynamic_feature_name,
    num_bins=5,
    group_mapping=None,
    save_path=None,
    figsize=(18, 8)
):
    """
    Plot Cross-Attention heatmap showing how static feature attention
    changes with dynamic feature values.

    This function visualizes the relationship between dynamic weather
    conditions and the model's attention to different static geographic
    features. It answers the question: "When wind speed is high, which
    static features does the model focus on?"

    Args:
        attention_weights: np.ndarray [num_samples, num_heads, num_static_features]
            Cross-attention weights from the model
        dynamic_feature_values: np.ndarray [num_samples] or [num_samples, num_nodes]
            Values of the dynamic feature to bin by (e.g., wind speed)
        static_feature_names: List[str]
            Names of static features (e.g., ['x', 'y', 'height', ...])
        dynamic_feature_name: str
            Name of the dynamic feature for axis label (e.g., 'Wind Speed (m/s)')
        num_bins: int
            Number of bins for dynamic feature (default: 5)
        group_mapping: dict, optional
            Mapping for grouped subplot. Format:
            {'Geographic': [0,1,2], 'Building': [3,4,5,6,7], 'LandCover': [8,9,10,11]}
            If None, uses default STATIC_FEATURE_GROUPS
        save_path: str or Path, optional
            Path to save the figure
        figsize: tuple
            Figure size (width, height)

    Returns:
        dict: Statistics including bin edges and mean attention per bin

    Output:
        Two-subplot figure:
        - Left: All static features heatmap (12 features x num_bins)
        - Right: Grouped features heatmap (3 groups x num_bins)

    Example:
        >>> attention_weights = model_output['attention_weights']  # [100, 4, 12]
        >>> wind_speed = dynamic_features[:, :, 9].mean(axis=1)    # [100]
        >>> plot_cross_attention_by_dynamic_feature(
        ...     attention_weights, wind_speed,
        ...     static_feature_names=['x', 'y', 'height', ...],
        ...     dynamic_feature_name='Wind Speed (m/s)',
        ...     save_path='cross_attention_wind.png'
        ... )
    """
    from .utils import STATIC_FEATURE_GROUPS

    # Handle multi-dimensional dynamic feature (average across nodes)
    if dynamic_feature_values.ndim > 1:
        dynamic_feature_values = np.mean(dynamic_feature_values, axis=1)

    # 处理不同维度的注意力权重，兼容新旧格式
    if attention_weights.ndim == 4:
        # 旧格式: [num_samples, num_nodes, num_heads, num_features]
        attention_mean = np.mean(attention_weights, axis=(1, 2))
    elif attention_weights.ndim == 3:
        # 新格式: [num_samples, num_heads, num_features]
        attention_mean = np.mean(attention_weights, axis=1)
    elif attention_weights.ndim == 2:
        # 已是期望格式: [num_samples, num_features]
        attention_mean = attention_weights
    else:
        raise ValueError(
            f"Unexpected attention_weights shape: {attention_weights.shape}, "
            f"expected 2, 3, or 4 dimensions"
        )

    # Create bins for dynamic feature
    bin_edges = np.percentile(
        dynamic_feature_values,
        np.linspace(0, 100, num_bins + 1)
    )
    bin_labels = []
    for i in range(num_bins):
        bin_labels.append(f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}')

    # Bin the samples
    bin_indices = np.digitize(dynamic_feature_values, bin_edges[1:-1])

    # Calculate mean attention for each bin
    # Shape: [num_bins, num_static_features]
    binned_attention = np.zeros((num_bins, attention_mean.shape[1]))
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) > 0:
            binned_attention[bin_idx] = np.mean(attention_mean[mask], axis=0)

    # Use default group mapping if not provided
    if group_mapping is None:
        group_mapping = {
            name: info['indices']
            for name, info in STATIC_FEATURE_GROUPS.items()
        }

    # Calculate grouped attention
    grouped_attention = {}
    group_names = list(group_mapping.keys())
    for group_name, indices in group_mapping.items():
        grouped_attention[group_name] = np.mean(
            binned_attention[:, indices], axis=1
        )

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=300)

    # ==================== Left: All Features Heatmap ====================
    # Transpose for correct orientation: features on Y-axis, bins on X-axis
    heatmap_data = binned_attention.T  # [num_features, num_bins]

    im1 = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(num_bins))
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_yticks(range(len(static_feature_names)))
    ax1.set_yticklabels(static_feature_names, fontsize=10)
    ax1.set_xlabel(f'{dynamic_feature_name} Bins', fontsize=12, weight='bold')
    ax1.set_ylabel('Static Features', fontsize=12, weight='bold')
    ax1.set_title('Feature-Level Cross-Attention Weights',
                  fontsize=13, weight='bold')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('Attention Weight', fontsize=10)

    # Add value annotations
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data[i, j]
            text_color = 'white' if value > 0.5 * heatmap_data.max() else 'black'
            ax1.text(j, i, f'{value:.3f}', ha='center', va='center',
                     fontsize=7, color=text_color)

    # ==================== Right: Grouped Features Heatmap ====================
    grouped_data = np.array([grouped_attention[g] for g in group_names])

    im2 = ax2.imshow(grouped_data, cmap='YlOrRd', aspect='auto')
    ax2.set_xticks(range(num_bins))
    ax2.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_yticks(range(len(group_names)))
    ax2.set_yticklabels(group_names, fontsize=11)
    ax2.set_xlabel(f'{dynamic_feature_name} Bins', fontsize=12, weight='bold')
    ax2.set_ylabel('Feature Groups', fontsize=12, weight='bold')
    ax2.set_title('Grouped Cross-Attention Weights',
                  fontsize=13, weight='bold')

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label('Attention Weight', fontsize=10)

    # Add value annotations
    for i in range(grouped_data.shape[0]):
        for j in range(grouped_data.shape[1]):
            value = grouped_data[i, j]
            text_color = 'white' if value > 0.5 * grouped_data.max() else 'black'
            ax2.text(j, i, f'{value:.3f}', ha='center', va='center',
                     fontsize=9, color=text_color)

    plt.suptitle(
        f'Cross-Attention Analysis: Static Feature Importance vs {dynamic_feature_name}',
        fontsize=14, weight='bold', y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Cross-attention heatmap saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return {
        'bin_edges': bin_edges,
        'bin_labels': bin_labels,
        'binned_attention': binned_attention,
        'grouped_attention': grouped_attention
    }


def plot_node_embedding_tsne(
    tsne_2d,
    color_values,
    color_label,
    station_ids=None,
    save_path=None,
    cmap='viridis',
    figsize=(12, 10),
    title_suffix=''
):
    """
    Plot t-SNE visualization of node embeddings with customizable coloring.

    This is a generalized version of plot_tsne_clusters that allows coloring
    by any continuous variable (altitude, building height, NDVI, etc.).

    Args:
        tsne_2d: np.ndarray [num_nodes, 2]
            t-SNE 2D projection coordinates
        color_values: np.ndarray [num_nodes]
            Values to use for coloring (e.g., altitude, BH, NDVI)
        color_label: str
            Label for the colorbar (e.g., 'Altitude (m)', 'Building Height (m)')
        station_ids: np.ndarray [num_nodes], optional
            Station IDs for annotation
        save_path: str or Path, optional
            Path to save the figure
        cmap: str
            Matplotlib colormap name (default: 'viridis')
        figsize: tuple
            Figure size (default: (12, 10))
        title_suffix: str
            Additional text to append to title

    Returns:
        None

    Example:
        >>> tsne_2d = tsne_reduce_embeddings(node_embeddings)
        >>> plot_node_embedding_tsne(
        ...     tsne_2d,
        ...     color_values=static_features[:, 2],  # height
        ...     color_label='Altitude (m)',
        ...     save_path='tsne_height.png'
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Create scatter plot
    scatter = ax.scatter(
        tsne_2d[:, 0],
        tsne_2d[:, 1],
        c=color_values,
        cmap=cmap,
        s=200,
        edgecolors='black',
        linewidths=1.5,
        alpha=0.85
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(color_label, fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Annotate station IDs
    if station_ids is not None:
        for i, (x, y) in enumerate(tsne_2d):
            label = str(int(station_ids[i])) if station_ids[i] > 1000 else str(i)
            ax.text(
                x, y + 0.1,  # Slight offset above point
                label,
                fontsize=8,
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.7
                )
            )

    # Labels and title
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13, weight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13, weight='bold')
    title = f'Node Embedding t-SNE (Colored by {color_label})'
    if title_suffix:
        title += f' {title_suffix}'
    ax.set_title(title, fontsize=14, weight='bold', pad=15)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] t-SNE plot saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_node_embedding_tsne_cluster(
    tsne_2d,
    node_embeddings,
    station_ids=None,
    n_clusters=4,
    save_path=None,
    figsize=(12, 10)
):
    """
    Plot t-SNE visualization with automatic K-Means clustering.

    Uses K-Means algorithm to automatically discover cluster patterns
    in the node embeddings, then visualizes on the t-SNE projection.

    Args:
        tsne_2d: np.ndarray [num_nodes, 2]
            t-SNE 2D projection coordinates
        node_embeddings: np.ndarray [num_nodes, embedding_dim]
            Original high-dimensional node embeddings (used for clustering)
        station_ids: np.ndarray [num_nodes], optional
            Station IDs for annotation
        n_clusters: int
            Number of clusters for K-Means (default: 4)
        save_path: str or Path, optional
            Path to save the figure
        figsize: tuple
            Figure size (default: (12, 10))

    Returns:
        dict: {'cluster_labels': np.ndarray, 'cluster_centers': np.ndarray}

    Example:
        >>> result = plot_node_embedding_tsne_cluster(
        ...     tsne_2d, node_embeddings, n_clusters=4,
        ...     save_path='tsne_cluster.png'
        ... )
        >>> print(f"Cluster labels: {result['cluster_labels']}")
    """
    from sklearn.cluster import KMeans

    # Perform K-Means clustering on original embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(node_embeddings)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Color palette for clusters
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        ax.scatter(
            tsne_2d[mask, 0],
            tsne_2d[mask, 1],
            c=[colors[cluster_id]],
            label=f'Cluster {cluster_id + 1} (n={np.sum(mask)})',
            s=200,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.85
        )

    # Annotate station IDs
    if station_ids is not None:
        for i, (x, y) in enumerate(tsne_2d):
            label = str(int(station_ids[i])) if station_ids[i] > 1000 else str(i)
            ax.text(
                x, y + 0.1,
                label,
                fontsize=8,
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.7
                )
            )

    # Labels and title
    ax.set_xlabel('t-SNE Dimension 1', fontsize=13, weight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=13, weight='bold')
    ax.set_title(f'Node Embedding t-SNE with K-Means Clustering (K={n_clusters})',
                 fontsize=14, weight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] t-SNE cluster plot saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return {
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_
    }


def plot_embedding_feature_correlation(
    node_embeddings,
    static_features,
    feature_names,
    save_path=None,
    figsize=(10, 8)
):
    """
    Plot correlation heatmap between node embedding dimensions and static features.

    Visualizes the Pearson correlation between each dimension of the learned
    node embeddings and each static feature. This helps understand what
    physical/geographic information the embeddings have captured.

    Args:
        node_embeddings: np.ndarray [num_nodes, embedding_dim]
            Learned node embeddings from the model
        static_features: np.ndarray [num_nodes, num_static_features]
            Static features of each node
        feature_names: List[str]
            Names of static features
        save_path: str or Path, optional
            Path to save the figure
        figsize: tuple
            Figure size (default: (10, 8))

    Returns:
        np.ndarray: Correlation matrix [embedding_dim, num_static_features]

    Interpretation:
        - High positive correlation: embedding dimension encodes that feature
        - High negative correlation: embedding dimension encodes inverse
        - Near zero: embedding dimension independent of that feature

    Example:
        >>> corr_matrix = plot_embedding_feature_correlation(
        ...     node_embeddings,  # [28, 4]
        ...     static_features,  # [28, 12]
        ...     feature_names=['x', 'y', 'height', ...],
        ...     save_path='embedding_correlation.png'
        ... )
    """
    embedding_dim = node_embeddings.shape[1]
    num_features = min(len(feature_names), static_features.shape[1])

    # Calculate correlation matrix
    corr_matrix = np.zeros((embedding_dim, num_features))

    for i in range(embedding_dim):
        for j in range(num_features):
            corr = np.corrcoef(node_embeddings[:, i], static_features[:, j])[0, 1]
            corr_matrix[i, j] = corr

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1)

    # Labels
    embedding_labels = [f'Emb_{i}' for i in range(embedding_dim)]
    ax.set_xticks(range(num_features))
    ax.set_xticklabels(feature_names[:num_features], rotation=45,
                       ha='right', fontsize=10)
    ax.set_yticks(range(embedding_dim))
    ax.set_yticklabels(embedding_labels, fontsize=11)

    ax.set_xlabel('Static Features', fontsize=12, weight='bold')
    ax.set_ylabel('Embedding Dimensions', fontsize=12, weight='bold')
    ax.set_title('Node Embedding vs Static Feature Correlation',
                 fontsize=14, weight='bold', pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Pearson Correlation', fontsize=11)

    # Add value annotations
    for i in range(embedding_dim):
        for j in range(num_features):
            value = corr_matrix[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color, weight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] Embedding correlation plot saved: {save_path}")
    else:
        plt.show()

    plt.close()

    return corr_matrix


# ==============================================================================
# GAT_SeparateEncoder Specialized Visualization Entry Point
# ==============================================================================

def generate_gat_separate_encoder_visualizations(
    model,
    test_loader,
    device,
    save_dir,
    station_info,
    static_feature_names,
    dynamic_feature_configs,
    tsne_color_configs,
    num_samples=100,
    n_clusters=4
):
    """
    Generate all visualizations specific to GAT_SeparateEncoder model.

    This is the main entry point for GAT_SeparateEncoder explainability
    visualizations, including Cross-Attention analysis and Node Embedding
    t-SNE plots.

    Args:
        model: GAT_SeparateEncoder model instance
        test_loader: PyTorch DataLoader for test set
        device: torch device
        save_dir: Directory to save visualizations
        station_info: np.ndarray [num_nodes, 4] - [id, lon, lat, height]
        static_feature_names: List[str] - Names of 12 static features
        dynamic_feature_configs: List[dict] - Dynamic features to analyze
            Example: [{'name': 'Wind Speed', 'index': 9, 'unit': 'm/s'}, ...]
        tsne_color_configs: List[dict] - Features for t-SNE coloring
            Example: [{'name': 'height', 'index': 2, 'label': 'Altitude (m)'}, ...]
        num_samples: int - Number of samples for analysis
        n_clusters: int - Number of clusters for K-Means

    Returns:
        dict: Analysis results including attention weights, embeddings, etc.

    Example:
        >>> results = generate_gat_separate_encoder_visualizations(
        ...     model, test_loader, 'cuda',
        ...     save_dir='visualizations/',
        ...     station_info=station_info,
        ...     static_feature_names=['x', 'y', 'height', ...],
        ...     dynamic_feature_configs=[
        ...         {'name': 'Wind Speed', 'index': 9, 'unit': 'm/s'}
        ...     ],
        ...     tsne_color_configs=[
        ...         {'name': 'height', 'index': 2, 'label': 'Altitude (m)'}
        ...     ]
        ... )
    """
    import torch
    from pathlib import Path
    from .utils import (
        extract_cross_attention_weights,
        extract_node_embeddings,
        tsne_reduce_embeddings,
        save_node_embedding_analysis,
        STATIC_FEATURE_GROUPS
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    results = {}

    print("\n" + "=" * 70)
    print("GAT_SeparateEncoder Explainability Analysis")
    print("=" * 70)

    # ==================== 1. Cross-Attention Analysis ====================
    print("\n[1/3] Extracting Cross-Attention weights...")

    try:
        cross_attn_data = extract_cross_attention_weights(
            model, test_loader, device, num_samples=num_samples
        )
        results['cross_attention'] = cross_attn_data

        print(f"  Attention weights shape: {cross_attn_data['attention_weights'].shape}")
        print(f"  Samples collected: {len(cross_attn_data['sample_indices'])}")

        # Generate heatmaps for each configured dynamic feature
        print("\n[2/3] Generating Cross-Attention heatmaps...")

        for feat_config in dynamic_feature_configs:
            feat_name = feat_config['name']
            feat_idx = feat_config['index']
            feat_unit = feat_config.get('unit', '')

            print(f"  Processing: {feat_name}...")

            # Extract dynamic feature values
            # 处理不同维度的动态特征数据，兼容新旧格式
            dynamic_features = cross_attn_data['dynamic_features']

            if dynamic_features.ndim == 4:
                # 旧格式: [num_samples, num_nodes, hist_len, dynamic_dim]
                # 取最后一个时间步，然后在节点维度求平均
                dynamic_values = dynamic_features[:, :, -1, feat_idx]
                dynamic_values_mean = np.mean(dynamic_values, axis=1)
            elif dynamic_features.ndim == 3:
                # 新格式: [num_samples, hist_len, dynamic_dim]
                # 取最后一个时间步
                dynamic_values_mean = dynamic_features[:, -1, feat_idx]
            elif dynamic_features.ndim == 2:
                # [num_samples, dynamic_dim]
                dynamic_values_mean = dynamic_features[:, feat_idx]
            else:
                raise ValueError(
                    f"Unexpected dynamic_features shape: {dynamic_features.shape}"
                )

            # Generate heatmap
            label = f'{feat_name} ({feat_unit})' if feat_unit else feat_name
            safe_name = feat_name.lower().replace(' ', '_')

            plot_cross_attention_by_dynamic_feature(
                attention_weights=cross_attn_data['attention_weights'],
                dynamic_feature_values=dynamic_values_mean,
                static_feature_names=static_feature_names,
                dynamic_feature_name=label,
                group_mapping={
                    name: info['indices']
                    for name, info in STATIC_FEATURE_GROUPS.items()
                },
                save_path=save_dir / f'cross_attention_{safe_name}.png'
            )

    except Exception as e:
        print(f"  [ERROR] Cross-Attention analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # ==================== 2. Node Embedding t-SNE ====================
    print("\n[3/3] Generating Node Embedding visualizations...")

    try:
        # Extract node embeddings
        node_embeddings = extract_node_embeddings(model)
        results['node_embeddings'] = node_embeddings
        print(f"  Node embeddings shape: {node_embeddings.shape}")

        # t-SNE dimensionality reduction
        tsne_2d = tsne_reduce_embeddings(
            node_embeddings, perplexity=min(10, node_embeddings.shape[0] // 3)
        )
        results['tsne_2d'] = tsne_2d

        # 获取每个节点的静态特征用于 t-SNE 着色
        # 静态特征是不随时间变化的，使用配置中的静态特征索引
        # 从原始气象数据加载（如果可用）或使用 station_info
        num_nodes = node_embeddings.shape[0]

        # 尝试从 test_loader 的第一个样本获取静态特征
        # 注意：PyG batch 可能合并多个图，需要只取第一个图的节点
        static_dim = model.static_dim
        first_batch = next(iter(test_loader))
        first_data = first_batch[0] if isinstance(
            first_batch, (list, tuple)
        ) else first_batch

        # 检查数据形状，只取前 num_nodes 个节点的数据
        all_nodes = first_data.x.shape[0]
        if all_nodes > num_nodes:
            # PyG 合并了多个图，只取第一个图
            static_features = first_data.x[:num_nodes, 0, :static_dim].cpu().numpy()
        else:
            static_features = first_data.x[:, 0, :static_dim].cpu().numpy()

        # Generate t-SNE plots colored by different features
        station_ids = station_info[:, 0]  # WMO codes

        for color_config in tsne_color_configs:
            feat_name = color_config['name']
            feat_idx = color_config['index']
            feat_label = color_config['label']

            print(f"  Generating t-SNE colored by {feat_name}...")

            # Get color values
            if feat_idx < static_features.shape[1]:
                color_values = static_features[:, feat_idx]
            else:
                # Try from station_info (e.g., height is column 3)
                color_values = station_info[:, 3]  # Default to height

            safe_name = feat_name.lower().replace(' ', '_')

            plot_node_embedding_tsne(
                tsne_2d,
                color_values=color_values,
                color_label=feat_label,
                station_ids=station_ids,
                save_path=save_dir / f'node_embedding_tsne_{safe_name}.png',
                cmap='viridis' if feat_name != 'height' else 'terrain'
            )

        # Generate K-Means cluster plot
        print(f"  Generating t-SNE with K-Means clustering (K={n_clusters})...")
        cluster_result = plot_node_embedding_tsne_cluster(
            tsne_2d,
            node_embeddings,
            station_ids=station_ids,
            n_clusters=n_clusters,
            save_path=save_dir / 'node_embedding_tsne_cluster.png'
        )
        results['cluster_labels'] = cluster_result['cluster_labels']

        # Generate embedding-feature correlation plot
        print("  Generating embedding-feature correlation heatmap...")
        corr_matrix = plot_embedding_feature_correlation(
            node_embeddings,
            static_features,
            feature_names=static_feature_names,
            save_path=save_dir / 'embedding_correlation.png'
        )
        results['embedding_correlation'] = corr_matrix

        # Save CSV
        print("  Saving node embedding analysis to CSV...")
        csv_path = save_node_embedding_analysis(
            node_embeddings=node_embeddings,
            tsne_2d=tsne_2d,
            station_info=station_info,
            static_features=static_features,
            feature_names=static_feature_names,
            save_path=save_dir / 'node_embeddings.csv'
        )
        print(f"  [OK] CSV saved: {csv_path}")

    except Exception as e:
        print(f"  [ERROR] Node embedding analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("GAT_SeparateEncoder Explainability Analysis Completed!")
    print("=" * 70)

    return results
