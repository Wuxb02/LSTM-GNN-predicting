"""
训练结果可视化脚本(固定配置版)

为GNN气温预测模型生成完整的可视化分析:
- 每个预测步长独立的可视化文件夹
- 28个站点的时间序列对比图
- RMSE空间分布地图
- 多步长性能对比
- 保存所有绘图数据供后续分析

使用方法:
    1. 修改下面的配置区域
    2. 直接运行: python visualize_results.py

输出:
    - checkpoints/模型名/visualizations/step_N/ (每个预测步长)
    - checkpoints/模型名/visualizations/summary/ (多步长汇总)

作者: GNN气温预测项目
日期: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from pathlib import Path
import sys
from scipy.stats import gaussian_kde, t

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from myGNN.network_GNN import get_metric

# 地理底图相关导入
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib_scalebar.scalebar import ScaleBar
    BASEMAP_AVAILABLE = True
except ImportError as e:
    BASEMAP_AVAILABLE = False
    print(f"⚠ 地理底图功能不可用: {e}")
    print(f"  如需使用底图,请安装: pip install cartopy matplotlib-scalebar")


# ==================== 配置区域 ====================
# 修改这里的配置来可视化不同的模型

# 模型checkpoint目录(必填)
CHECKPOINT_DIR = r'.\checkpoints\GAT_SeparateEncoder_20251221_235912'

# 输出目录('auto'表示自动在checkpoint下创建visualizations/)
OUTPUT_DIR = 'auto'

# 显示哪些预测步长('all'表示全部, 或列表如[0,1,2])
PRED_STEPS = 'all'  # 'all' 或 [0, 1, 2]

# 是否为所有28个站点生成时间序列图
PLOT_ALL_STATIONS = True

# 如果不绘制全部,选择哪些站点
SAMPLE_STATIONS = [0, 5, 10, 15, 20, 25]

# 时间序列采样率(1=全部点, 5=每5个点显示1个)
TIME_SAMPLE_RATE = 1

# 是否保存中间数据(用于后续分析)
SAVE_INTERMEDIATE_DATA = True

# 图表DPI
DPI = 300

# 是否尝试配置中文字体（已禁用，使用Arial英文）
USE_CHINESE = False

# ==================== 地理底图配置 ====================
# 是否生成带地理底图的RMSE空间分布图
USE_BASEMAP = True

# 底图配置
# True: 使用Mapbox WMTS在线底图, False: 使用Natural Earth离线矢量
USE_WMTS_BASEMAP = False

# 是否添加比例尺和指北针
ADD_SCALEBAR = False
ADD_NORTH_ARROW = False

# ================================================


def setup_font():
    """配置Arial字体用于英文图表"""
    try:
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        print("✓ Arial字体已配置")
    except Exception:
        print("⚠ 字体配置失败")


class ResultVisualizer:
    """训练结果可视化器"""

    def __init__(self, checkpoint_dir):
        """
        初始化可视化器

        Args:
            checkpoint_dir: checkpoint目录路径
        """
        self.checkpoint_dir = Path(checkpoint_dir)

        # 尝试多种路径
        if not self.checkpoint_dir.exists():
            # 尝试补全myGNN前缀
            alt_path = Path('myGNN') / checkpoint_dir
            if alt_path.exists():
                self.checkpoint_dir = alt_path
            else:
                # 尝试相对于脚本位置
                script_dir = Path(__file__).parent
                alt_path2 = script_dir / checkpoint_dir
                if alt_path2.exists():
                    self.checkpoint_dir = alt_path2
                else:
                    raise FileNotFoundError(
                        f"Checkpoint目录不存在: {checkpoint_dir}\n"
                        f"已尝试路径:\n"
                        f"  - {Path(checkpoint_dir).absolute()}\n"
                        f"  - {alt_path.absolute()}\n"
                        f"  - {alt_path2.absolute()}\n"
                        f"请检查CHECKPOINT_DIR配置是否正确"
                    )

        print(f"加载checkpoint: {self.checkpoint_dir}")

        # 加载数据
        self.load_results()
        self.load_station_info()

        print(f"✓ 数据加载完成")
        print(f"  测试样本数: {self.num_samples}")
        print(f"  站点数: {self.num_stations}")
        print(f"  预测步长: {self.pred_len}")

    def load_results(self):
        """加载测试集结果"""
        test_pred_path = self.checkpoint_dir / 'test_predict.npy'
        test_label_path = self.checkpoint_dir / 'test_label.npy'
        test_time_path = self.checkpoint_dir / 'test_time.npy'

        if not test_pred_path.exists():
            raise FileNotFoundError(
                f"未找到测试集预测结果: {test_pred_path}\n"
                f"请确保已运行过训练脚本并保存了测试集结果"
            )

        self.test_predict = np.load(test_pred_path)
        self.test_label = np.load(test_label_path)
        self.test_time = np.load(test_time_path)

        print(f"  数据shape:")
        print(f"    test_predict: {self.test_predict.shape}")
        print(f"    test_label: {self.test_label.shape}")
        print(f"    test_time: {self.test_time.shape}")

        # 处理数据维度
        if len(self.test_predict.shape) == 2:
            # 如果是2维 [num_samples, num_stations],添加pred_len维度
            print(f"  检测到2维数据,自动转换为3维 (添加pred_len=1)")
            self.test_predict = self.test_predict[:, :, np.newaxis]  # [N, S, 1]
            self.test_label = self.test_label[:, :, np.newaxis]
            self.num_samples, self.num_stations, self.pred_len = self.test_predict.shape
        elif len(self.test_predict.shape) == 3:
            # 标准3维数据: [num_samples, num_stations, pred_len]
            self.num_samples, self.num_stations, self.pred_len = self.test_predict.shape
        else:
            raise ValueError(
                f"不支持的数据维度: {self.test_predict.shape}\n"
                f"期望维度: [num_samples, num_stations, pred_len] 或 [num_samples, num_stations]"
            )

        # 检查NaN
        nan_count = np.isnan(self.test_predict).sum()
        if nan_count > 0:
            print(f"  ⚠ 警告: 预测值包含{nan_count}个NaN")

    def load_station_info(self):
        """加载站点信息(经纬度)"""
        station_info_path = project_root / 'data' / 'station_info.npy'

        if not station_info_path.exists():
            print(f"  ⚠ 警告: 未找到站点信息文件: {station_info_path}")
            print("  将使用默认站点ID作为坐标")
            # 使用默认值
            self.station_ids = np.arange(self.num_stations)
            self.lon = np.arange(self.num_stations)
            self.lat = np.arange(self.num_stations)
            self.height = np.zeros(self.num_stations)
        else:
            station_info = np.load(station_info_path)
            # shape: [N, 4] → [id, lon, lat, height]

            # 检查站点数量是否匹配
            if station_info.shape[0] != self.num_stations:
                print(f"  ⚠ 警告: station_info有{station_info.shape[0]}个站点,"
                      f"但数据有{self.num_stations}个站点")
                print(f"  将只使用前{self.num_stations}个站点的信息")

            # 只取需要的站点数量
            num_to_use = min(station_info.shape[0], self.num_stations)
            self.station_ids = station_info[:num_to_use, 0].astype(int)
            self.lon = station_info[:num_to_use, 1]
            self.lat = station_info[:num_to_use, 2]
            self.height = station_info[:num_to_use, 3]

            # 如果数据站点数更多,用默认值填充
            if self.num_stations > num_to_use:
                print(f"  用默认值填充剩余{self.num_stations - num_to_use}个站点")
                extra_ids = np.arange(num_to_use, self.num_stations)
                self.station_ids = np.concatenate([self.station_ids, extra_ids])
                self.lon = np.concatenate([self.lon, extra_ids.astype(float)])
                self.lat = np.concatenate([self.lat, extra_ids.astype(float)])
                self.height = np.concatenate([self.height, np.zeros(self.num_stations - num_to_use)])

    def calculate_metrics_for_step(self, pred_step):
        """
        计算某一预测步长的所有站点指标

        Args:
            pred_step: 预测步长索引(0表示第1步)

        Returns:
            dict: 包含各种指标的字典
        """
        metrics = {
            'rmse_per_station': [],
            'mae_per_station': [],
            'r2_per_station': [],
            'bias_per_station': []
        }

        # 提取该步长的数据
        pred_step_data = self.test_predict[:, :, pred_step]  # [num_samples, num_stations]
        label_step_data = self.test_label[:, :, pred_step]

        # 计算每个站点的指标
        for station_id in range(self.num_stations):
            pred_station = pred_step_data[:, station_id]  # [num_samples]
            label_station = label_step_data[:, station_id]

            rmse, mae, r2, bias = get_metric(pred_station, label_station)

            metrics['rmse_per_station'].append(rmse)
            metrics['mae_per_station'].append(mae)
            metrics['r2_per_station'].append(r2)
            metrics['bias_per_station'].append(bias)

        # 转换为numpy数组
        for key in metrics:
            metrics[key] = np.array(metrics[key])

        # 计算整体指标
        pred_all = pred_step_data.flatten()
        label_all = label_step_data.flatten()

        overall_rmse, overall_mae, overall_r2, overall_bias = get_metric(
            pred_all, label_all
        )

        metrics['overall_rmse'] = overall_rmse
        metrics['overall_mae'] = overall_mae
        metrics['overall_r2'] = overall_r2
        metrics['overall_bias'] = overall_bias

        return metrics

    def save_plot_data(self, output_dir, pred_step, metrics):
        """
        保存绘图数据为.npz格式

        Args:
            output_dir: 输出目录
            pred_step: 预测步长索引
            metrics: 指标字典
        """
        save_path = output_dir / 'plot_data.npz'

        # 提取该步长的预测和标签
        predictions = self.test_predict[:, :, pred_step]  # [num_samples, 28]
        labels = self.test_label[:, :, pred_step]

        np.savez(
            save_path,
            # 原始预测和标签
            predictions=predictions,
            labels=labels,
            time_indices=self.test_time,

            # 站点信息
            station_ids=self.station_ids,
            lon=self.lon,
            lat=self.lat,
            height=self.height,

            # 每个站点的指标
            rmse_per_station=metrics['rmse_per_station'],
            mae_per_station=metrics['mae_per_station'],
            r2_per_station=metrics['r2_per_station'],
            bias_per_station=metrics['bias_per_station'],

            # 整体指标
            overall_rmse=metrics['overall_rmse'],
            overall_mae=metrics['overall_mae'],
            overall_r2=metrics['overall_r2'],
            overall_bias=metrics['overall_bias'],

            # 元数据
            pred_step=pred_step,
            checkpoint_dir=str(self.checkpoint_dir)
        )

        print(f"  ✓ 绘图数据已保存: {save_path}")

    def plot_station_timeseries(self, station_id, pred_step, save_path):
        """
        绘制单个站点的时间序列对比图

        Args:
            station_id: 站点ID
            pred_step: 预测步长索引
            save_path: 保存路径
        """
        # 提取数据
        pred = self.test_predict[:, station_id, pred_step]
        label = self.test_label[:, station_id, pred_step]
        time_idx = self.test_time

        # 采样(如果需要)
        if TIME_SAMPLE_RATE > 1:
            indices = np.arange(0, len(pred), TIME_SAMPLE_RATE)
            pred = pred[indices]
            label = label[indices]
            time_idx = time_idx[indices]

        # 计算指标
        rmse, mae, r2, bias = get_metric(
            self.test_predict[:, station_id, pred_step],
            self.test_label[:, station_id, pred_step]
        )

        # 计算误差带(±1σ)
        errors = pred - label
        error_std = np.std(errors)

        # 绘图
        fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI)

        # 真实值(蓝色实线)
        ax.plot(range(len(label)), label, 'o-', color='#1f77b4',
                label='Ground Truth', linewidth=1.5, markersize=3, alpha=0.8)

        # 预测值(橙色虚线)
        ax.plot(range(len(pred)), pred, 's--', color='#ff7f0e',
                label='Prediction', linewidth=1.5, markersize=3, alpha=0.8)

        # 误差带(半透明灰色)
        ax.fill_between(range(len(pred)),
                        pred - error_std, pred + error_std,
                        color='gray', alpha=0.2, label='±1σ Error Band')

        # 设置标签和标题
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Temperature (°C)', fontsize=11)
        ax.set_title(
            f'Station {station_id} - Step {pred_step+1} Prediction Comparison\n'
            f'RMSE: {rmse:.4f}°C, MAE: {mae:.4f}°C, R²: {r2:.4f}, Bias: {bias:+.4f}°C',
            fontsize=12, pad=15
        )

        # 网格和图例
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def plot_all_timeseries(self, pred_step, output_dir):
        """
        批量生成28个站点的时间序列图

        Args:
            pred_step: 预测步长索引
            output_dir: 输出目录
        """
        ts_dir = output_dir / 'timeseries'
        ts_dir.mkdir(exist_ok=True)

        stations = (range(self.num_stations) if PLOT_ALL_STATIONS
                   else SAMPLE_STATIONS)

        print(f"  生成时间序列图 (共{len(stations)}个站点)...")

        for i, station_id in enumerate(stations, 1):
            save_path = ts_dir / f'station_{station_id:02d}.png'
            self.plot_station_timeseries(station_id, pred_step, save_path)

            # 进度条
            progress = i / len(stations) * 100
            bar_len = 40
            filled_len = int(bar_len * i / len(stations))
            bar = '█' * filled_len + '░' * (bar_len - filled_len)
            print(f'\r    [{bar}] {progress:.1f}% (站点{station_id})', end='')

        print()  # 换行

    def plot_rmse_spatial_map(self, metrics, save_path):
        """
        绘制RMSE空间分布散点图

        Args:
            metrics: 指标字典
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(20, 4), dpi=DPI)

        # 散点图
        scatter = ax.scatter(
            self.lon, self.lat,
            c=metrics['rmse_per_station'],
            s=200,
            cmap='RdYlGn_r',  # 红→黄→绿(反转)
            edgecolors='black',
            linewidth=1.5,
            alpha=0.8,
            vmin=np.percentile(metrics['rmse_per_station'], 5),
            vmax=np.percentile(metrics['rmse_per_station'], 95)
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('RMSE (°C)', fontsize=11)

        # 添加站点标签
        for i in range(self.num_stations):
            ax.annotate(
                f'{i}',
                (self.lon[i], self.lat[i]),
                fontsize=8,
                ha='center',
                va='center'
            )

        # 设置标签和标题
        ax.set_xlabel('Longitude (°E)', fontsize=11)
        ax.set_ylabel('Latitude (°N)', fontsize=11)

        ax.grid(True, alpha=0.3, linestyle='--')

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def plot_rmse_spatial_map_with_basemap(self, metrics, save_path, annotation_type='ids'):
        """
        绘制带专业地理底图的RMSE空间分布图 (使用cartopy)

        Args:
            metrics: 指标字典
            save_path: 保存路径
            annotation_type: 标注类型 ('ids'=站点下标, 'values'=RMSE数值)
        """
        if not BASEMAP_AVAILABLE:
            print(f"  ⚠ 跳过底图版本(依赖未安装): {save_path.name}")
            return
        fontsize = 16
        # 创建cartopy地理坐标轴
        fig = plt.figure(figsize=(10, 10), dpi=DPI)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # 散点图 (直接使用经纬度)
        scatter = ax.scatter(
            self.lon, self.lat,
            transform=ccrs.PlateCarree(),  # 关键!指定数据投影
            c=metrics['rmse_per_station'],
            s=500,
            cmap='RdYlGn_r',
            edgecolors='black',
            linewidth=1.5,
            # alpha=0.85,
            vmin=np.percentile(metrics['rmse_per_station'], 5),
            vmax=np.percentile(metrics['rmse_per_station'], 95),
            zorder=5  # 确保散点在底图之上
        )

        # 添加标注
        if annotation_type == 'ids':
            # 版本1: 在散点内部标注站点下标
            for i in range(self.num_stations):
                ax.text(self.lon[i], self.lat[i], str(i),
                       transform=ccrs.PlateCarree(),  # 关键!指定数据投影
                       fontsize=fontsize, ha='center', va='center',
                       color='black', weight='bold', zorder=6)
        else:
            # 版本2: 在散点旁边标注RMSE数值
            for i in range(self.num_stations):
                rmse_val = metrics['rmse_per_station'][i]
                ax.text(self.lon[i], self.lat[i], f'{rmse_val:.2f}',
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
            print(f"  ✓ Mapbox WMTS底图加载成功")

        else:
            # 直接使用Natural Earth
            print(f"  使用Natural Earth离线底图...")
            from myGNN.utils.cartopy_helpers import add_basemap_features
            add_basemap_features(ax, style='natural_earth', add_gridlines=False)
            basemap_loaded = True
            basemap_name = 'Natural Earth'
            print(f"  ✓ Natural Earth底图加载成功")

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
        cbar.ax.set_title('RMSE (°C)', fontsize=fontsize)
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
        gl.xlocator = mticker.MaxNLocator(nbins=4)  #
        gl.ylocator = mticker.MaxNLocator(nbins=2)  #

        # 设置刻度标签字体大小
        gl.xlabel_style = {'size': fontsize}
        gl.ylabel_style = {'size': fontsize, 'rotation':90}

        # 添加比例尺（如果配置启用）
        if ADD_SCALEBAR and basemap_loaded:
            try:
                # cartopy PlateCarree投影下,需要根据纬度计算度→米的换算
                avg_lat = np.mean(self.lat)
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
                print(f"  ⚠ 比例尺添加失败: {e}")

        # 添加指北针（如果配置启用）
        if ADD_NORTH_ARROW and basemap_loaded:
            try:
                arrow_x, arrow_y = 0.05, 0.92

                # 指北针文字
                ax.text(arrow_x, arrow_y + 0.05, 'N',
                       transform=ax.transAxes, fontsize=14, ha='center', va='center',
                       weight='bold',
                       bbox=dict(boxstyle='circle', facecolor='white',
                                edgecolor='black', linewidth=1.5, alpha=0.8))

                # 指北箭头
                ax.annotate('', xy=(arrow_x, arrow_y + 0.04), xytext=(arrow_x, arrow_y),
                          xycoords='axes fraction', textcoords='axes fraction',
                          arrowprops=dict(arrowstyle='->', lw=2, color='black'))
            except Exception as e:
                print(f"  ⚠ 指北针添加失败: {e}")

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✓ 带底图的RMSE空间分布图已生成")

    def plot_rmse_barplot(self, metrics, save_path):
        """
        绘制各站点RMSE对比柱状图

        Args:
            metrics: 指标字典
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(14, 5), dpi=DPI)

        rmse_values = metrics['rmse_per_station']
        mean_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)

        # 根据RMSE分级着色
        colors = []
        for rmse in rmse_values:
            if rmse < mean_rmse - 0.5 * std_rmse:
                colors.append('#2ecc71')  # 绿色(优秀)
            elif rmse > mean_rmse + 0.5 * std_rmse:
                colors.append('#e74c3c')  # 红色(较差)
            else:
                colors.append('#f39c12')  # 黄色(中等)

        # 柱状图
        bars = ax.bar(range(self.num_stations), rmse_values, color=colors,
                      edgecolor='black', linewidth=0.5, alpha=0.8)

        # 平均线
        ax.axhline(mean_rmse, color='blue', linestyle='--',
                  linewidth=2, label=f'Mean RMSE: {mean_rmse:.4f}°C', alpha=0.7)

        # 设置标签
        ax.set_xlabel('Station ID', fontsize=11)
        ax.set_ylabel('RMSE (°C)', fontsize=11)
        ax.set_title(
            f'RMSE Comparison Across Stations (Mean: {mean_rmse:.4f}°C, Std: {std_rmse:.4f}°C)',
            fontsize=12, pad=15
        )

        ax.set_xticks(range(self.num_stations))
        ax.set_xticklabels([str(i) for i in range(self.num_stations)], fontsize=9)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def plot_pred_vs_true_scatter(self, pred_step, save_path):
        """
        绘制预测vs真实值整体散点图

        Args:
            pred_step: 预测步长索引
            save_path: 保存路径
        """
        # 提取数据
        pred = self.test_predict[:, :, pred_step].flatten()
        label = self.test_label[:, :, pred_step].flatten()

        # 计算整体指标
        rmse, mae, r2, bias = get_metric(pred, label)

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8), dpi=DPI)

        # 散点图
        ax.scatter(label, pred, alpha=0.3, s=10, color='steelblue', edgecolors='none')

        # 对角线(y=x)
        min_val = min(label.min(), pred.min())
        max_val = max(label.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, label='y=x', alpha=0.7)

        # 文本标注
        text_str = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}°C\nMAE = {mae:.4f}°C\nBias = {bias:+.4f}°C'
        ax.text(0.05, 0.95, text_str,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 设置标签
        ax.set_xlabel('Ground Truth (°C)', fontsize=11)
        ax.set_ylabel('Prediction (°C)', fontsize=11)
        ax.set_title(f'Step {pred_step+1} Prediction vs Ground Truth (All Stations)', fontsize=12, pad=15)

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_aspect('equal', adjustable='box')

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def plot_pred_vs_true_marginal(self, pred_step, save_path,
                                    group_mode='season', group_threshold=20.0):
        """
        绘制带边缘密度分布的预测vs真实值散点图

        特性:
        - 顶部和右侧KDE密度曲线
        - 支持季节或温度阈值分组着色
        - 每组独立的回归拟合线和置信区间
        - 分组R²标注

        Args:
            pred_step: 预测步长索引
            save_path: 保存路径
            group_mode: 分组模式, 'season'(季节) 或 'temperature'(温度阈值)
            group_threshold: 温度分组阈值(°C), 仅在group_mode='temperature'时使用
        """
        # 1. 数据提取
        pred = self.test_predict[:, :, pred_step].flatten()
        label = self.test_label[:, :, pred_step].flatten()

        # 2. 根据分组模式确定分组
        if group_mode == 'season':
            # 从test_time索引获取月份信息
            try:
                # 加载原始气象数据获取月份 (使用绝对路径)
                project_root = Path(__file__).parent.parent
                metdata_path = project_root / 'data' / 'real_weather_data_2010_2017.npy'

                if not metdata_path.exists():
                    print(f"  ⚠ 未找到原始数据: {metdata_path}")
                    print(f"  回退到温度分组")
                    group_mode = 'temperature'
                else:
                    metdata = np.load(metdata_path)  # [2922, 28, 28]

                    # 从test_time获取时间索引,提取对应月份
                    # month特征在索引27
                    months = []
                    for time_idx in self.test_time:
                        month = metdata[int(time_idx), 0, 27]  # 提取第一个站点的月份
                        months.append(month)
                    months = np.array(months)

                    # 扩展到所有站点 (每个时间步重复num_stations次)
                    months_expanded = np.repeat(months, self.num_stations)

                    # 定义两组：夏季 vs 其他季节
                    # 夏季: 6,7,8月  其他: 3,4,5,9,10,11,12,1,2月
                    group_masks = [
                        np.isin(months_expanded, [3, 4, 5, 9, 10, 11, 12, 1, 2]),  # 其他季节
                        np.isin(months_expanded, [6, 7, 8])                         # 夏季
                    ]

                    group_names = [
                        'Non-Summer',
                        'Summer'
                    ]

                    # 统计各组数据量
                    season_counts = [np.sum(mask) for mask in group_masks]
                    print(f"  季节分组: 非夏季{season_counts[0]}个点, 夏季{season_counts[1]}个点")

            except Exception as e:
                print(f"  ⚠ 季节分组失败: {e}")
                print(f"  回退到温度分组")
                group_mode = 'temperature'

        if group_mode == 'temperature':
            # 按温度阈值分组: 低温组 vs 高温组
            group_masks = [
                label < group_threshold,
                label >= group_threshold
            ]
            group_names = [
                f'Low Temp (<{group_threshold}°C)',
                f'High Temp (≥{group_threshold}°C)'
            ]
            print(f"  温度分组: 低温{np.sum(group_masks[0])}个点, 高温{np.sum(group_masks[1])}个点")

        colors = ['#5F9EA0', '#FF7F50']
        colors_line = ['#4682B4', '#FF6347']
        colors_text = ['#2F4F4F', '#8B4513']

        fontsize = 16

        # 检查分组数据量
        for i, mask in enumerate(group_masks):
            n_points = np.sum(mask)
            if n_points < 10:
                print(f"  ⚠ 警告: {group_names[i]}数据量过少 ({n_points}个点)")

        # 3. 创建GridSpec布局
        fig = plt.figure(figsize=(10, 10), dpi=DPI)
        gs = GridSpec(5, 5, hspace=0.05, wspace=0.05)

        ax_main = fig.add_subplot(gs[1:5, 0:4])
        ax_top = fig.add_subplot(gs[0, 0:4], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:5, 4], sharey=ax_main)

        # 4. 主散点图 + 回归线 + 置信区间
        for i, mask in enumerate(group_masks):
            if np.sum(mask) < 5:  # 数据点太少则跳过
                continue

            # 提取当前组数据
            label_group = label[mask]
            pred_group = pred[mask]

            # 散点图
            ax_main.scatter(label_group, pred_group,
                           alpha=0.5, s=20, color='none',
                           edgecolors=colors[i], label=group_names[i])

            # 线性回归拟合
            if len(label_group) >= 2:
                z = np.polyfit(label_group, pred_group, 1)
                p = np.poly1d(z)

                # 拟合线
                x_line = np.linspace(label_group.min(), label_group.max(), 100)
                y_line = p(x_line)
                ax_main.plot(x_line, y_line, color=colors_line[i],
                            linewidth=2, linestyle='--', alpha=0.8)

                # 严格计算95%置信区间
                # 基于线性回归的预测区间公式
                n = len(label_group)  # 样本数量
                residuals = pred_group - p(label_group)

                # 残差标准误差 (MSE的平方根)
                mse = np.sum(residuals**2) / (n - 2)  # 自由度 = n - 2 (两个参数: 斜率和截距)
                se = np.sqrt(mse)

                # t分布临界值 (95%置信水平, 双侧)
                alpha = 0.05
                t_critical = t.ppf(1 - alpha/2, n - 2)

                # 计算预测区间
                # 对于每个预测点x_line[j], 计算标准误差
                x_mean = np.mean(label_group)
                sxx = np.sum((label_group - x_mean)**2)

                # 预测区间: y_pred ± t * se * sqrt(1 + 1/n + (x - x_mean)^2 / Sxx)
                # 三个成分:
                # 1. 固定部分: 1 (新观测的随机误差)
                # 2. 样本量部分: 1/n (参数估计的不确定性)
                # 3. 距离部分: (x - x_mean)^2 / Sxx (远离均值的惩罚)

                distance_term = (x_line - x_mean)**2 / sxx
                prediction_se = se * np.sqrt(1 + 1/n + distance_term)
                margin = t_critical * prediction_se

                # 调试信息（可选）
                center_width = t_critical * se * np.sqrt(1 + 1/n)
                edge_width = margin[0]  # 边缘宽度
                width_ratio = edge_width / center_width
                print(f"    {group_names[i]}: 中心宽度={center_width:.3f}°C, 边缘宽度={edge_width:.3f}°C, 比值={width_ratio:.2f}")

                # 绘制置信区间
                ax_main.fill_between(x_line, y_line - margin, y_line + margin,
                                    color=colors[i], alpha=0.15, label=f'95% PI ({group_names[i]})')

                # 计算R²
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((pred_group - np.mean(pred_group))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

                # R²文本标注
                text_y = 0.95 - i * 0.06
                ax_main.text(0.05, text_y,
                            f'{group_names[i]}: $R^2$={r2:.3f}',
                            transform=ax_main.transAxes,
                            fontsize=fontsize, color=colors_text[i],
                            # weight='bold',
                            # bbox=dict(boxstyle='round', facecolor='white',
                            #          edgecolor=colors[i], alpha=0.8, linewidth=2)
                                     )

        # 对角线 y=x 和坐标轴范围设置
        min_val = min(label.min(), pred.min())
        max_val = max(label.max(), pred.max())

        # 添加少量边距 (5%)
        margin = (max_val - min_val) * 0.05
        axis_min = min_val - margin
        axis_max = max_val + margin

        # 设置相同的x和y轴范围，确保正方形
        ax_main.set_xlim(axis_min, axis_max)
        ax_main.set_ylim(axis_min, axis_max)

        # 绘制对角线
        ax_main.plot([axis_min, axis_max], [axis_min, axis_max],
                    'k--', linewidth=1.5, alpha=0.5, zorder=0, label='y=x')

        # 5. 顶部密度曲线 (KDE)
        x_range = np.linspace(label.min(), label.max(), 200)

        for i, mask in enumerate(group_masks):
            if np.sum(mask) < 5:
                continue

            label_group = label[mask]

            try:
                kde_x = gaussian_kde(label_group)
                density_x = kde_x(x_range)

                ax_top.fill_between(x_range, 0, density_x,
                                   color=colors[i], alpha=0.4)
                ax_top.plot(x_range, density_x,
                           color=colors[i], linewidth=2)
            except Exception as e:
                print(f"  ⚠ KDE计算失败 ({group_names[i]}): {e}")

        ax_top.set_xlim(ax_main.get_xlim())
        ax_top.set_ylim(bottom=0)
        ax_top.axis('off')

        # 6. 右侧密度曲线 (KDE, 旋转90度)
        y_range = np.linspace(pred.min(), pred.max(), 200)

        for i, mask in enumerate(group_masks):
            if np.sum(mask) < 5:
                continue

            pred_group = pred[mask]

            try:
                kde_y = gaussian_kde(pred_group)
                density_y = kde_y(y_range)

                ax_right.fill_betweenx(y_range, 0, density_y,
                                      color=colors[i], alpha=0.4)
                ax_right.plot(density_y, y_range,
                             color=colors[i], linewidth=2)
            except Exception as e:
                print(f"  ⚠ KDE计算失败 ({group_names[i]}): {e}")

        ax_right.set_ylim(ax_main.get_ylim())
        ax_right.set_xlim(left=0)
        ax_right.axis('off')

        # 7. 主图美化
        ax_main.set_xlabel('Observation (°C)', fontsize=fontsize)
        ax_main.set_ylabel('Prediction (°C)', fontsize=fontsize)
        ax_main.tick_params(labelsize=fontsize)

        ax_main.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        ax_main.legend(loc='lower right', fontsize=fontsize, framealpha=0.95,
                      edgecolor='gray', fancybox=True)
        ax_main.set_aspect('equal', adjustable='box')

        # 隐藏边缘子图的刻度标签
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)

        # 8. 保存
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

    def plot_error_distribution(self, pred_step, save_path):
        """
        绘制误差分布直方图

        Args:
            pred_step: 预测步长索引
            save_path: 保存路径
        """
        # 计算误差
        pred = self.test_predict[:, :, pred_step].flatten()
        label = self.test_label[:, :, pred_step].flatten()
        errors = pred - label

        mean_error = np.mean(errors)
        std_error = np.std(errors)

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

        # 直方图
        n, bins, patches = ax.hist(errors, bins=50, color='steelblue',
                                    alpha=0.7, edgecolor='black', linewidth=0.5)

        # 均值线
        ax.axvline(mean_error, color='red', linestyle='-',
                  linewidth=2, label=f'Mean (Bias): {mean_error:+.4f}°C')

        # ±1σ线
        ax.axvline(mean_error + std_error, color='red', linestyle='--',
                  linewidth=1.5, label=f'+1σ: {mean_error+std_error:+.4f}°C', alpha=0.7)
        ax.axvline(mean_error - std_error, color='red', linestyle='--',
                  linewidth=1.5, label=f'-1σ: {mean_error-std_error:+.4f}°C', alpha=0.7)

        # 设置标签
        ax.set_xlabel('Error (Prediction - Ground Truth, °C)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(
            f'Step {pred_step+1} Prediction Error Distribution (Bias: {mean_error:+.4f}°C, Std: {std_error:.4f}°C)',
            fontsize=12, pad=15
        )

        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def plot_metrics_comparison(self, metrics, save_path):
        """
        绘制多指标对比图(4子图)

        Args:
            metrics: 指标字典
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

        metric_names = ['rmse_per_station', 'mae_per_station',
                       'r2_per_station', 'bias_per_station']
        titles = ['RMSE (°C)', 'MAE (°C)', 'R²', 'Bias (°C)']
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

        for idx, (ax, metric_name, title, color) in enumerate(
            zip(axes.flatten(), metric_names, titles, colors)
        ):
            values = metrics[metric_name]

            # 柱状图
            ax.bar(range(self.num_stations), values, color=color,
                  alpha=0.7, edgecolor='black', linewidth=0.5)

            # 平均线
            mean_val = np.mean(values)
            ax.axhline(mean_val, color='blue', linestyle='--',
                      linewidth=2, label=f'Mean: {mean_val:.4f}', alpha=0.7)

            # 设置标签
            ax.set_xlabel('Station ID', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(f'{title} Comparison', fontsize=11, pad=10)
            ax.set_xticks(range(0, self.num_stations, 2))
            ax.set_xticklabels([str(i) for i in range(0, self.num_stations, 2)],
                              fontsize=8)
            ax.grid(True, axis='y', alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)

        plt.suptitle('Metric Comparison Across Stations', fontsize=13, y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def save_metrics_csv(self, metrics, save_path):
        """
        保存指标CSV表格

        Args:
            metrics: 指标字典
            save_path: 保存路径
        """
        df = pd.DataFrame({
            'Station_ID': self.station_ids,
            'Longitude': self.lon,
            'Latitude': self.lat,
            'Height': self.height,
            'RMSE': metrics['rmse_per_station'],
            'MAE': metrics['mae_per_station'],
            'R2': metrics['r2_per_station'],
            'Bias': metrics['bias_per_station']
        })

        df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"  ✓ 指标CSV已保存: {save_path}")

    def plot_rmse_by_step(self, all_metrics, save_path):
        """
        绘制各步长RMSE对比

        Args:
            all_metrics: 字典,键为步长索引,值为指标字典
            save_path: 保存路径
        """
        steps = sorted(all_metrics.keys())

        # 整体RMSE
        overall_rmse = [all_metrics[step]['overall_rmse'] for step in steps]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

        # 主线:整体RMSE
        ax.plot([s+1 for s in steps], overall_rmse, 'o-',
               color='#e74c3c', linewidth=2.5, markersize=8,
               label='Overall RMSE', alpha=0.8)

        # 辅助线:部分代表性站点(可选)
        sample_stations_ids = [0, 7, 14, 21, 27]
        colors_sample = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        for station_id, color in zip(sample_stations_ids, colors_sample):
            if station_id < self.num_stations:
                rmse_by_step = [
                    all_metrics[step]['rmse_per_station'][station_id]
                    for step in steps
                ]
                ax.plot([s+1 for s in steps], rmse_by_step, 's--',
                       color=color, linewidth=1.5, markersize=5,
                       label=f'Station {station_id}', alpha=0.6)

        # 设置标签
        ax.set_xlabel('Prediction Step (Days)', fontsize=11)
        ax.set_ylabel('RMSE (°C)', fontsize=11)
        ax.set_title('RMSE Comparison Across Prediction Steps', fontsize=12, pad=15)
        ax.set_xticks([s+1 for s in steps])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def plot_mae_by_step(self, all_metrics, save_path):
        """
        绘制各步长MAE对比

        Args:
            all_metrics: 字典,键为步长索引,值为指标字典
            save_path: 保存路径
        """
        steps = sorted(all_metrics.keys())

        # 整体MAE
        overall_mae = [all_metrics[step]['overall_mae'] for step in steps]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)

        # 主线:整体MAE
        ax.plot([s+1 for s in steps], overall_mae, 'o-',
               color='#f39c12', linewidth=2.5, markersize=8,
               label='Overall MAE', alpha=0.8)

        # 设置标签
        ax.set_xlabel('Prediction Step (Days)', fontsize=11)
        ax.set_ylabel('MAE (°C)', fontsize=11)
        ax.set_title('MAE Comparison Across Prediction Steps', fontsize=12, pad=15)
        ax.set_xticks([s+1 for s in steps])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)

    def save_summary_metrics(self, all_metrics, save_path):
        """
        保存各步长汇总表格

        Args:
            all_metrics: 字典,键为步长索引,值为指标字典
            save_path: 保存路径
        """
        steps = sorted(all_metrics.keys())

        data = {
            'Step': [s+1 for s in steps],
            'Overall_RMSE': [all_metrics[s]['overall_rmse'] for s in steps],
            'Overall_MAE': [all_metrics[s]['overall_mae'] for s in steps],
            'Overall_R2': [all_metrics[s]['overall_r2'] for s in steps],
            'Overall_Bias': [all_metrics[s]['overall_bias'] for s in steps]
        }

        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"  ✓ 汇总表格已保存: {save_path}")

    def visualize_single_step(self, pred_step, output_dir):
        """
        为单个预测步长生成所有可视化

        Args:
            pred_step: 预测步长索引
            output_dir: 输出目录

        Returns:
            dict: 该步长的指标字典
        """
        step_dir = output_dir / f'step_{pred_step + 1}'
        step_dir.mkdir(exist_ok=True, parents=True)

        print(f"\n{'='*80}")
        print(f"生成第{pred_step+1}步预测的可视化")
        print(f"{'='*80}")

        # 1. 计算指标
        print("  计算评估指标...")
        metrics = self.calculate_metrics_for_step(pred_step)
        print(f"  ✓ 整体RMSE: {metrics['overall_rmse']:.4f}°C, "
              f"MAE: {metrics['overall_mae']:.4f}°C, "
              f"R²: {metrics['overall_r2']:.4f}")

        # 2. 生成所有图表
        print("  生成可视化图表...")

        # 时间序列图(28张)
        self.plot_all_timeseries(pred_step, step_dir)

        # 其他汇总图
        self.plot_rmse_spatial_map(metrics, step_dir / 'rmse_spatial_map.png')
        print("  ✓ RMSE空间分布图已生成")

        # 带地理底图的版本（如果配置启用）
        if USE_BASEMAP and BASEMAP_AVAILABLE:
            print("  生成带地理底图的RMSE空间分布图...")
            self.plot_rmse_spatial_map_with_basemap(
                metrics,
                step_dir / 'rmse_spatial_map_with_ids.png',
                annotation_type='ids'
            )
            self.plot_rmse_spatial_map_with_basemap(
                metrics,
                step_dir / 'rmse_spatial_map_with_values.png',
                annotation_type='values'
            )

        self.plot_rmse_barplot(metrics, step_dir / 'rmse_barplot.png')
        print("  ✓ RMSE柱状图已生成")

        self.plot_pred_vs_true_scatter(pred_step, step_dir / 'pred_vs_true.png')
        print("  ✓ 预测vs真实散点图已生成")

        self.plot_pred_vs_true_marginal(pred_step, step_dir / 'pred_vs_true_marginal.png')
        print("  ✓ 边缘密度分布图已生成")

        self.plot_error_distribution(pred_step, step_dir / 'error_distribution.png')
        print("  ✓ 误差分布图已生成")

        self.plot_metrics_comparison(metrics, step_dir / 'metrics_comparison.png')
        print("  ✓ 多指标对比图已生成")

        # 3. 保存数据
        self.save_metrics_csv(metrics, step_dir / 'station_metrics.csv')

        if SAVE_INTERMEDIATE_DATA:
            self.save_plot_data(step_dir, pred_step, metrics)

        return metrics

    def generate_all(self, output_dir=None):
        """
        生成所有可视化

        Args:
            output_dir: 输出目录(None表示使用默认)
        """
        # 1. 确定输出目录
        if output_dir is None:
            output_dir = self.checkpoint_dir / 'visualizations'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True, parents=True)

        # 2. 确定要可视化的步长
        if PRED_STEPS == 'all':
            pred_steps = range(self.pred_len)
        else:
            pred_steps = PRED_STEPS

        print(f"\n将生成{len(pred_steps)}个预测步长的可视化")

        # 3. 为每个步长生成可视化
        all_metrics = {}
        for step in pred_steps:
            metrics = self.visualize_single_step(step, output_dir)
            all_metrics[step] = metrics

        # 4. 生成多步长汇总(如果有多个步长)
        if len(all_metrics) > 1:
            print(f"\n{'='*80}")
            print("生成多步长汇总可视化")
            print(f"{'='*80}")

            summary_dir = output_dir / 'summary'
            summary_dir.mkdir(exist_ok=True)

            self.plot_rmse_by_step(all_metrics, summary_dir / 'rmse_by_step.png')
            print("  ✓ RMSE步长对比图已生成")

            self.plot_mae_by_step(all_metrics, summary_dir / 'mae_by_step.png')
            print("  ✓ MAE步长对比图已生成")

            self.save_summary_metrics(all_metrics, summary_dir / 'metrics_by_step.csv')

        # 5. 打印完成信息
        print(f"\n{'='*80}")
        print("✨ 可视化完成!")
        print(f"{'='*80}")
        print(f"📁 结果保存在: {output_dir}")
        print(f"\n生成内容:")

        for step in pred_steps:
            step_dir = output_dir / f'step_{step + 1}'
            num_ts_plots = (self.num_stations if PLOT_ALL_STATIONS
                           else len(SAMPLE_STATIONS))
            print(f"  - 第{step+1}步预测: {step_dir}")
            print(f"    · {num_ts_plots}张时间序列图")
            print(f"    · 6张汇总分析图")
            print(f"    · 1个指标CSV表格")
            if SAVE_INTERMEDIATE_DATA:
                print(f"    · 1个绘图数据文件(NPZ)")

        if len(all_metrics) > 1:
            print(f"  - 多步长汇总: {output_dir / 'summary'}")
            print(f"    · 2张步长对比图")
            print(f"    · 1个汇总CSV表格")


def visualize_checkpoint(checkpoint_dir,
                         output_dir='auto',
                         pred_steps='all',
                         plot_all_stations=True,
                         time_sample_rate=1,
                         save_intermediate_data=True,
                         dpi=300,
                         use_basemap=True,
                         add_scalebar=True,
                         add_north_arrow=True,
                         use_chinese=True,
                         silent=False):
    """
    可视化训练结果（函数式接口）

    此函数封装了完整的可视化流程，可在训练完成后自动调用，
    无需手动修改配置文件。

    Args:
        checkpoint_dir (str|Path): checkpoint目录路径
            例如: 'checkpoints/GAT_LSTM_Attention_20251123_021443'
        output_dir (str|Path): 输出目录
            - 'auto': 在checkpoint下创建visualizations/子目录（推荐）
            - 自定义路径: 指定任意输出路径
        pred_steps (str|list): 预测步长
            - 'all': 可视化全部步长（默认）
            - [0, 1, 2]: 仅可视化指定步长（索引从0开始）
        plot_all_stations (bool): 是否绘制全部28个站点的时间序列图
            - True: 绘制全部站点（默认，生成28张图）
            - False: 仅绘制SAMPLE_STATIONS中的站点
        time_sample_rate (int): 时间序列采样率
            - 1: 显示全部时间点（默认）
            - 5: 每5个点显示1个（用于长时间序列）
        save_intermediate_data (bool): 是否保存中间数据NPZ
            - True: 保存绘图数据（默认，方便后续复现）
            - False: 仅保存图片
        dpi (int): 图表分辨率
            - 300: 高分辨率（默认，适合论文）
            - 150: 中等分辨率（快速预览）
            - 100: 低分辨率（加快生成）
        use_basemap (bool): 是否使用地理底图（需要cartopy库）
            - True: 使用底图（默认，更专业）
            - False: 仅显示站点（更快）
        add_scalebar (bool): 是否添加比例尺（默认True）
        add_north_arrow (bool): 是否添加指北针（默认True）
        use_chinese (bool): 是否使用中文字体（默认True）
        silent (bool): 是否静默模式（减少输出，默认False）

    Returns:
        bool: 是否成功生成可视化
            - True: 所有可视化生成成功
            - False: 发生错误

    Raises:
        ValueError: checkpoint目录不存在
        FileNotFoundError: 缺少必需文件

    Example:
        >>> # 基本用法（使用默认配置）
        >>> from visualize_results import visualize_checkpoint
        >>> visualize_checkpoint('checkpoints/GAT_LSTM_20251123_021443')

        >>> # 快速预览（降低分辨率，跳过底图）
        >>> visualize_checkpoint(
        ...     'checkpoints/GAT_LSTM_20251123_021443',
        ...     dpi=150,
        ...     use_basemap=False
        ... )

        >>> # 仅可视化第1步预测
        >>> visualize_checkpoint(
        ...     'checkpoints/GAT_LSTM_20251123_021443',
        ...     pred_steps=[0]
        ... )

        >>> # 静默模式（用于自动化脚本）
        >>> success = visualize_checkpoint(
        ...     'checkpoints/GAT_LSTM_20251123_021443',
        ...     silent=True
        ... )
        >>> if success:
        ...     print("可视化生成成功")
    """
    # 全局配置（使用局部变量避免污染全局状态）
    global OUTPUT_DIR, PRED_STEPS, PLOT_ALL_STATIONS, TIME_SAMPLE_RATE
    global SAVE_INTERMEDIATE_DATA, DPI, USE_CHINESE
    global USE_BASEMAP, ADD_SCALEBAR, ADD_NORTH_ARROW

    # 保存原始配置（用于恢复）
    original_config = {
        'OUTPUT_DIR': OUTPUT_DIR,
        'PRED_STEPS': PRED_STEPS,
        'PLOT_ALL_STATIONS': PLOT_ALL_STATIONS,
        'TIME_SAMPLE_RATE': TIME_SAMPLE_RATE,
        'SAVE_INTERMEDIATE_DATA': SAVE_INTERMEDIATE_DATA,
        'DPI': DPI,
        'USE_CHINESE': USE_CHINESE,
        'USE_BASEMAP': USE_BASEMAP,
        'ADD_SCALEBAR': ADD_SCALEBAR,
        'ADD_NORTH_ARROW': ADD_NORTH_ARROW
    }

    try:
        # ========== 1. 参数验证 ==========
        checkpoint_path = Path(checkpoint_dir)

        # 检查checkpoint目录是否存在
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint目录不存在: {checkpoint_dir}")

        # 检查必需文件
        required_files = ['test_predict.npy', 'test_label.npy', 'test_time.npy']
        missing_files = [f for f in required_files
                         if not (checkpoint_path / f).exists()]

        if missing_files:
            raise FileNotFoundError(
                f"缺少必需文件: {', '.join(missing_files)}\n"
                f"请确保训练脚本已运行完成并保存了测试集结果"
            )

        # 验证pred_steps参数
        if pred_steps != 'all' and not isinstance(pred_steps, (list, tuple)):
            raise TypeError("pred_steps必须是'all'或列表/元组")

        # 验证DPI范围
        if not (50 <= dpi <= 600):
            if not silent:
                print(f"⚠ DPI值{dpi}不常见，建议范围[50, 600]")

        # ========== 2. 应用配置 ==========
        OUTPUT_DIR = output_dir
        PRED_STEPS = pred_steps
        PLOT_ALL_STATIONS = plot_all_stations
        TIME_SAMPLE_RATE = time_sample_rate
        SAVE_INTERMEDIATE_DATA = save_intermediate_data
        DPI = dpi
        USE_CHINESE = use_chinese
        USE_BASEMAP = use_basemap
        ADD_SCALEBAR = add_scalebar
        ADD_NORTH_ARROW = add_north_arrow

        # 配置字体
        setup_font()

        # ========== 3. 打印配置信息 ==========
        if not silent:
            print("=" * 80)
            print("训练结果可视化工具（函数式调用）")
            print("=" * 80)
            print(f"\n当前配置:")
            print(f"  Checkpoint: {checkpoint_dir}")
            print(f"  输出目录: {output_dir}")
            print(f"  预测步长: {pred_steps}")
            print(f"  绘制站点: {'全部28个' if plot_all_stations else '部分'}")
            print(f"  时间采样: 每{time_sample_rate}个点")
            print(f"  保存数据: {'是' if save_intermediate_data else '否'}")
            print(f"  图表DPI: {dpi}")
            print(f"  使用底图: {'是' if use_basemap else '否'}")

        # ========== 4. 创建可视化器 ==========
        if not silent:
            print(f"\n{'='*80}")
            print("加载数据")
            print(f"{'='*80}")

        visualizer = ResultVisualizer(checkpoint_dir)

        # ========== 5. 生成所有图表 ==========
        output_path = (Path(checkpoint_dir) / 'visualizations'
                       if output_dir == 'auto' else Path(output_dir))

        visualizer.generate_all(output_path)

        # ========== 6. 完成 ==========
        if not silent:
            print(f"\n{'='*80}")
            print("✅ 可视化生成成功!")
            print(f"{'='*80}")
            print(f"输出路径: {output_path}")

        return True

    except Exception as e:
        if not silent:
            print(f"\n❌ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n请检查:")
            print(f"  1. checkpoint_dir 路径是否正确")
            print(f"  2. 是否存在必需文件: {', '.join(required_files)}")
            print(f"  3. 依赖库是否完整安装")
        return False

    finally:
        # ========== 7. 恢复原始配置 ==========
        OUTPUT_DIR = original_config['OUTPUT_DIR']
        PRED_STEPS = original_config['PRED_STEPS']
        PLOT_ALL_STATIONS = original_config['PLOT_ALL_STATIONS']
        TIME_SAMPLE_RATE = original_config['TIME_SAMPLE_RATE']
        SAVE_INTERMEDIATE_DATA = original_config['SAVE_INTERMEDIATE_DATA']
        DPI = original_config['DPI']
        USE_CHINESE = original_config['USE_CHINESE']
        USE_BASEMAP = original_config['USE_BASEMAP']
        ADD_SCALEBAR = original_config['ADD_SCALEBAR']
        ADD_NORTH_ARROW = original_config['ADD_NORTH_ARROW']


def main():
    """主函数"""
    print("=" * 80)
    print("训练结果可视化工具")
    print("=" * 80)

    print(f"\n当前配置:")
    print(f"  Checkpoint: {CHECKPOINT_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  预测步长: {PRED_STEPS}")
    print(f"  绘制站点: {'全部28个' if PLOT_ALL_STATIONS else f'{len(SAMPLE_STATIONS)}个'}")
    print(f"  时间采样: 每{TIME_SAMPLE_RATE}个点")
    print(f"  保存数据: {'是' if SAVE_INTERMEDIATE_DATA else '否'}")
    print(f"  图表DPI: {DPI}")

    # 配置字体
    setup_font()

    # 创建可视化器
    print(f"\n{'='*80}")
    print("加载数据")
    print(f"{'='*80}")

    try:
        visualizer = ResultVisualizer(CHECKPOINT_DIR)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print(f"\n请检查:")
        print(f"  1. CHECKPOINT_DIR 路径是否正确")
        print(f"  2. 是否存在 test_predict.npy, test_label.npy, test_time.npy")
        return

    # 生成所有图表
    output_path = (Path(CHECKPOINT_DIR) / 'visualizations'
                   if OUTPUT_DIR == 'auto' else Path(OUTPUT_DIR))

    visualizer.generate_all(output_path)

    print(f"\n{'='*80}")
    print("✅ 全部完成!")
    print(f"{'='*80}")
    print(f"\n提示: 如需分析其他模型,请修改文件顶部的 CHECKPOINT_DIR 配置")


if __name__ == "__main__":
    main()
