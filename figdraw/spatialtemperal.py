import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ============ 配置区域 ============
# Checkpoint目录路径
script_dir = Path(__file__).parent.parent
CHECKPOINT_DIR = str(script_dir / 'myGNN' / 'checkpoints' / 'LSTM_20251222_144549')
PRED_STEP = 0  # 预测步长索引（0-4对应第1-5天）
FILL_MISSING = True  # 是否填充缺失日期（True=填充NaN以显示完整年份）
SELECTED_STATIONS = [24,25]  # 指定站点列表，如[0, 5, 10, 15]；None表示全部28个站点
# ==================================

# ---------------------------------------------------------
# 1. 数据加载与处理
# ---------------------------------------------------------
def load_test_data_with_dates(checkpoint_dir: str, pred_step: int = 0,
                              fill_missing: bool = False,
                              selected_stations: list = None) -> pd.DataFrame:
    """
    从checkpoint加载真实测试集数据并计算每天的RMSE

    Args:
        checkpoint_dir: checkpoint目录路径
        pred_step: 预测步长索引（0表示第1天预测）
        fill_missing: 是否填充缺失日期以显示完整年份
        selected_stations: 指定站点索引列表（如[0, 5, 10]），None表示全部站点

    Returns:
        DataFrame with columns: ['datetime', 'Month', 'Day', 'RMSE']
    """
    ckpt_path = Path(checkpoint_dir)

    # 加载数据
    predictions = np.load(ckpt_path / 'test_predict.npy')  # [num_samples, num_stations, pred_len]
    labels = np.load(ckpt_path / 'test_label.npy')          # [num_samples, num_stations, pred_len]

    print(f"[INFO] 加载测试集数据:")
    print(f"  - Checkpoint: {checkpoint_dir}")
    print(f"  - 数据形状: {predictions.shape}")
    print(f"  - 预测步长: {pred_step} (第{pred_step+1}天)")

    # 提取指定预测步长的数据
    pred_step_data = predictions[:, :, pred_step]  # [num_samples, num_stations]
    label_step_data = labels[:, :, pred_step]      # [num_samples, num_stations]

    # 过滤指定站点
    if selected_stations is not None:
        pred_step_data = pred_step_data[:, selected_stations]
        label_step_data = label_step_data[:, selected_stations]
        print(f"  - 选定站点: {selected_stations} (共{len(selected_stations)}个)")
    else:
        print(f"  - 使用全部28个站点")

    pd.DataFrame({'pred_step_data':np.mean(pred_step_data,axis=1),'label_step_data':np.mean(label_step_data,axis=1)}).to_csv('data.csv')
    # 计算每个样本的RMSE（选定站点的平均）
    rmse_per_sample = np.sqrt(np.mean((pred_step_data - label_step_data)**2, axis=1))
    # 构造日期（测试集是2017年，共346个有效样本）
    # 有效样本对应DOY 15-360（1月15日至12月26日）
    test_start_date = pd.Timestamp('2017-01-15')  # DOY 15
    dates = pd.date_range(start=test_start_date, periods=len(rmse_per_sample), freq='D')

    # 创建DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'Month': dates.month,
        'Day': dates.day,
        'RMSE': rmse_per_sample
    })
    print(f"  - 有效样本数: {len(df)}天")
    print(f"  - 日期范围: {df['datetime'].min().date()} 至 {df['datetime'].max().date()}")
    print(f"  - RMSE范围: {df['RMSE'].min():.4f} - {df['RMSE'].max():.4f}°C")

    # 是否填充缺失日期
    if fill_missing:
        # 创建完整的2017年日期范围
        full_dates = pd.date_range(start='2017-01-01', end='2017-12-31', freq='D')
        df_full = pd.DataFrame({
            'datetime': full_dates,
            'Month': full_dates.month,
            'Day': full_dates.day
        })
        # 合并数据，缺失的日期会自动填充NaN
        df_full = df_full.merge(df[['datetime', 'RMSE']], on='datetime', how='left')
        print(f"  - 填充后: 365天（包含{df_full['RMSE'].isna().sum()}个NaN值）")
        
        return df_full
    else:
        return df


# 加载真实数据
data = load_test_data_with_dates(CHECKPOINT_DIR, pred_step=PRED_STEP,
                                  fill_missing=FILL_MISSING,
                                  selected_stations=SELECTED_STATIONS)
heatmap_data = data.pivot_table(index='Month', columns='Day', values='RMSE', aggfunc='mean')
daily_mean = data.groupby('Day')['RMSE'].mean()
monthly_mean = data.groupby('Month')['RMSE'].mean()

# ---------------------------------------------------------
# 2. 辅助函数：设置完整边框
# ---------------------------------------------------------
def set_full_border(ax, color='#333333', linewidth=1.0):
    """为坐标轴添加完整的四周边框，并确保可见"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(linewidth)

# ---------------------------------------------------------
# 3. 绘图代码 (修改版)
# ---------------------------------------------------------
def plot_daily_heatmap_styled(heatmap_data, daily_mean, monthly_mean, data_stats=None):
    """
    绘制日均RMSE热力图

    Args:
        heatmap_data: 热力图数据
        daily_mean: 每日平均RMSE
        monthly_mean: 每月平均RMSE
        data_stats: 数据统计信息字典（可选）
    """
    # 修改1: 使用 "white" 风格，基础不带网格，方便我们手动添加
    sns.set_style("white")

    # 动态调整colorbar范围
    if data_stats is not None and 'vmin' in data_stats and 'vmax' in data_stats:
        vmin = data_stats['vmin']
        vmax = data_stats['vmax']
    else:
        vmin = 0.5
        vmax = 2.0

    fig = plt.figure(figsize=(14, 6), dpi=120)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 0.1, 0.03], height_ratios=[0.2, 1],
                           wspace=0.05, hspace=0.05)

    # 添加图表标题
    if data_stats is not None:
        fig.suptitle(data_stats.get('title', 'Daily RMSE Heatmap'),
                     fontsize=14, fontweight='bold', y=0.98)

    # --- A. 上方折线图 ---
    ax_top = plt.subplot(gs[0, 0])
    ax_top.plot(daily_mean.index, daily_mean.values, marker='o', markersize=3, color='#2b8cbe')
    ax_top.set_xlim(0.5, 31.5)
    ax_top.set_ylabel('RMSE', fontsize=9)
    ax_top.set_xticks([])

    # 手动添加内部虚线网格
    ax_top.grid(True, linestyle='--', alpha=0.7, color='#cccccc')

    # 修改2: 设置完整边框，移除了原来的 sns.despine()
    set_full_border(ax_top)

    # --- B. 主热力图 ---
    ax_main = plt.subplot(gs[1, 0])
    # 修改3: 设置 linewidths=0，确保热力图格子间无分割线
    sns.heatmap(heatmap_data, ax=ax_main, cmap='coolwarm', cbar=False,
                vmin=vmin, vmax=vmax, linewidths=0)

    ax_main.set_xlabel('Day of Month', fontsize=11)
    ax_main.set_ylabel('Month', fontsize=11)
    ax_main.tick_params(axis='y', rotation=0)
    ax_main.tick_params(axis='x', rotation=0)

    # 修改4: 显式关闭热力图区域的网格（双重保险）
    ax_main.grid(False)
    # 为热力图也添加一个完整的黑色边框，使整体看起来更整洁
    set_full_border(ax_main)

    # --- C. 右侧折线图 ---
    ax_right = plt.subplot(gs[1, 1])
    y_pos = np.arange(len(monthly_mean)) + 0.5
    ax_right.plot(monthly_mean.values, y_pos, marker='o', markersize=4, color='#2b8cbe')
    ax_right.set_ylim(len(monthly_mean), 0)
    ax_right.set_xlabel('RMSE', fontsize=9)
    ax_right.set_yticks([])
    ax_right.xaxis.tick_bottom()

    # 手动添加内部虚线网格 (仅X轴方向)
    ax_right.grid(True, linestyle='--', axis='x', alpha=0.7, color='#cccccc')

    # 修改5: 设置完整边框，移除了原来的 sns.despine()
    set_full_border(ax_right)

    # --- D. 颜色条 ---
    ax_cbar = plt.subplot(gs[1, 2])
    mappable = ax_main.collections[0]
    plt.colorbar(mappable, cax=ax_cbar, label='RMSE (°C)')
    # 为了风格统一，也给颜色条加上边框
    set_full_border(ax_cbar)

    plt.tight_layout()

    # 保存图表
    output_path = Path(__file__).parent / 'result' / 'spatiotemporal_rmse.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 图表已保存至: {output_path}")

    plt.show()


# 准备数据统计信息
title_suffix = f'Test Set Daily RMSE Heatmap (2017, Step {PRED_STEP+1})'
if SELECTED_STATIONS is not None:
    title_suffix += f' - Stations: {SELECTED_STATIONS}'
else:
    title_suffix += ' - All 28 Stations'

data_stats = {
    'vmin': data['RMSE'].min() * 0.9,
    'vmax': data['RMSE'].max() * 1.1,
    'title': title_suffix
}

# 执行绘图
plot_daily_heatmap_styled(heatmap_data, daily_mean, monthly_mean, data_stats)