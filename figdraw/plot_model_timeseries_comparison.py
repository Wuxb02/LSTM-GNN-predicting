"""
双模型测试集时序对比图（指定站点，预测第1天）

绘制四条折线：
  1. 真实气温（Ground Truth）
  2. GAT_SeparateEncoder_MSE 模型预测
  3. GAT_SeparateEncoder_WEIGHTED 模型预测
  4. 90 分位数阈值线

用法：直接修改下方 ── 配置区域 ── 中的参数后运行。
"""

import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# ── 配置区域 ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

CKPT_MSE = ROOT_DIR / 'myGNN' / 'checkpoints' / 'GAT_SeparateEncoder_MSE'
CKPT_WGT = ROOT_DIR / 'myGNN' / 'checkpoints' / 'GAT_SeparateEncoder_20260420_125517'
STATION_INFO_PATH = ROOT_DIR / 'data' / 'station_info.npy'

# 指定要绘制的站点索引（0-based，共28个站点）
STATION_IDX = 1

# 图表输出路径
OUTPUT_DIR = SCRIPT_DIR / 'result'
OUTPUT_NAME = f'timeseries_comparison_station{STATION_IDX:02d}.png'

# 字体大小
FONTSIZE = 14
# ────────────────────────────────────────────────────────────────────────────

# 2020-01-01 对应全局天数索引（测试集年份）
_YEAR_2020_START = 3652


def load_checkpoint(ckpt_dir: Path) -> dict:
    """加载 checkpoint 目录中的测试集预测结果与阈值表。"""
    data = {
        'predict':   np.load(ckpt_dir / 'test_predict.npy'),   # [N, 28, 5]
        'label':     np.load(ckpt_dir / 'test_label.npy'),     # [N, 28, 5]
        'time':      np.load(ckpt_dir / 'test_time.npy'),      # [N]
        'threshold': np.load(ckpt_dir / 'threshold_map.npy'),  # [365, 28]
    }
    return data


def global_idx_to_date(global_idx: int) -> datetime.date:
    """将全局天数索引（2010-01-01 = 0）转换为 date 对象。"""
    base = datetime.date(2010, 1, 1)
    return base + datetime.timedelta(days=int(global_idx))


def extract_station_series(data: dict, station_idx: int, pred_step: int = 0):
    """
    提取指定站点、指定预测步长的时序数据。

    Returns
    -------
    dates       : list[datetime.date]  预测目标日期
    pred_vals   : np.ndarray           预测气温 [N]
    label_vals  : np.ndarray           真实气温 [N]
    thresh_vals : np.ndarray           90分位数阈值 [N]
    """
    time_arr = data['time']                                  # [N]
    pred_vals   = data['predict'][:, station_idx, pred_step]  # [N]
    label_vals  = data['label'][:, station_idx, pred_step]    # [N]
    threshold   = data['threshold']                            # [365, 28]

    dates = [global_idx_to_date(t) for t in time_arr]

    # threshold_map 以 0-based DOY 为索引（非闰年，doy=1 → index 0）
    # 2020 为闰年，offset = global_idx - 3652 即为 0-based DOY index
    doy_indices = (time_arr - _YEAR_2020_START).astype(int)
    # 截断到 [0, 364] 以防边界越界
    doy_indices = np.clip(doy_indices, 0, 364)
    thresh_vals = threshold[doy_indices, station_idx]

    return dates, pred_vals, label_vals, thresh_vals


def compute_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """计算 RMSE、MAE、Bias、R²。"""
    err = pred - label
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((label - label.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    return {'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'R2': r2}


def plot_comparison(
    dates,
    label_vals,
    pred_mse,
    pred_wgt,
    thresh_vals,
    station_idx: int,
    station_id: int,
    metrics_mse: dict,
    metrics_wgt: dict,
    save_path: Path,
):
    """绘制四线时序对比图并保存。"""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)

    date_arr = np.array(dates, dtype='datetime64[D]')

    # ── 极端高温背景着色 ────────────────────────────────────────────────────
    extreme_mask = label_vals >= thresh_vals
    in_event = False
    event_start = None
    for i, flag in enumerate(extreme_mask):
        if flag and not in_event:
            in_event = True
            event_start = dates[i]
        elif not flag and in_event:
            ax.axvspan(event_start, dates[i - 1],
                       color='#ffe0b2', alpha=0.45, linewidth=0, zorder=1)
            in_event = False
    if in_event:
        ax.axvspan(event_start, dates[-1],
                   color='#ffe0b2', alpha=0.45, linewidth=0, zorder=1)

    # ── 四条折线 ────────────────────────────────────────────────────────────
    ax.plot(
        dates, label_vals,
        color='#1a3a5c', linewidth=1.4, alpha=0.9,
        label='Ground Truth', zorder=5,
    )
    ax.plot(
        dates, pred_mse,
        color='#e07b39', linewidth=1.4, linestyle='--', alpha=0.9,
        label=(
            f'MSE Model  '
            f'(RMSE={metrics_mse["RMSE"]:.2f}'
        ),
        zorder=4,
    )
    ax.plot(
        dates, pred_wgt,
        color='#2e8b57', linewidth=1.4, linestyle='-.', alpha=0.9,
        label=(
            f'Weighted Model  '
            f'(RMSE={metrics_wgt["RMSE"]:.2f})'
        ),
        zorder=4,
    )
    ax.plot(
        dates, thresh_vals,
        color='#c0392b', linewidth=1.2, linestyle=':', alpha=0.85,
        label='90th Percentile Threshold', zorder=3,
    )

    # ── 坐标轴格式 ──────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.set_xlabel('Month', fontsize=FONTSIZE)
    ax.set_ylabel('Temperature (°C)', fontsize=FONTSIZE)

    # ax.set_title(
    #     f'Station {station_idx} (ID {station_id}) — Test Set Step-1 Prediction',
    #     fontsize=FONTSIZE + 1, fontweight='bold',
    # )

    # ── 图例与网格 ──────────────────────────────────────────────────────────
    ax.legend(fontsize=FONTSIZE - 1, loc='upper left',
              frameon=True, framealpha=0.85, edgecolor='#cccccc')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4, color='#888888')

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'[OK] 图表已保存: {save_path}')
    plt.close()


def main():
    # ── 加载站点信息 ────────────────────────────────────────────────────────
    station_info = np.load(STATION_INFO_PATH, allow_pickle=True)  # [28, 4]
    station_id = int(station_info[STATION_IDX, 0])
    print(f'[INFO] 站点索引: {STATION_IDX}, 站点ID: {station_id}')

    # ── 加载两个模型的数据 ──────────────────────────────────────────────────
    print(f'[INFO] 加载 MSE checkpoint: {CKPT_MSE}')
    data_mse = load_checkpoint(CKPT_MSE)

    print(f'[INFO] 加载 WEIGHTED checkpoint: {CKPT_WGT}')
    data_wgt = load_checkpoint(CKPT_WGT)

    # ── 提取时序 ────────────────────────────────────────────────────────────
    dates, pred_mse, label_mse, thresh_mse = extract_station_series(
        data_mse, STATION_IDX
    )
    _, pred_wgt, label_wgt, thresh_wgt = extract_station_series(
        data_wgt, STATION_IDX
    )

    # 两个模型的 label 应相同；以 MSE 的为准，阈值也取 MSE checkpoint 中的
    label_vals  = label_mse
    thresh_vals = thresh_mse

    # ── 计算指标 ────────────────────────────────────────────────────────────
    metrics_mse = compute_metrics(pred_mse, label_vals)
    metrics_wgt = compute_metrics(pred_wgt, label_vals)

    print(f'\n[MSE Model]      RMSE={metrics_mse["RMSE"]:.4f}  '
          f'MAE={metrics_mse["MAE"]:.4f}  '
          f'Bias={metrics_mse["Bias"]:.4f}  R2={metrics_mse["R2"]:.4f}')
    print(f'[Weighted Model] RMSE={metrics_wgt["RMSE"]:.4f}  '
          f'MAE={metrics_wgt["MAE"]:.4f}  '
          f'Bias={metrics_wgt["Bias"]:.4f}  R2={metrics_wgt["R2"]:.4f}')

    extreme_days = int(np.sum(label_vals >= thresh_vals))
    print(f'\n[INFO] 极端高温天数（真实值≥90分位线）: {extreme_days} 天')

    # ── 绘图 ────────────────────────────────────────────────────────────────
    save_path = OUTPUT_DIR / OUTPUT_NAME
    plot_comparison(
        dates=dates,
        label_vals=label_vals,
        pred_mse=pred_mse,
        pred_wgt=pred_wgt,
        thresh_vals=thresh_vals,
        station_idx=STATION_IDX,
        station_id=station_id,
        metrics_mse=metrics_mse,
        metrics_wgt=metrics_wgt,
        save_path=save_path,
    )


if __name__ == '__main__':
    main()
