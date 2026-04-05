"""
调优结果可视化模块

生成 Optuna 标准可视化图表：
- 优化历史曲线
- 参数重要性排序
- 并行坐标图
- 关键参数切片图
- 二维等高线图
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import optuna
    import optuna.visualization as vis
except ImportError:
    raise ImportError("请安装 optuna: pip install optuna")


def visualize_results(
    study: optuna.Study,
    output_dir: str = "tuning_results/visualizations",
) -> Dict[str, str]:
    """
    生成所有可视化图表并保存到 output_dir。

    Args:
        study: 已完成优化的 Optuna Study 对象
        output_dir: 图表保存目录

    Returns:
        图表路径字典 {名称: 文件路径}
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    # 1. 优化历史
    fig = vis.plot_optimization_history(study)
    path = str(out / "optimization_history.png")
    fig.write_image(path)
    saved["optimization_history"] = path

    # 2. 参数重要性
    try:
        fig = vis.plot_param_importances(study)
        path = str(out / "param_importances.png")
        fig.write_image(path)
        saved["param_importances"] = path
    except Exception:
        pass  # 某些情况下重要性计算可能失败

    # 3. 并行坐标图
    try:
        fig = vis.plot_parallel_coordinate(study)
        path = str(out / "parallel_coordinate.png")
        fig.write_image(path)
        saved["parallel_coordinate"] = path
    except Exception:
        pass

    # 4. 关键参数切片图
    for param_name in ["lr", "hid_dim", "c_under", "c_over", "GAT_layer", "heads"]:
        try:
            fig = vis.plot_slice(study, params=[param_name])
            path = str(out / f"slice_{param_name}.png")
            fig.write_image(path)
            saved[f"slice_{param_name}"] = path
        except Exception:
            pass

    # 5. 二维等高线图（hid_dim vs lr）
    try:
        fig = vis.plot_contour(study, params=["hid_dim", "lr"])
        path = str(out / "contour_hid_dim_vs_lr.png")
        fig.write_image(path)
        saved["contour_hid_dim_vs_lr"] = path
    except Exception:
        pass

    # 6. 损失参数等高线图（c_under vs c_over）
    try:
        fig = vis.plot_contour(study, params=["c_under", "c_over"])
        path = str(out / "contour_c_under_vs_c_over.png")
        fig.write_image(path)
        saved["contour_c_under_vs_c_over"] = path
    except Exception:
        pass

    print(f"✓ 可视化图表已保存到: {out}")
    for name, path in saved.items():
        print(f"  - {name}: {path}")

    return saved


def save_best_config(
    study: optuna.Study,
    output_dir: str = "tuning_results",
) -> Dict[str, Any]:
    """
    保存最佳配置为 JSON 文件。

    Args:
        study: 已完成优化的 Optuna Study 对象
        output_dir: 输出目录

    Returns:
        最佳配置字典
    """
    best = study.best_trial
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 构建最佳配置
    best_config = {
        "trial_number": best.number,
        "val_rmse": best.value,
        "params": best.params,
        "user_attrs": {k: v for k, v in best.user_attrs.items()},
    }

    # 保存 JSON
    json_path = out / "best_config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"✓ 最佳配置已保存: {json_path}")

    # 保存 Top-10 配置
    df = study.trials_dataframe()
    df = df.sort_values("value")
    top10 = df.head(10)

    top10_configs = []
    for _, row in top10.iterrows():
        config_entry = {
            "trial_number": int(row["number"]),
            "val_rmse": float(row["value"]),
            "params": {},
        }
        for col in df.columns:
            if col.startswith("params_"):
                param_name = col[len("params_") :]
                val = row[col]
                # 处理 numpy 类型
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, (np.bool_,)):
                    val = bool(val)
                config_entry["params"][param_name] = val
        top10_configs.append(config_entry)

    top10_path = out / "top10_configs.json"
    with open(top10_path, "w", encoding="utf-8") as f:
        json.dump(top10_configs, f, indent=2, ensure_ascii=False)
    print(f"✓ Top-10 配置已保存: {top10_path}")

    # 保存所有 trials 为 CSV
    csv_path = out / "trials_dataframe.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ 所有试验记录已保存: {csv_path}")

    return best_config


def print_best_summary(study: optuna.Study):
    """在终端打印最佳试验摘要。"""
    best = study.best_trial
    print("\n" + "=" * 70)
    print("最佳超参数配置")
    print("=" * 70)
    print(f"Trial 编号: {best.number}")
    print(f"验证集 RMSE: {best.value:.4f} °C")

    print("\n超参数:")
    for name, value in best.params.items():
        print(f"  {name}: {value}")

    print("\n附加指标:")
    for attr_name, attr_value in best.user_attrs.items():
        print(f"  {attr_name}: {attr_value}")
    print("=" * 70)
