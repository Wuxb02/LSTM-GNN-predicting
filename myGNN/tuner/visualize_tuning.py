"""
调优结果可视化模块

生成超参数调优的可视化图表，包括：
- 优化历史曲线
- 参数重要性分析
- 平行坐标图
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TuningVisualizer:
    """
    调优结果可视化器

    使用 Optuna 内置可视化功能生成图表
    """

    def __init__(self, study, save_dir: Path):
        """
        初始化可视化器

        Args:
            study: Optuna Study 对象
            save_dir: 图表保存目录
        """
        self.study = study
        self.save_dir = Path(save_dir) / 'visualizations'
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_optimization_history(self) -> Optional[str]:
        """
        绘制优化历史曲线

        显示验证 RMSE 随试验次数的变化
        """
        try:
            import optuna.visualization as vis

            fig = vis.plot_optimization_history(self.study)
            save_path = self.save_dir / 'optimization_history.png'
            fig.write_image(str(save_path), scale=2)
            return str(save_path)
        except Exception as e:
            print(f"绘制优化历史失败: {e}")
            return self._plot_optimization_history_fallback()

    def _plot_optimization_history_fallback(self) -> Optional[str]:
        """
        使用 matplotlib 绘制优化历史（备用方案）
        """
        try:
            trials = self.study.trials
            completed = [t for t in trials if t.state.name == 'COMPLETE']

            if not completed:
                return None

            trial_numbers = [t.number for t in completed]
            values = [t.value for t in completed]

            # 计算累计最优
            best_so_far = []
            current_best = float('inf')
            for v in values:
                current_best = min(current_best, v)
                best_so_far.append(current_best)

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.scatter(trial_numbers, values, alpha=0.6, c='blue', label='试验结果')
            ax.plot(trial_numbers, best_so_far, 'r-', linewidth=2,
                    label='当前最优', alpha=0.8)

            ax.set_xlabel('试验编号', fontsize=12)
            ax.set_ylabel('验证 RMSE (°C)', fontsize=12)
            ax.set_title('超参数搜索历史', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            save_path = self.save_dir / 'optimization_history.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(save_path)

        except Exception as e:
            print(f"绘制优化历史失败（备用）: {e}")
            return None

    def plot_param_importance(self) -> Optional[str]:
        """
        绘制参数重要性分析图
        """
        try:
            import optuna.visualization as vis

            fig = vis.plot_param_importances(self.study)
            save_path = self.save_dir / 'param_importance.png'
            fig.write_image(str(save_path), scale=2)
            return str(save_path)
        except Exception as e:
            print(f"绘制参数重要性失败: {e}")
            return self._plot_param_importance_fallback()

    def _plot_param_importance_fallback(self) -> Optional[str]:
        """
        使用 matplotlib 绘制参数重要性（备用方案）
        """
        try:
            from optuna.importance import get_param_importances

            importances = get_param_importances(self.study)

            if not importances:
                print("  无法计算参数重要性（试验数量不足）")
                return None

            # 排序
            sorted_items = sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            )
            params = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

            fig, ax = plt.subplots(figsize=(10, max(6, len(params) * 0.4)))

            colors = plt.cm.Blues(
                [0.3 + 0.7 * v / max(values) for v in values]
            )
            bars = ax.barh(params, values, color=colors)

            ax.set_xlabel('重要性', fontsize=12)
            ax.set_title('超参数重要性分析', fontsize=14)
            ax.invert_yaxis()

            # 添加数值标签
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=10
                )

            plt.tight_layout()
            save_path = self.save_dir / 'param_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(save_path)

        except Exception as e:
            print(f"绘制参数重要性失败（备用）: {e}")
            return None

    def plot_parallel_coordinates(self) -> Optional[str]:
        """
        绘制平行坐标图
        """
        try:
            import optuna.visualization as vis

            fig = vis.plot_parallel_coordinate(self.study)
            save_path = self.save_dir / 'parallel_coordinates.png'
            fig.write_image(str(save_path), scale=2)
            return str(save_path)
        except Exception as e:
            print(f"绘制平行坐标图失败: {e}")
            return self._plot_parallel_coordinates_fallback()

    def _plot_parallel_coordinates_fallback(self) -> Optional[str]:
        """
        使用 matplotlib 绘制平行坐标图（备用方案）
        """
        try:
            import numpy as np
            from matplotlib.colors import Normalize
            from matplotlib.cm import ScalarMappable

            trials = self.study.trials
            completed = [t for t in trials if t.state.name == 'COMPLETE']

            if len(completed) < 2:
                print("  试验数量不足，无法绘制平行坐标图")
                return None

            # 获取所有参数名
            param_names = list(completed[0].params.keys())
            if not param_names:
                return None

            # 构建数据矩阵
            n_trials = len(completed)
            n_params = len(param_names)

            # 标准化参数值到 [0, 1]
            param_data = {name: [] for name in param_names}
            values = []

            for trial in completed:
                values.append(trial.value)
                for name in param_names:
                    param_data[name].append(trial.params.get(name, 0))

            # 标准化
            normalized_data = np.zeros((n_trials, n_params))
            for i, name in enumerate(param_names):
                data = np.array(param_data[name])
                if isinstance(data[0], (int, float)):
                    min_val, max_val = data.min(), data.max()
                    if max_val > min_val:
                        normalized_data[:, i] = (data - min_val) / (max_val - min_val)
                    else:
                        normalized_data[:, i] = 0.5
                else:
                    # 分类变量
                    unique_vals = list(set(data))
                    for j, v in enumerate(data):
                        normalized_data[j, i] = unique_vals.index(v) / max(
                            len(unique_vals) - 1, 1
                        )

            # 绘图
            fig, ax = plt.subplots(figsize=(12, 6))

            # 颜色映射
            norm = Normalize(vmin=min(values), vmax=max(values))
            cmap = plt.cm.RdYlGn_r

            x = np.arange(n_params)
            for i in range(n_trials):
                color = cmap(norm(values[i]))
                ax.plot(x, normalized_data[i], c=color, alpha=0.5, linewidth=1)

            # 设置坐标轴
            ax.set_xticks(x)
            ax.set_xticklabels(param_names, rotation=45, ha='right')
            ax.set_ylabel('标准化参数值', fontsize=12)
            ax.set_title('超参数平行坐标图', fontsize=14)

            # 添加颜色条
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('验证 RMSE (°C)', fontsize=10)

            plt.tight_layout()
            save_path = self.save_dir / 'parallel_coordinates.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            return str(save_path)

        except Exception as e:
            print(f"绘制平行坐标图失败（备用）: {e}")
            return None

    def plot_slice(self) -> Optional[str]:
        """
        绘制参数切片图
        """
        try:
            import optuna.visualization as vis

            fig = vis.plot_slice(self.study)
            save_path = self.save_dir / 'param_slice.png'
            fig.write_image(str(save_path), scale=2)
            return str(save_path)
        except Exception as e:
            print(f"绘制参数切片图失败: {e}")
            return None

    def generate_all_plots(self) -> list:
        """
        生成所有可视化图表

        Returns:
            List[str]: 生成的图表路径列表
        """
        plots = []

        print("生成优化历史图...")
        path = self.plot_optimization_history()
        if path:
            plots.append(path)

        print("生成参数重要性图...")
        path = self.plot_param_importance()
        if path:
            plots.append(path)

        print("生成平行坐标图...")
        path = self.plot_parallel_coordinates()
        if path:
            plots.append(path)

        print("生成参数切片图...")
        path = self.plot_slice()
        if path:
            plots.append(path)

        print(f"\n共生成 {len(plots)} 个可视化图表")
        return plots

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        生成调优报告

        Args:
            output_path: 输出路径（可选）

        Returns:
            str: 报告文本
        """
        trials = self.study.trials
        completed = [t for t in trials if t.state.name == 'COMPLETE']
        pruned = [t for t in trials if t.state.name == 'PRUNED']
        failed = [t for t in trials if t.state.name == 'FAIL']

        best_trial = self.study.best_trial if completed else None

        report = f"""
{'=' * 80}
超参数调优报告
{'=' * 80}

【实验概况】
  总试验次数: {len(trials)}
  完成试验: {len(completed)}
  剪枝试验: {len(pruned)}
  失败试验: {len(failed)}
  完成率: {len(completed) / max(len(trials), 1) * 100:.1f}%

"""

        if completed:
            values = [t.value for t in completed]
            report += f"""【性能统计】
  最佳验证 RMSE: {min(values):.4f} °C
  最差验证 RMSE: {max(values):.4f} °C
  平均验证 RMSE: {sum(values) / len(values):.4f} °C

"""

        if best_trial:
            report += f"""【最优参数】
  试验编号: {best_trial.number}
  验证 RMSE: {best_trial.value:.4f} °C

  参数配置:
"""
            for key, value in best_trial.params.items():
                report += f"    {key}: {value}\n"

        report += f"\n{'=' * 80}\n"

        if output_path:
            output_path = Path(output_path)
            output_path.write_text(report, encoding='utf-8')

        return report

    def save_best_params(self, output_path: Optional[Path] = None) -> dict:
        """
        保存最优参数到 JSON 文件

        Args:
            output_path: 输出路径（可选）

        Returns:
            dict: 最优参数字典
        """
        if not self.study.best_trial:
            return {}

        best_params = {
            'trial_number': self.study.best_trial.number,
            'value': self.study.best_trial.value,
            'params': self.study.best_trial.params
        }

        if output_path is None:
            output_path = self.save_dir.parent / 'best_params.json'

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

        return best_params
