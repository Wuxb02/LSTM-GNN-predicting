#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征相关性分析脚本

分析 real_weather_data_2010_2017.npy 中各变量与三种气温(tmin, tmax, tave)的相关性，
为后续特征选择提供依据。

输出:
    1. 相关性热力图
    2. 各变量与气温的相关系数排名
    3. 统计检验结果(p值)
    4. 特征选择建议

作者: GNN气温预测项目组
日期: 2025-12-16
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 特征名称定义(共28个特征)
FEATURE_NAMES = [
    'x',                          # 0: 经度
    'y',                          # 1: 纬度
    'height',                     # 2: 海拔高度(m)
    'tmin',                       # 3: 日最低气温(°C)
    'tmax',                       # 4: 日最高气温(°C)
    'tave',                       # 5: 日平均气温(°C)
    'pre',                        # 6: 日降水量(mm)
    'prs',                        # 7: 气压(hPa)
    'rh',                         # 8: 相对湿度(%)
    'win',                        # 9: 风速(m/s)
    'BH',                         # 10: 建筑平均高度(m)
    'BHstd',                      # 11: 建筑高度标准差(m)
    'SCD',                        # 12: 地表覆盖密度
    'PLA',                        # 13: 路面铺装率
    'λp',                         # 14: 天空开阔度参数λp
    'λb',                         # 15: 天空开阔度参数λb
    'POI',                        # 16: 兴趣点密度
    'POW',                        # 17: 工作人口密度
    'POV',                        # 18: 访问人口密度
    'NDVI',                       # 19: 归一化植被指数
    'surface_pressure',           # 20: ERA5地表气压(Pa)
    'surface_solar_radiation',    # 21: ERA5太阳辐射(J/m²)
    'u_wind_10m',                 # 22: 10m U分量风速(m/s)
    'v_wind_10m',                 # 23: 10m V分量风速(m/s)
    'VegHeight_mean',             # 24: 植被高度均值(m)
    'VegHeight_std',              # 25: 植被高度标准差(m)
    'doy',                        # 26: 年内日序数(1-366)
    'month',                      # 27: 月份(1-12)
]

# 特征分类
FEATURE_CATEGORIES = {
    '地理特征': [0, 1, 2],                          # x, y, height
    '气温特征': [3, 4, 5],                          # tmin, tmax, tave
    '气象要素': [6, 7, 8, 9],                       # pre, prs, rh, win
    '城市形态': [10, 11, 12, 13, 14, 15],           # BH, BHstd, SCD, PLA, λp, λb
    '人口活动': [16, 17, 18],                       # POI, POW, POV
    '植被特征': [19, 24, 25],                       # NDVI, VegHeight_mean/std
    'ERA5再分析': [20, 21, 22, 23],                 # surface_pressure, radiation, u/v wind
    '时间特征': [26, 27],                           # doy, month
}

# 目标气温索引
TARGET_INDICES = {
    'tmin': 3,
    'tmax': 4,
    'tave': 5,
}


def load_data(data_path: str) -> np.ndarray:
    """
    加载气象数据

    Args:
        data_path: 数据文件路径

    Returns:
        np.ndarray: 形状为[2922, 28, 28]的数据数组
    """
    print(f"正在加载数据: {data_path}")
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")
    print(f"时间步数: {data.shape[0]}天")
    print(f"气象站数: {data.shape[1]}个")
    print(f"特征数量: {data.shape[2]}个")
    return data


def compute_correlation_matrix(
    data: np.ndarray,
    method: str = 'pearson'
) -> tuple:
    """
    计算所有特征之间的相关系数矩阵

    Args:
        data: 形状为[time, stations, features]的数据
        method: 相关性计算方法 ('pearson', 'spearman', 'kendall')

    Returns:
        tuple: (相关系数矩阵, p值矩阵)
    """
    print(f"\n使用 {method} 方法计算相关系数...")

    # 将数据展平为 [time*stations, features]
    n_time, n_stations, n_features = data.shape
    data_flat = data.reshape(-1, n_features)

    # 移除包含NaN的行
    valid_mask = ~np.isnan(data_flat).any(axis=1)
    data_valid = data_flat[valid_mask]
    print(f"有效样本数: {data_valid.shape[0]} / {data_flat.shape[0]}")

    # 计算相关系数矩阵
    corr_matrix = np.zeros((n_features, n_features))
    pval_matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            elif i < j:
                if method == 'pearson':
                    corr, pval = stats.pearsonr(data_valid[:, i], data_valid[:, j])
                elif method == 'spearman':
                    corr, pval = stats.spearmanr(data_valid[:, i], data_valid[:, j])
                elif method == 'kendall':
                    corr, pval = stats.kendalltau(data_valid[:, i], data_valid[:, j])
                else:
                    raise ValueError(f"不支持的方法: {method}")

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                pval_matrix[i, j] = pval
                pval_matrix[j, i] = pval

    return corr_matrix, pval_matrix


def compute_target_correlations(
    data: np.ndarray,
    target_indices: dict = TARGET_INDICES,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    计算各特征与目标气温变量的相关性

    Args:
        data: 形状为[time, stations, features]的数据
        target_indices: 目标变量索引字典
        method: 相关性计算方法

    Returns:
        pd.DataFrame: 包含相关系数和p值的数据框
    """
    print(f"\n计算各特征与气温变量的{method}相关系数...")

    n_time, n_stations, n_features = data.shape
    data_flat = data.reshape(-1, n_features)

    # 移除NaN
    valid_mask = ~np.isnan(data_flat).any(axis=1)
    data_valid = data_flat[valid_mask]

    results = []

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        row = {'特征索引': feat_idx, '特征名称': feat_name}

        for target_name, target_idx in target_indices.items():
            if method == 'pearson':
                corr, pval = stats.pearsonr(
                    data_valid[:, feat_idx],
                    data_valid[:, target_idx]
                )
            elif method == 'spearman':
                corr, pval = stats.spearmanr(
                    data_valid[:, feat_idx],
                    data_valid[:, target_idx]
                )
            else:
                corr, pval = stats.kendalltau(
                    data_valid[:, feat_idx],
                    data_valid[:, target_idx]
                )

            row[f'{target_name}_corr'] = corr
            row[f'{target_name}_pval'] = pval
            row[f'{target_name}_abs_corr'] = abs(corr)

        results.append(row)

    df = pd.DataFrame(results)
    return df


def compute_seasonal_correlations(
    data: np.ndarray,
    target_indices: dict = TARGET_INDICES,
    method: str = 'pearson'
) -> dict:
    """
    按季节计算相关性

    Args:
        data: 形状为[time, stations, features]的数据
        target_indices: 目标变量索引字典
        method: 相关性计算方法

    Returns:
        dict: 各季节的相关性数据框
    """
    print("\n按季节计算相关性...")

    # 获取月份信息(索引27)
    months = data[:, 0, 27].astype(int)

    # 季节定义
    seasons = {
        '春季(3-5月)': [3, 4, 5],
        '夏季(6-8月)': [6, 7, 8],
        '秋季(9-11月)': [9, 10, 11],
        '冬季(12-2月)': [12, 1, 2],
    }

    seasonal_results = {}

    for season_name, season_months in seasons.items():
        # 筛选该季节的数据
        mask = np.isin(months, season_months)
        season_data = data[mask]

        print(f"  {season_name}: {season_data.shape[0]}天")

        # 计算相关性
        df = compute_target_correlations(season_data, target_indices, method)
        seasonal_results[season_name] = df

    return seasonal_results


def compute_station_correlations(
    data: np.ndarray,
    target_indices: dict = TARGET_INDICES,
    method: str = 'pearson'
) -> dict:
    """
    按气象站计算相关性

    Args:
        data: 形状为[time, stations, features]的数据
        target_indices: 目标变量索引字典
        method: 相关性计算方法

    Returns:
        dict: 各站点的相关性数据框
    """
    print("\n按气象站计算相关性...")

    n_stations = data.shape[1]
    station_results = {}

    for station_idx in range(n_stations):
        station_data = data[:, station_idx:station_idx + 1, :]
        df = compute_target_correlations(station_data, target_indices, method)
        station_results[f'站点{station_idx}'] = df

    return station_results


def plot_full_correlation_heatmap(
    corr_matrix: np.ndarray,
    save_path: str,
    figsize: tuple = (16, 14)
):
    """
    绘制完整的特征相关性热力图

    Args:
        corr_matrix: 相关系数矩阵
        save_path: 保存路径
        figsize: 图形大小
    """
    print(f"\n绘制完整相关性热力图...")

    fig, ax = plt.subplots(figsize=figsize)

    # 创建掩码(只显示下三角)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # 绘制热力图
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=FEATURE_NAMES,
        yticklabels=FEATURE_NAMES,
        annot_kws={'size': 6},
        ax=ax
    )

    ax.set_title('特征相关性矩阵热力图', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_target_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str,
    figsize: tuple = (12, 10)
):
    """
    绘制与气温变量相关性的热力图

    Args:
        df: 相关性数据框
        save_path: 保存路径
        figsize: 图形大小
    """
    print(f"\n绘制气温相关性热力图...")

    # 提取相关系数列
    corr_cols = ['tmin_corr', 'tmax_corr', 'tave_corr']
    corr_data = df[corr_cols].values

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    sns.heatmap(
        corr_data,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        xticklabels=['最低气温(tmin)', '最高气温(tmax)', '平均气温(tave)'],
        yticklabels=df['特征名称'].values,
        annot_kws={'size': 9},
        ax=ax
    )

    ax.set_title('各特征与气温变量的相关系数', fontsize=14, fontweight='bold')
    ax.set_xlabel('气温变量', fontsize=12)
    ax.set_ylabel('特征', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_correlation_ranking(
    df: pd.DataFrame,
    target: str,
    save_path: str,
    top_n: int = 20,
    figsize: tuple = (12, 8)
):
    """
    绘制相关性排名条形图

    Args:
        df: 相关性数据框
        target: 目标变量名('tmin', 'tmax', 'tave')
        save_path: 保存路径
        top_n: 显示前N个特征
        figsize: 图形大小
    """
    print(f"\n绘制{target}相关性排名图...")

    # 排除目标变量自身
    df_filtered = df[~df['特征名称'].isin(['tmin', 'tmax', 'tave'])].copy()

    # 按绝对值排序
    df_sorted = df_filtered.sort_values(
        f'{target}_abs_corr',
        ascending=True
    ).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    # 根据正负相关性设置颜色
    colors = ['#d73027' if x > 0 else '#4575b4'
              for x in df_sorted[f'{target}_corr']]

    bars = ax.barh(
        df_sorted['特征名称'],
        df_sorted[f'{target}_corr'],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )

    # 添加数值标签
    for bar, val in zip(bars, df_sorted[f'{target}_corr']):
        width = bar.get_width()
        ax.text(
            width + 0.02 if width > 0 else width - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}',
            va='center',
            ha='left' if width > 0 else 'right',
            fontsize=9
        )

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel('Pearson相关系数', fontsize=12)
    ax.set_title(
        f'各特征与{target}的相关性排名 (Top {top_n})',
        fontsize=14,
        fontweight='bold'
    )

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d73027', edgecolor='black', label='正相关'),
        Patch(facecolor='#4575b4', edgecolor='black', label='负相关')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_seasonal_comparison(
    seasonal_results: dict,
    target: str,
    save_path: str,
    top_n: int = 10,
    figsize: tuple = (14, 8)
):
    """
    绘制季节相关性对比图

    Args:
        seasonal_results: 各季节相关性结果
        target: 目标变量名
        save_path: 保存路径
        top_n: 显示前N个特征
        figsize: 图形大小
    """
    print(f"\n绘制{target}季节对比图...")

    # 收集所有季节的数据
    all_data = []
    for season, df in seasonal_results.items():
        df_temp = df[~df['特征名称'].isin(['tmin', 'tmax', 'tave'])].copy()
        df_temp['季节'] = season
        all_data.append(df_temp)

    df_all = pd.concat(all_data, ignore_index=True)

    # 选择全年平均绝对相关性最高的特征
    avg_corr = df_all.groupby('特征名称')[f'{target}_abs_corr'].mean()
    top_features = avg_corr.nlargest(top_n).index.tolist()

    df_plot = df_all[df_all['特征名称'].isin(top_features)]

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制分组条形图
    seasons = list(seasonal_results.keys())
    x = np.arange(len(top_features))
    width = 0.2

    for i, season in enumerate(seasons):
        season_data = df_plot[df_plot['季节'] == season]
        # 确保顺序一致
        season_data = season_data.set_index('特征名称').loc[top_features]
        ax.bar(
            x + i * width,
            season_data[f'{target}_corr'],
            width,
            label=season,
            alpha=0.8
        )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(top_features, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Pearson相关系数', fontsize=12)
    ax.set_title(
        f'各特征与{target}的季节相关性对比 (Top {top_n})',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def plot_category_correlation(
    df: pd.DataFrame,
    save_path: str,
    figsize: tuple = (12, 6)
):
    """
    按特征类别绘制相关性分布箱线图

    Args:
        df: 相关性数据框
        save_path: 保存路径
        figsize: 图形大小
    """
    print(f"\n绘制特征类别相关性分布图...")

    # 为每个特征添加类别标签
    df_plot = df.copy()
    df_plot['类别'] = ''
    for category, indices in FEATURE_CATEGORIES.items():
        for idx in indices:
            df_plot.loc[df_plot['特征索引'] == idx, '类别'] = category

    # 移除气温特征自身
    df_plot = df_plot[df_plot['类别'] != '气温特征']

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    targets = ['tmin', 'tmax', 'tave']
    target_names = ['最低气温', '最高气温', '平均气温']

    for ax, target, name in zip(axes, targets, target_names):
        sns.boxplot(
            data=df_plot,
            x='类别',
            y=f'{target}_corr',
            ax=ax,
            palette='Set2'
        )
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('')
        ax.set_ylabel('相关系数')
        ax.set_title(f'与{name}的相关性')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('各类别特征与气温的相关性分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def generate_feature_selection_report(
    df: pd.DataFrame,
    seasonal_results: dict,
    save_path: str,
    significance_level: float = 0.05,
    high_corr_threshold: float = 0.3
):
    """
    生成特征选择建议报告

    Args:
        df: 全年相关性数据框
        seasonal_results: 各季节相关性结果
        save_path: 保存路径
        significance_level: 显著性水平
        high_corr_threshold: 高相关性阈值
    """
    print(f"\n生成特征选择建议报告...")

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("特征相关性分析报告 - 特征选择建议\n")
        f.write("=" * 80 + "\n\n")

        # 1. 数据概述
        f.write("一、数据概述\n")
        f.write("-" * 40 + "\n")
        f.write(f"总特征数量: {len(FEATURE_NAMES)}\n")
        f.write(f"目标变量: tmin(最低气温), tmax(最高气温), tave(平均气温)\n")
        f.write(f"显著性水平: α = {significance_level}\n")
        f.write(f"高相关性阈值: |r| ≥ {high_corr_threshold}\n\n")

        # 2. 全年相关性排名
        for target in ['tmin', 'tmax', 'tave']:
            target_name = {'tmin': '最低气温', 'tmax': '最高气温', 'tave': '平均气温'}
            f.write(f"\n二、与{target_name[target]}({target})的相关性分析\n")
            f.write("-" * 40 + "\n")

            # 排除气温变量自身
            df_filtered = df[~df['特征名称'].isin(['tmin', 'tmax', 'tave'])]

            # 按绝对相关性排序
            df_sorted = df_filtered.sort_values(
                f'{target}_abs_corr',
                ascending=False
            )

            f.write("\n1. 相关性排名 (按|r|降序):\n")
            f.write(f"{'排名':<4} {'特征名称':<25} {'相关系数':>10} {'p值':>12} "
                    f"{'显著性':>8}\n")
            f.write("-" * 65 + "\n")

            for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
                corr = row[f'{target}_corr']
                pval = row[f'{target}_pval']
                sig = "***" if pval < 0.001 else (
                    "**" if pval < 0.01 else (
                        "*" if pval < 0.05 else "ns"
                    )
                )
                f.write(f"{rank:<4} {row['特征名称']:<25} {corr:>10.4f} "
                        f"{pval:>12.2e} {sig:>8}\n")

            # 高相关性特征
            high_corr = df_sorted[df_sorted[f'{target}_abs_corr'] >= high_corr_threshold]
            f.write(f"\n2. 高相关性特征 (|r| ≥ {high_corr_threshold}): "
                    f"{len(high_corr)}个\n")
            for _, row in high_corr.iterrows():
                f.write(f"   - {row['特征名称']}: r = {row[f'{target}_corr']:.4f}\n")

            # 低相关性特征(可能需要移除)
            low_corr = df_sorted[df_sorted[f'{target}_abs_corr'] < 0.1]
            f.write(f"\n3. 低相关性特征 (|r| < 0.1): {len(low_corr)}个\n")
            for _, row in low_corr.iterrows():
                f.write(f"   - {row['特征名称']}: r = {row[f'{target}_corr']:.4f}\n")

        # 3. 季节差异分析
        f.write("\n\n三、季节差异分析\n")
        f.write("-" * 40 + "\n")
        f.write("各季节相关性变化最大的特征(可能需要季节性处理):\n\n")

        for target in ['tmin', 'tmax', 'tave']:
            f.write(f"\n{target}:\n")

            # 计算各特征在不同季节的相关性变化范围
            feature_variation = {}
            for feat_name in FEATURE_NAMES:
                if feat_name in ['tmin', 'tmax', 'tave']:
                    continue

                corrs = []
                for season, df_season in seasonal_results.items():
                    row = df_season[df_season['特征名称'] == feat_name]
                    if len(row) > 0:
                        corrs.append(row[f'{target}_corr'].values[0])

                if corrs:
                    feature_variation[feat_name] = max(corrs) - min(corrs)

            # 按变化范围排序
            sorted_variation = sorted(
                feature_variation.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for feat, var in sorted_variation:
                f.write(f"   - {feat}: 季节变化幅度 = {var:.4f}\n")

        # 4. 特征选择建议
        f.write("\n\n四、特征选择建议\n")
        f.write("-" * 40 + "\n")

        # 综合三种气温的相关性
        df_filtered = df[~df['特征名称'].isin(['tmin', 'tmax', 'tave'])].copy()
        df_filtered['avg_abs_corr'] = (
            df_filtered['tmin_abs_corr'] +
            df_filtered['tmax_abs_corr'] +
            df_filtered['tave_abs_corr']
        ) / 3

        # 推荐特征
        recommended = df_filtered[
            df_filtered['avg_abs_corr'] >= high_corr_threshold
        ].sort_values('avg_abs_corr', ascending=False)

        f.write("\n1. 强烈推荐使用的特征 (三种气温平均|r| ≥ 0.3):\n")
        for _, row in recommended.iterrows():
            f.write(f"   [{row['特征索引']:2d}] {row['特征名称']:<25} "
                    f"(平均|r| = {row['avg_abs_corr']:.4f})\n")

        # 可选特征
        optional = df_filtered[
            (df_filtered['avg_abs_corr'] >= 0.1) &
            (df_filtered['avg_abs_corr'] < high_corr_threshold)
        ].sort_values('avg_abs_corr', ascending=False)

        f.write("\n2. 可选特征 (0.1 ≤ 平均|r| < 0.3):\n")
        for _, row in optional.iterrows():
            f.write(f"   [{row['特征索引']:2d}] {row['特征名称']:<25} "
                    f"(平均|r| = {row['avg_abs_corr']:.4f})\n")

        # 不推荐特征
        not_recommended = df_filtered[
            df_filtered['avg_abs_corr'] < 0.1
        ].sort_values('avg_abs_corr', ascending=False)

        f.write("\n3. 不推荐使用的特征 (平均|r| < 0.1):\n")
        for _, row in not_recommended.iterrows():
            f.write(f"   [{row['特征索引']:2d}] {row['特征名称']:<25} "
                    f"(平均|r| = {row['avg_abs_corr']:.4f})\n")

        # 5. 配置建议
        f.write("\n\n五、config.py 配置建议\n")
        f.write("-" * 40 + "\n")

        recommended_indices = recommended['特征索引'].tolist()
        optional_indices = optional['特征索引'].tolist()

        f.write("\n# 方案1: 仅使用强相关特征\n")
        f.write(f"config.feature_indices = {recommended_indices}\n")

        f.write("\n# 方案2: 使用强相关 + 可选特征\n")
        all_indices = sorted(recommended_indices + optional_indices)
        f.write(f"config.feature_indices = {all_indices}\n")

        f.write("\n# 方案3: 使用全部特征(默认)\n")
        f.write("config.feature_indices = None  # 使用所有特征\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("注: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著\n")
        f.write("=" * 80 + "\n")

    print(f"  已保存: {save_path}")


def save_correlation_csv(df: pd.DataFrame, save_path: str):
    """
    保存相关性结果为CSV文件

    Args:
        df: 相关性数据框
        save_path: 保存路径
    """
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"  已保存: {save_path}")


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    data_path = project_root / 'data' / 'real_weather_data_2010_2017.npy'
    output_dir = project_root / 'myGNN' / 'analysis_results'

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("特征相关性分析")
    print("=" * 60)

    # 1. 加载数据
    data = load_data(str(data_path))

    # 2. 计算完整相关系数矩阵
    corr_matrix, pval_matrix = compute_correlation_matrix(data, method='pearson')

    # 3. 计算与气温变量的相关性
    df_corr = compute_target_correlations(data, method='pearson')

    # 4. 按季节计算相关性
    seasonal_results = compute_seasonal_correlations(data, method='pearson')

    # 5. 生成可视化图表
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)

    # 5.1 完整相关性热力图
    plot_full_correlation_heatmap(
        corr_matrix,
        str(output_dir / 'correlation_matrix_full.png')
    )

    # 5.2 气温相关性热力图
    plot_target_correlation_heatmap(
        df_corr,
        str(output_dir / 'correlation_with_temperature.png')
    )

    # 5.3 各气温变量相关性排名
    for target in ['tmin', 'tmax', 'tave']:
        plot_correlation_ranking(
            df_corr,
            target,
            str(output_dir / f'correlation_ranking_{target}.png')
        )

    # 5.4 季节对比图
    for target in ['tmin', 'tmax', 'tave']:
        plot_seasonal_comparison(
            seasonal_results,
            target,
            str(output_dir / f'seasonal_comparison_{target}.png')
        )

    # 5.5 特征类别相关性分布
    plot_category_correlation(
        df_corr,
        str(output_dir / 'category_correlation_boxplot.png')
    )

    # 6. 保存数据文件
    print("\n" + "=" * 60)
    print("保存分析结果")
    print("=" * 60)

    # 保存相关性矩阵
    np.save(str(output_dir / 'correlation_matrix.npy'), corr_matrix)
    np.save(str(output_dir / 'pvalue_matrix.npy'), pval_matrix)
    print(f"  已保存: {output_dir / 'correlation_matrix.npy'}")
    print(f"  已保存: {output_dir / 'pvalue_matrix.npy'}")

    # 保存CSV
    save_correlation_csv(df_corr, str(output_dir / 'feature_correlation.csv'))

    # 保存季节相关性
    for season, df_season in seasonal_results.items():
        season_name = season.replace('(', '_').replace(')', '').replace('-', '_')
        save_correlation_csv(
            df_season,
            str(output_dir / f'correlation_{season_name}.csv')
        )

    # 7. 生成特征选择报告
    generate_feature_selection_report(
        df_corr,
        seasonal_results,
        str(output_dir / 'feature_selection_report.txt')
    )

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"\n所有结果已保存至: {output_dir}")
    print("\n生成的文件列表:")
    print("  - correlation_matrix_full.png     : 完整特征相关性热力图")
    print("  - correlation_with_temperature.png: 与气温变量相关性热力图")
    print("  - correlation_ranking_*.png       : 相关性排名条形图")
    print("  - seasonal_comparison_*.png       : 季节相关性对比图")
    print("  - category_correlation_boxplot.png: 特征类别相关性分布")
    print("  - feature_correlation.csv         : 相关性数据(CSV)")
    print("  - feature_selection_report.txt    : 特征选择建议报告")


if __name__ == '__main__':
    main()
