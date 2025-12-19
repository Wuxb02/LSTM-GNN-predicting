"""
GNN模型可解释性分析 - 主运行脚本

使用方法:
    1. 在下方"配置参数"部分设置参数
    2. 直接运行: python explain_model.py

作者: GNN气温预测项目
日期: 2025
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from myGNN.config import Config, ArchConfig
from myGNN.network_GNN import get_model
from myGNN.dataset import create_dataloaders
from myGNN.explainer import HybridExplainer, ExplainerConfig
from myGNN.explainer.visualize_explainer import generate_all_visualizations


# ============================================================================
# 配置参数 - 在这里修改您的分析参数
# ============================================================================

# 模型路径(必需)
MODEL_PATH = r'.\checkpoints\GAT_SeparateEncoder_20251216_125133\best_model.pth'

# 分析参数
NUM_SAMPLES = 50           # 分析样本数量
EXPLAINER_DATA = 'val_test'  # 分析数据集: 'test'(仅测试集) 或 'val_test'(验证集+测试集)
SEASON = None              # 季节筛选: None(全年), 'spring', 'summer', 'autumn', 'winter'
EPOCHS = 200               # GNNExplainer训练轮数
IG_STEPS = 50              # Integrated Gradients积分步数
TOP_K_EDGES = 90           # 保存Top-K重要边
SAVE_DIR = None            # 保存目录(None=自动设置为模型目录/explanations/)
VISUALIZE = True           # 是否生成可视化图表

# 底图配置
USE_WMTS_BASEMAP = False    # 是否使用Mapbox WMTS在线底图 (True=在线, False=Natural Earth离线)

# ============================================================================
# GAT_SeparateEncoder 专属分析配置 (v3.0)
# ============================================================================

# Cross-Attention分析的动态特征配置（可灵活修改）
# 索引基于动态特征顺序(去除静态特征后的索引)
CROSS_ATTENTION_DYNAMIC_FEATURES = [
    {'name': 'Wind Speed', 'index': 6, 'unit': 'm/s'},       # win (动态特征索引6)
    {'name': 'Temperature', 'index': 2, 'unit': 'C'},        # tave (动态特征索引2)
    {'name': 'Relative Humidity', 'index': 5, 'unit': '%'},  # rh (动态特征索引5)
]

# t-SNE着色的静态特征配置（可灵活修改）
# 索引基于静态特征顺序 [x, y, height, BH, BHstd, SCD, lambda_p, lambda_b, PLA, POI, POW, POV]
TSNE_COLOR_FEATURES = [
    {'name': 'height', 'index': 2, 'label': 'Altitude (m)'},
    {'name': 'BH', 'index': 3, 'label': 'Building Height (m)'},
    {'name': 'lambda_p', 'index': 6, 'label': 'Sky View Factor'},
    {'name': 'PLA', 'index': 8, 'label': 'Green Space Area'},
]

# 静态特征名称列表(与模型配置保持一致)
STATIC_FEATURE_NAMES = [
    'x', 'y', 'height',           # 地理位置 (0-2)
    'BH', 'BHstd', 'SCD',         # 建筑形态 (3-5)
    'lambda_p', 'lambda_b',       # 天空开阔度 (6-7)
    'PLA', 'POI', 'POW', 'POV',   # 地表覆盖 (8-11)
]

# K-Means聚类数量
N_CLUSTERS = 4

# ============================================================================


def main():
    """主函数"""

    print("="*70)
    print("GNN气温预测模型 - 可解释性分析")
    print("="*70)
    print(f"模型路径: {MODEL_PATH}")
    print(f"分析样本数: {NUM_SAMPLES}")
    print(f"季节筛选: {SEASON or '全年'}")
    print(f"生成可视化: {'是' if VISUALIZE else '否'}")
    print("="*70 + "\n")

    # 1. 加载配置和模型
    print("[1/5] 加载配置和模型...")
    config, arch_config = load_config_from_checkpoint(MODEL_PATH)

    model = get_model(config, arch_config)
    checkpoint = torch.load(MODEL_PATH, map_location=config.device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(config.device)
    model.eval()
    print(f"  ✓ 模型加载成功 ({config.exp_model})")

    # 2. 准备数据
    print("\n[2/5] 准备测试数据...")

    # 创建图结构 - 从checkpoint加载,保证与训练时一致
    checkpoint = torch.load(MODEL_PATH, map_location=config.device, weights_only=False)

    if 'graph' in checkpoint:
        # ✅ 使用训练时保存的图对象(保证一致性)
        graph = checkpoint['graph']
        print(f"  ✓ 使用checkpoint中保存的图结构(类型: {config.graph_type})")
        print(f"    节点数: {graph.node_num}, 边数: {graph.edge_index.shape[1]}")
    else:
        # ❌ checkpoint不包含图对象
        raise ValueError(
            "checkpoint中未找到图对象!\n"
            "请确保使用新版训练脚本训练的模型。\n"
            "如果必须使用旧模型,请重新训练或手动修改此脚本。"
        )

    # 创建数据加载器
    _, val_loader, test_loader, stats = create_dataloaders(config, graph)

    # 根据配置选择分析数据集
    if EXPLAINER_DATA == 'val_test':
        # 合并验证集和测试集
        from torch.utils.data import ConcatDataset, DataLoader
        combined_dataset = ConcatDataset([val_loader.dataset, test_loader.dataset])
        explainer_loader = DataLoader(
            combined_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_loader.collate_fn  # PyG 数据需要特殊的 collate 函数
        )
        print(f"  ✓ 使用验证集+测试集进行分析")
        print(f"    验证集: {len(val_loader.dataset)} 样本")
        print(f"    测试集: {len(test_loader.dataset)} 样本")
        print(f"    合计: {len(combined_dataset)} 样本")
    else:
        explainer_loader = test_loader
        print(f"  ✓ 使用测试集进行分析: {len(test_loader.dataset)} 样本")

    # 更新config中的标准化参数(从stats中获取)
    config.ta_mean = stats['ta_mean']
    config.ta_std = stats['ta_std']

    # 3. 配置解释器
    print("\n[3/5] 配置可解释性分析器...")
    exp_config = ExplainerConfig(
        num_samples=NUM_SAMPLES,
        epochs=EPOCHS,
        season=SEASON,
        ig_steps=IG_STEPS,
        top_k_edges=TOP_K_EDGES,
        extract_attention=True  # ⭐启用GAT注意力权重提取
    )

    explainer = HybridExplainer(model, config, exp_config)

    # 4. 运行解释
    print("\n[4/5] 运行可解释性分析...")

    # 确定保存路径
    if SAVE_DIR:
        save_path = SAVE_DIR
    else:
        model_dir = Path(MODEL_PATH).parent
        season_suffix = f"_{SEASON}" if SEASON else ""
        save_path = model_dir / f"explanations{season_suffix}"

    explanation = explainer.explain_full(
        explainer_loader,
        num_samples=NUM_SAMPLES,
        save_path=save_path
    )

    # 5. 生成可视化
    if VISUALIZE:
        print("\n[5/5] 生成可视化图表...")
        viz_dir = Path(save_path) / "visualizations"

        # 加载站点坐标
        station_coords = None
        station_info_path = Path(__file__).parent.parent / 'data' / 'station_info.npy'
        if station_info_path.exists():
            station_info = np.load(station_info_path)
            # station_info shape: [N, 4] → [id, lon, lat, height]
            # station_coords需要 [N, 2] → [lon, lat]
            station_coords = station_info[:config.node_num, 1:3]  # 取lon和lat列
            print(f"  ✓ 加载站点坐标: {station_coords.shape}")
        else:
            print(f"  ⚠ 未找到站点信息文件: {station_info_path}")

        # 定义特征名称(与config.py中的特征索引对应)
        feature_names = [
            'x', 'y', 'height',           # 0-2: 经纬度和海拔
            'tmin', 'tmax', 'tave',       # 3-5: 温度
            'pre', 'prs', 'rh', 'win',    # 6-9: 气象要素
            'BH', 'BHstd',                # 10-11: 建筑高度
            'SCD', 'PLA',                 # 12-13: 地表覆盖
            'λp', 'λb',                   # 14-15: 天空开阔度
            'POI', 'POW', 'POV',          # 16-18: 兴趣点/人口
            'NDVI',                       # 19: 植被指数
            'sfc_pres', 'sfc_solar',      # 20-21: ERA5
            'u_wind', 'v_wind',           # 22-23: 风速分量
            'VegHeight_mean', 'VegHeight_std',  # 24-25: 植被高度特征
        ]
        # 如果启用了时间编码,添加时间特征名称
        if config.add_temporal_encoding:
            feature_names.extend(['doy_sin', 'doy_cos', 'month_sin', 'month_cos'])

        # 获取气象数据路径(用于温度相关性分析)
        weather_data_path = Path(__file__).parent.parent / 'data' / 'real_weather_data_2010_2017.npy'

        try:
            generate_all_visualizations(
                explanation_data_path=Path(save_path) / 'explanation_data.npz',
                save_dir=viz_dir,
                station_coords=station_coords,
                feature_names=feature_names,
                use_basemap=USE_WMTS_BASEMAP,
                weather_data_path=weather_data_path,
                train_indices=(config.train_start, config.train_end),
                generate_attention_analysis=True
            )
        except Exception as e:
            print(f"  ⚠ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
            print(f"  您可以稍后手动调用visualize_explainer.py")
    else:
        print("\n[5/5] 跳过可视化生成")

    # ============ GAT_SeparateEncoder 专属可解释性分析 ============
    if config.exp_model == 'GAT_SeparateEncoder':
        print("\n" + "="*70)
        print("[6/6] GAT_SeparateEncoder 专属分析")
        print("="*70)

        try:
            from myGNN.explainer.visualize_explainer import (
                generate_gat_separate_encoder_visualizations
            )

            # 加载站点信息
            station_info_path = Path(__file__).parent.parent / 'data' / 'station_info.npy'
            if station_info_path.exists():
                station_info = np.load(station_info_path)
                station_info = station_info[:config.node_num]  # 仅取使用的站点

                # 生成专属可视化
                gat_results = generate_gat_separate_encoder_visualizations(
                    model=model,
                    test_loader=explainer_loader,
                    device=config.device,
                    save_dir=viz_dir,
                    station_info=station_info,
                    static_feature_names=STATIC_FEATURE_NAMES,
                    dynamic_feature_configs=CROSS_ATTENTION_DYNAMIC_FEATURES,
                    tsne_color_configs=TSNE_COLOR_FEATURES,
                    num_samples=NUM_SAMPLES,
                    n_clusters=N_CLUSTERS
                )

                print(f"\n  Cross-Attention分析完成:")
                print(f"    - 动态特征: {[f['name'] for f in CROSS_ATTENTION_DYNAMIC_FEATURES]}")
                print(f"  Node Embedding分析完成:")
                print(f"    - t-SNE着色特征: {[f['name'] for f in TSNE_COLOR_FEATURES]}")
                print(f"    - K-Means聚类数: {N_CLUSTERS}")
            else:
                print(f"  [WARNING] 未找到站点信息文件: {station_info_path}")

        except Exception as e:
            print(f"  [ERROR] GAT_SeparateEncoder专属分析失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("可解释性分析完成!")
    print("="*70)
    print(f"\n结果保存在: {save_path}")
    print(f"  - explanation_data.npz: 原始数据")
    print(f"  - important_edges.txt: Top-K重要边列表")
    if VISUALIZE:
        print(f"  - visualizations/: 可视化图表")
        if config.exp_model == 'GAT_SeparateEncoder':
            print(f"  - visualizations/cross_attention_*.png: Cross-Attention热力图")
            print(f"  - visualizations/node_embedding_tsne_*.png: t-SNE可视化")
            print(f"  - visualizations/embedding_correlation.png: 嵌入-特征相关性")
            print(f"  - visualizations/node_embeddings.csv: 节点嵌入CSV")
    print()

    return explanation


def load_config_from_checkpoint(model_path):
    """
    从checkpoint加载配置

    支持两种格式:
    1. 新格式: checkpoint包含 'model_state_dict', 'config', 'arch_config'
    2. 旧格式: checkpoint仅包含 model.state_dict()

    对于旧格式，尝试从同目录的config.txt推断关键参数

    Args:
        model_path: 模型权重路径

    Returns:
        config, arch_config
    """
    model_dir = Path(model_path).parent

    # 尝试从checkpoint加载配置
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict) and 'config' in checkpoint and 'arch_config' in checkpoint:
        config = checkpoint['config']
        arch_config = checkpoint['arch_config']
        print(f"  ✓ 从checkpoint加载配置")
    else:
        # 旧格式checkpoint,尝试从config.txt推断参数
        print(f"  ⚠ checkpoint不包含配置")
        config = Config()
        arch_config = ArchConfig()

        # 尝试从目录名解析模型类型
        dir_name = model_dir.name
        if 'GAT_LSTM' in dir_name:
            config.exp_model = 'GAT_LSTM'
        elif 'GSAGE_LSTM' in dir_name:
            config.exp_model = 'GSAGE_LSTM'

        # 尝试从config.txt读取参数
        config_txt = model_dir / 'config.txt'
        if config_txt.exists():
            print(f"  ⚠ 尝试从config.txt推断参数...")
            try:
                with open(config_txt, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 解析关键参数
                    import re
                    # 历史长度
                    match = re.search(r'历史长度:\s*(\d+)', content)
                    if match:
                        config.hist_len = int(match.group(1))
                    # 预测长度
                    match = re.search(r'预测长度:\s*(\d+)', content)
                    if match:
                        config.pred_len = int(match.group(1))
                    # 输入维度（in_dim是计算属性，此处仅作参考）
                    # 时间编码
                    if '时间编码: 禁用' in content:
                        config.add_temporal_encoding = False

                print(f"  ⚠ 从config.txt推断: hist_len={config.hist_len}, "
                      f"pred_len={config.pred_len}, model={config.exp_model}")
            except Exception as e:
                print(f"  ⚠ 解析config.txt失败: {e}")
        else:
            print(f"  ⚠ 使用默认配置，可能与训练时参数不匹配！")
            print(f"     请确保config.py中的参数与训练时一致")

    return config, arch_config


if __name__ == '__main__':
    main()
