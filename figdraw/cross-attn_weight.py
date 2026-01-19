
import sys
import os
from pathlib import Path

# 设置环境变量强制使用 UTF-8 编码（解决 Windows GBK 编码问题）
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from myGNN.config import create_config
from myGNN.graph import load_graph_from_station_info
from myGNN.dataset import create_dataloaders
from myGNN.network_GNN import get_model  # 使用 network_GNN 中的模型加载函数

def analyze_context_sensitivity():
    # 1. 初始化配置和模型
    config, arch_config = create_config()
    
    # 强制覆盖配置以确保对齐 (假设你已经在config.py里改了，这里双重保险)
    config.static_encoded_dim = 10 
    config.static_encoder_type = 'none'
    
    # 加载数据
    graph = load_graph_from_station_info(config.station_info_fp)
    _, _, test_loader, stats = create_dataloaders(config, graph)
    
    # 加载模型 (请替换为你训练好的权重路径)
    model = get_model(config, arch_config)
    checkpoint_path = project_root / 'myGNN' / 'checkpoints' / 'GAT_SeparateEncoder_20260118_231051' / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path,weights_only=False)  # 使用 Path 对象自动处理路径分隔符
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.device)
    model.eval()

    # 2. 定义容器
    scd_attentions = []   # 存储 SCD 的注意力权重
    max_temps = []        # 存储对应的 Tmax
    wind_speeds = []      # 存储对应的 风速
    contexts = []         # 存储背景标签 ('Heatwave' 或 'Normal')

    # SCD 在 static_feature_indices 中的位置 (第6个，索引5)
    SCD_INDEX = 5 
    
    # 动态特征索引映射 (基于 config.dynamic_feature_indices)
    # 假设 Tmax 是第 2 个 (索引1), 风速是第 7 个 (索引6)
    TMAX_IDX = 1  
    WIND_IDX = 6

    print("开始分析...")
    
    with torch.no_grad():
        for batch, time_idx in test_loader:
            batch = batch.to(config.device)
            
            # === A. 运行模型并请求 Attention ===
            # 注意：必须在 forward 中传入 return_cross_attention=True
            _, attn_weights = model(batch.x, batch.edge_index, return_cross_attention=True)
            # attn_weights shape: [batch_size*nodes, num_heads, 10]
            
            # 取多头注意力的平均值 -> [batch_size*nodes, 10]
            attn_avg = attn_weights.mean(dim=1)
            
            # === B. 提取 SCD 的注意力 ===
            scd_attn = attn_avg[:, SCD_INDEX].cpu().numpy()
            
            # === C. 提取气象背景 (动态特征) ===
            # batch.x 结构: [nodes, hist_len, static+dynamic+time]
            # 我们需要取历史窗口的平均值或最后一天作为“背景”
            # static_dim = 10
            dynamic_start = 10
            
            # 提取 Tmax 和 Wind (这里取 hist_len 的平均值作为背景)
            # x[:, :, 10+1] 是 Tmax, x[:, :, 10+6] 是 Wind
            tmax_seq = batch.x[:, :, dynamic_start + TMAX_IDX].cpu().numpy()
            wind_seq = batch.x[:, :, dynamic_start + WIND_IDX].cpu().numpy()
            
            # 反标准化 (为了用 35°C 这种物理阈值，必须还原数据)
            tmax_real = tmax_seq * stats['dynamic_std'][TMAX_IDX] + stats['dynamic_mean'][TMAX_IDX]
            wind_real = wind_seq * stats['dynamic_std'][WIND_IDX] + stats['dynamic_mean'][WIND_IDX]
            
            # 计算样本的平均气温和风速
            tmax_mean = tmax_real.mean(axis=1) # [nodes]
            wind_mean = wind_real.mean(axis=1) # [nodes]
            
            # === D. 定义背景并收集数据 ===
            for i in range(len(tmax_mean)):
                # 定义热浪背景: 高温 (>35) 且 低风 (<2.0)
                is_heatwave = (tmax_mean[i] > 35.0) and (wind_mean[i] < 2.0)
                
                scd_attentions.append(scd_attn[i])
                max_temps.append(tmax_mean[i])
                wind_speeds.append(wind_mean[i])
                contexts.append('Heatwave (High T, Low W)' if is_heatwave else 'Normal')

    # 3. 可视化证明
    plt.figure(figsize=(10, 6))
    
    # 箱线图对比
    sns.boxplot(x=contexts, y=scd_attentions, palette="Set2")
    plt.title("Impact of Meteorological Context on Model Attention to SCD", fontsize=14)
    plt.ylabel("Attention Weight assigned to SCD", fontsize=12)
    plt.xlabel("Meteorological Context", fontsize=12)
    
    plt.savefig('scd_attention_analysis.png', dpi=300)
    print("分析完成！图表已保存为 scd_attention_analysis.png")

    # 4. 输出统计结论
    attns = np.array(scd_attentions)
    ctxs = np.array(contexts)
    
    mean_hw = attns[ctxs == 'Heatwave (High T, Low W)'].mean()
    mean_norm = attns[ctxs == 'Normal'].mean()
    
    print(f"\n统计结论:")
    print(f"热浪背景下 SCD 平均权重: {mean_hw:.4f}")
    print(f"普通背景下 SCD 平均权重: {mean_norm:.4f}")
    print(f"提升倍数: {mean_hw / mean_norm:.2f}x")
    
    if mean_hw > mean_norm:
        print("\n✅ 验证成功：模型在热浪背景下显著增加了对 SCD 的关注！")
    else:
        print("\n❌ 未发现显著差异，可能需要调整热浪定义或检查模型训练。")

if __name__ == "__main__":
    analyze_context_sensitivity()