"""
增强训练流程模块

提供可配置的损失函数和训练逻辑，支持：
1. 多种损失函数的动态选择
2. 与原训练流程兼容
3. 支持传递时间信息（doy）到损失函数

使用方法:
    from myGNN.train_enhanced import get_loss_function, train_epoch

    # 创建损失函数
    criterion = get_loss_function(config)

    # 训练一个epoch
    loss = train_epoch(model, dataloader, optimizer, criterion, config, device)
"""
import numpy as np
import torch
import torch.nn as nn
from myGNN.losses import WeightedTrendMSELoss


def get_loss_function(config):
    """
    根据配置创建损失函数

    参数:
        config: 配置对象，需包含loss_config属性

    返回:
        nn.Module: 损失函数实例

    支持的损失函数类型:
        - 'MSE': 标准均方误差（默认）
        - 'WeightedTrend': 加权趋势损失（🔥论文方法，推荐）

    示例:
        >>> from myGNN.config import Config
        >>> config = Config()
        >>> config.loss_config.loss_type = 'WeightedTrend'
        >>> criterion = get_loss_function(config)
    """
    # 如果没有loss_config，使用默认MSE
    if not hasattr(config, 'loss_config'):
        print("警告: config中未找到loss_config，使用默认MSE损失")
        return nn.MSELoss()

    loss_cfg = config.loss_config

    # 标准MSE损失
    if loss_cfg.loss_type == 'MSE':
        print("使用标准MSE损失函数")
        return nn.MSELoss()

    # 🔥 加权趋势损失（论文方法 - 推荐）
    elif loss_cfg.loss_type == 'WeightedTrend':
        print(f"使用自适应加权趋势MSE损失函数 (温度加权 + 趋势约束)")
        print(f"  - 固定阈值: {loss_cfg.alert_temp}°C")
        print(f"  - 漏报权重c_under: {loss_cfg.c_under} (低估高温的惩罚)")
        print(f"  - 误报权重c_over: {loss_cfg.c_over} (高估的惩罚)")
        print(f"  - 正确预报高温权重: {loss_cfg.c_default_high}")
        print(f"  - 趋势权重α: {loss_cfg.trend_weight}")

        return WeightedTrendMSELoss(
            alert_temp=loss_cfg.alert_temp,
            c_under=loss_cfg.c_under,
            c_over=loss_cfg.c_over,
            delta=loss_cfg.delta,
            trend_weight=loss_cfg.trend_weight,
            ta_mean=config.ta_mean,
            ta_std=config.ta_std
        )

    else:
        raise ValueError(
            f"未知的损失函数类型: {loss_cfg.loss_type}\n"
            f"支持的类型: MSE, WeightedTrend"
        )


def compute_loss(criterion, pred, label, doy=None, config=None):
    """
    计算损失（根据损失函数类型传递不同参数）

    参数:
        criterion (nn.Module): 损失函数
        pred (torch.Tensor): 预测值（标准化），shape [B, P]
        label (torch.Tensor): 真实值（标准化），shape [B, P]
        doy (torch.Tensor, optional): 年内日序数，shape [B]（暂时保留参数兼容性）
        config: 配置对象（用于获取ta_mean和ta_std）

    返回:
        torch.Tensor: 标量损失值

    异常:
        ValueError: 当损失函数需要的参数未提供时

    示例:
        >>> loss = compute_loss(criterion, pred, label, doy, config)
    """
    # 标准MSE - 不需要额外参数
    if isinstance(criterion, nn.MSELoss):
        return criterion(pred, label)

    # 加权趋势损失 - 需要ta_mean和ta_std
    elif isinstance(criterion, WeightedTrendMSELoss):
        if config is None:
            raise ValueError("WeightedTrendMSELoss需要config参数")
        return criterion(pred, label)

    else:
        # 未知的损失函数类型，尝试直接调用
        return criterion(pred, label)


def train_epoch(model, dataloader, optimizer, scheduler, criterion, config, device):
    """
    训练一个epoch（支持增强损失函数）

    参数:
        model (nn.Module): 模型
        dataloader (DataLoader): 数据加载器
        optimizer (torch.optim.Optimizer): 优化器
        scheduler: 学习率调度器（可选）
        criterion (nn.Module): 损失函数
        config: 配置对象
        device (torch.device): 设备

    返回:
        float: 平均RMSE损失值（°C）- 已反标准化

    注意:
        - 如果dataloader返回(graph_data, doy)元组，会自动提取doy
        - 如果只返回graph_data，doy将为None（仅适用于不需要doy的损失函数）
        - scheduler会在epoch结束时自动调用（针对非ReduceLROnPlateau调度器）
        - 包含梯度裁剪（max_norm=1.0）和NaN/Inf检测

    示例:
        >>> avg_loss = train_epoch(
        ...     model, train_loader, optimizer, scheduler, criterion, config, device
        ... )
        >>> print(f"Epoch RMSE: {avg_loss:.4f} °C")
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_data in dataloader:
        # 检查是否使用增强数据集（返回doy）
        if isinstance(batch_data, tuple) and len(batch_data) == 2:
            graph_data, doy_batch = batch_data
            doy_batch = doy_batch.to(device)
        else:
            graph_data = batch_data
            doy_batch = None

        # 提取图数据（DataLoader返回的是列表，第一个元素是图数据）
        if isinstance(graph_data, list):
            graph_data = graph_data[0]

        # 前向传播 - 根据是否使用边属性调用模型
        if not config.use_edge_attr:
            # 不使用边属性
            feature = graph_data.x.to(device)
            label = graph_data.y.to(device)
            edge_index = graph_data.edge_index.to(device)
            pred = model(feature, edge_index)
        else:
            # 使用边属性
            feature = graph_data.x.to(device)
            label = graph_data.y.to(device)
            edge_index = graph_data.edge_index.to(device)
            edge_attr = graph_data.edge_attr.to(device)
            pred = model(feature, edge_index, edge_attr)

        # 计算损失
        loss = compute_loss(criterion, pred, label, doy_batch, config)

        # 检测 NaN 或 Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n警告: 检测到异常损失值!")
            print(f"  Loss: {loss.item()}")
            print(f"  Pred - min: {pred.min().item():.4f}, max: {pred.max().item():.4f}, mean: {pred.mean().item():.4f}")
            print(f"  Label - min: {label.min().item():.4f}, max: {label.max().item():.4f}, mean: {label.mean().item():.4f}")
            if doy_batch is not None:
                print(f"  DOY - min: {doy_batch.min().item():.1f}, max: {doy_batch.max().item():.1f}")
            raise ValueError("训练过程中出现 NaN 或 Inf 损失值，训练终止")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # 计算平均损失（归一化空间）
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 转换为RMSE（°C）- 与标准训练流程保持一致
    avg_loss_rmse = np.sqrt(avg_loss * (config.ta_std ** 2))

    # 更新学习率（非ReduceLROnPlateau调度器）
    # ReduceLROnPlateau需要验证集损失，在主训练循环中单独处理
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    return avg_loss_rmse


def validate_epoch(model, dataloader, criterion, config, device):
    """
    验证一个epoch（支持增强损失函数）

    参数:
        model (nn.Module): 模型
        dataloader (DataLoader): 数据加载器
        criterion (nn.Module): 损失函数
        config: 配置对象
        device (torch.device): 设备

    返回:
        tuple: (avg_loss_rmse, predict_epoch, label_epoch, time_epoch)
            - avg_loss_rmse (float): 平均RMSE损失值（°C）
            - predict_epoch (np.ndarray): 预测结果 [num_samples, num_stations, pred_len]
            - label_epoch (np.ndarray): 真实标签 [num_samples, num_stations, pred_len]
            - time_epoch (np.ndarray): 时间索引 [num_samples]

    示例:
        >>> val_loss, pred, label, time = validate_epoch(model, val_loader, criterion, config, device)
        >>> print(f"Validation RMSE: {val_loss:.4f} °C")
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    # 收集预测值、标签和时间信息
    predict_list = []
    label_list = []
    time_list = []

    with torch.no_grad():
        for batch_data in dataloader:
            # 检查是否使用增强数据集
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                graph_data, doy_batch = batch_data
                doy_batch = doy_batch.to(device)
            else:
                graph_data = batch_data
                doy_batch = None

            # 提取图数据（DataLoader返回的是列表，第一个元素是图数据）
            if isinstance(graph_data, list):
                graph_data = graph_data[0]

            # 前向传播 - 根据是否使用边属性调用模型
            if not config.use_edge_attr:
                # 不使用边属性
                feature = graph_data.x.to(device)
                label = graph_data.y.to(device)
                edge_index = graph_data.edge_index.to(device)
                pred = model(feature, edge_index)
            else:
                # 使用边属性
                feature = graph_data.x.to(device)
                label = graph_data.y.to(device)
                edge_index = graph_data.edge_index.to(device)
                edge_attr = graph_data.edge_attr.to(device)
                pred = model(feature, edge_index, edge_attr)

            # 计算损失
            loss = compute_loss(criterion, pred, label, doy_batch, config)

            total_loss += loss.item()
            num_batches += 1

            # 反标准化预测值和标签
            pred_denorm = pred.cpu().numpy() * config.ta_std + config.ta_mean
            label_denorm = label.cpu().numpy() * config.ta_std + config.ta_mean

            # 收集数据
            predict_list.append(pred_denorm)
            label_list.append(label_denorm)

            # 提取时间索引
            if hasattr(graph_data, 'time_idx'):
                time_list.append(graph_data.time_idx.cpu().numpy())

    # 计算平均损失（归一化空间）
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 转换为RMSE（°C）
    avg_loss_rmse = np.sqrt(avg_loss * (config.ta_std ** 2))

    # 拼接所有批次的结果
    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0) if time_list else np.array([])

    # 重塑为标准格式 [num_samples, num_stations, pred_len]
    # 当前格式: [num_nodes_total, pred_len]
    # 需要转换为: [num_samples, num_stations, pred_len]
    num_total_nodes = predict_epoch.shape[0]
    num_stations = config.node_num
    num_samples = num_total_nodes // num_stations

    predict_epoch = predict_epoch.reshape(num_samples, num_stations, config.pred_len)
    label_epoch = label_epoch.reshape(num_samples, num_stations, config.pred_len)

    return avg_loss_rmse, predict_epoch, label_epoch, time_epoch
