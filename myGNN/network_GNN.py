"""
myGNN训练核心模块

功能：
1. 模型工厂函数（get_model）
2. 训练、验证、测试函数
3. 评估指标计算（RMSE、MAE）
4. 实验信息输出

修复内容：
1. 修复if/if结构为if/elif（use_edge_attr判断）
2. 移除未实现的模型引用（GC_LSTM, GAT_test）
3. 更新导入路径
4. 简化实验信息输出函数

作者: GNN气温预测项目
日期: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from myGNN.models import GAT_LSTM, GSAGE_LSTM, LSTM_direct, GAT_Pure
from myGNN.models.GAT_SeparateEncoder import GAT_SeparateEncoder
from myGNN.models.GSAGE_SeparateEncoder import GSAGE_SeparateEncoder


def get_optimizer(model, config):
    """
    优化器工厂函数

    Args:
        model: PyTorch模型
        config: 配置对象

    Returns:
        optimizer: 对应的优化器实例

    支持的优化器：
        - 'Adam': Adam优化器
        - 'AdamW': AdamW优化器（带权重衰减）
        - 'SGD': 随机梯度下降（支持动量）
        - 'RMSprop': RMSprop优化器
    """
    if config.optimizer == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "RMSprop":
        return optim.RMSprop(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"未知的优化器类型: {config.optimizer}")


def get_scheduler(optimizer, config):
    """
    学习率调度器工厂函数

    Args:
        optimizer: 优化器
        config: 配置对象

    Returns:
        scheduler: 对应的调度器实例（如果配置为'None'则返回None）

    支持的调度器：
        - 'StepLR': 固定步长衰减
        - 'CosineAnnealingLR': 余弦退火
        - 'ReduceLROnPlateau': 自适应学习率（基于验证损失）
        - 'MultiStepLR': 多步长衰减
        - 'None': 不使用调度器
    """
    if config.scheduler == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    elif config.scheduler == "CosineAnnealingLR":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.T_max, eta_min=config.eta_min
        )
    elif config.scheduler == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.factor,
            patience=config.patience,
            # verbose=True
        )
    elif config.scheduler == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.milestones, gamma=config.gamma
        )
    elif config.scheduler == "None" or config.scheduler is None:
        return None
    else:
        raise ValueError(f"未知的调度器类型: {config.scheduler}")


def get_model(config, arch_arg):
    """
    模型工厂函数

    Args:
        config: 配置对象
        arch_arg: 模型架构参数对象

    Returns:
        model: 对应的模型实例

    支持的模型：
        基础模型:
        - 'GAT_LSTM': 图注意力网络 + LSTM
        - 'GSAGE_LSTM': GraphSAGE + LSTM
        - 'LSTM': 基线LSTM模型
        - 'GAT_Pure': 纯图注意力网络（无LSTM）

        分离式编码模型:
        - 'GAT_SeparateEncoder': GAT + 分离式编码器（静态/动态分离）
        - 'GSAGE_SeparateEncoder': GraphSAGE + 分离式编码器（静态/动态分离）
    """
    # ===== 基础模型 =====
    if config.exp_model == "GAT_LSTM":
        return GAT_LSTM(config, arch_arg)
    elif config.exp_model == "GSAGE_LSTM":
        return GSAGE_LSTM(config, arch_arg)
    elif config.exp_model == "LSTM":
        return LSTM_direct(config, arch_arg)
    elif config.exp_model == "GAT_Pure":
        return GAT_Pure(config, arch_arg)

    # ===== 分离式编码模型 =====
    elif config.exp_model == "GAT_SeparateEncoder":
        return GAT_SeparateEncoder(config, arch_arg)
    elif config.exp_model == "GSAGE_SeparateEncoder":
        return GSAGE_SeparateEncoder(config, arch_arg)

    else:
        raise ValueError(
            f"未知的模型类型: {config.exp_model}\n"
            f"支持: \n"
            f"基础模型: GAT_LSTM, GSAGE_LSTM, LSTM, GAT_Pure\n"
            f"分离式编码: GAT_SeparateEncoder, GSAGE_SeparateEncoder"
        )


def get_metric(predict_epoch, label_epoch):
    """
    计算评估指标

    Args:
        predict_epoch: 预测值数组
                      支持2D: [num_samples, pred_len]
                      或3D: [num_samples, num_stations, pred_len]
        label_epoch: 真实值数组 (与predict_epoch相同维度)

    Returns:
        rmse: 均方根误差 (Root Mean Square Error)
        mae: 平均绝对误差 (Mean Absolute Error)
        r2: 决定系数 (R-squared / Coefficient of Determination)
        bias: 偏差 (Bias = mean(predictions - labels))
    """
    # MAE: 平均绝对误差
    mae = np.mean(np.abs(predict_epoch - label_epoch))

    # RMSE: 均方根误差
    rmse = np.sqrt(np.mean(np.square(predict_epoch - label_epoch)))

    # R²: 决定系数
    # R² = 1 - (SS_res / SS_tot)
    # SS_res = Σ(y_true - y_pred)²  # 残差平方和
    # SS_tot = Σ(y_true - y_mean)²  # 总平方和
    ss_res = np.sum(np.square(label_epoch - predict_epoch))
    ss_tot = np.sum(np.square(label_epoch - np.mean(label_epoch)))

    # 避免除零：如果ss_tot为0（所有真实值相同），R²无意义，返回0
    if ss_tot < 1e-10:
        r2 = 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    # Bias: 系统性偏差
    # 正值表示模型倾向于高估，负值表示模型倾向于低估
    bias = np.mean(predict_epoch - label_epoch)

    return rmse, mae, r2, bias


def get_metrics_per_step(predict_epoch, label_epoch):
    """
    计算每个预测步长的指标

    Args:
        predict_epoch: 预测值数组 [num_samples, num_stations, pred_len]
        label_epoch: 真实值数组 [num_samples, num_stations, pred_len]

    Returns:
        list[dict]: 每个步长的指标字典列表
            每个字典包含: step, rmse, mae, r2, bias

    示例:
        >>> metrics = get_metrics_per_step(test_pred, test_label)
        >>> print(metrics[0])
        {'step': 1, 'rmse': 1.5234, 'mae': 1.1234, 'r2': 0.9234, 'bias': -0.0567}
    """
    # 获取预测步长数量
    pred_len = predict_epoch.shape[2]
    metrics_per_step = []

    # 为每个预测步长计算指标
    for step in range(pred_len):
        # 提取该步长的数据: [num_samples, num_stations]
        pred_step = predict_epoch[:, :, step]
        label_step = label_epoch[:, :, step]

        # 计算该步长的整体指标（所有样本、所有站点）
        rmse, mae, r2, bias = get_metric(pred_step, label_step)

        metrics_per_step.append(
            {
                "step": step + 1,  # 从1开始计数（第1步、第2步...）
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "bias": bias,
            }
        )

    return metrics_per_step


def get_extreme_metrics(
    predict_epoch,
    label_epoch,
    high_thresholds=[35],
    low_thresholds=None,
    threshold_array=None,
):
    """
    计算极端值监控指标（二分组：高于阈值 vs 低于阈值）

    Args:
        predict_epoch: 预测值数组 [num_samples, num_stations, pred_len]
        label_epoch: 真实值数组 [num_samples, num_stations, pred_len]
        high_thresholds: 高温阈值列表，仅当 threshold_array 为 None 时使用
        low_thresholds: 保留参数，不再使用
        threshold_array: 与 label_epoch 同形状的逐样本阈值数组。
                         若提供，则逐样本比较（动态阈值模式）。

    Returns:
        dict:
        {
            'high_temp': [{'threshold': T, 'sample_count': N, 'percentage': %, ...}],
            'low_temp':  [{'threshold': T, 'sample_count': N, 'percentage': %, ...}],
            'normal_temp': {'sample_count': N, 'percentage': %, 'range': '...'}
        }
    """
    pred_flat = predict_epoch.flatten()
    label_flat = label_epoch.flatten()
    total_samples = len(label_flat)

    result = {"high_temp": [], "low_temp": [], "normal_temp": {}}

    if threshold_array is not None:
        # 动态阈值模式：逐样本比较
        thr_flat = threshold_array.flatten()
        assert thr_flat.shape == label_flat.shape, (
            f"threshold_array shape {thr_flat.shape} != label shape {label_flat.shape}"
        )

        high_mask = label_flat >= thr_flat
        high_count = int(np.sum(high_mask))

        # 报告用中位数作为代表值
        report_threshold = float(np.median(threshold_array))

        if high_count > 0:
            pred_h = pred_flat[high_mask]
            label_h = label_flat[high_mask]
            thr_h = thr_flat[high_mask]

            rmse = np.sqrt(np.mean((pred_h - label_h) ** 2))
            mae = np.mean(np.abs(pred_h - label_h))
            bias = np.mean(pred_h - label_h)

            underestimate_rate = np.mean(pred_h < label_h) * 100
            overestimate_rate = np.mean(pred_h >= label_h) * 100

            hit_count = int(np.sum(pred_h >= thr_h))
            hit_rate = hit_count / high_count * 100
            miss_rate = 100 - hit_rate

            false_alarm_mask = (pred_flat >= thr_flat) & (label_flat < thr_flat)
            total_normal = total_samples - high_count
            false_alarm_rate = (
                np.sum(false_alarm_mask) / total_normal * 100
                if total_normal > 0
                else 0.0
            )

            result["high_temp"].append(
                {
                    "threshold": report_threshold,
                    "sample_count": high_count,
                    "percentage": high_count / total_samples * 100,
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias,
                    "underestimate_rate": underestimate_rate,
                    "overestimate_rate": overestimate_rate,
                    "hit_rate": hit_rate,
                    "false_alarm_rate": false_alarm_rate,
                    "miss_rate": miss_rate,
                }
            )

        # ── 低于阈值组 ──
        low_mask = label_flat < thr_flat
        low_count = int(np.sum(low_mask))

        result["normal_temp"] = {
            "sample_count": low_count,
            "percentage": low_count / total_samples * 100,
            "range": f"< 动态阈值 (中位数 {report_threshold:.2f}°C)",
        }

        if low_count > 0:
            pred_l = pred_flat[low_mask]
            label_l = label_flat[low_mask]
            thr_l = thr_flat[low_mask]

            # 低于阈值组：命中率=正确预测低于阈值，误报率=预测高于但实际低于
            hit_rate_l = np.mean(pred_l < thr_l) * 100
            false_alarm_rate_l = np.mean(pred_l >= thr_l) * 100

            result["low_temp"].append(
                {
                    "threshold": report_threshold,
                    "sample_count": low_count,
                    "percentage": low_count / total_samples * 100,
                    "rmse": np.sqrt(np.mean((pred_l - label_l) ** 2)),
                    "mae": np.mean(np.abs(pred_l - label_l)),
                    "bias": np.mean(pred_l - label_l),
                    "underestimate_rate": np.mean(pred_l < label_l) * 100,
                    "overestimate_rate": np.mean(pred_l >= label_l) * 100,
                    "hit_rate": hit_rate_l,
                    "false_alarm_rate": false_alarm_rate_l,
                    "miss_rate": 0.0,
                }
            )
    else:
        # 固定阈值模式
        threshold = sorted(high_thresholds, reverse=True)[0]

        high_mask = label_flat >= threshold
        high_count = int(np.sum(high_mask))

        if high_count > 0:
            pred_h = pred_flat[high_mask]
            label_h = label_flat[high_mask]

            rmse = np.sqrt(np.mean((pred_h - label_h) ** 2))
            mae = np.mean(np.abs(pred_h - label_h))
            bias = np.mean(pred_h - label_h)

            underestimate_rate = np.mean(pred_h < label_h) * 100
            overestimate_rate = np.mean(pred_h >= label_h) * 100

            hit_count = int(np.sum(pred_h >= threshold))
            hit_rate = hit_count / high_count * 100
            miss_rate = 100 - hit_rate

            false_alarm_mask = (pred_flat >= threshold) & (label_flat < threshold)
            total_normal = total_samples - high_count
            false_alarm_rate = (
                np.sum(false_alarm_mask) / total_normal * 100
                if total_normal > 0
                else 0.0
            )

            result["high_temp"].append(
                {
                    "threshold": threshold,
                    "sample_count": high_count,
                    "percentage": high_count / total_samples * 100,
                    "rmse": rmse,
                    "mae": mae,
                    "bias": bias,
                    "underestimate_rate": underestimate_rate,
                    "overestimate_rate": overestimate_rate,
                    "hit_rate": hit_rate,
                    "false_alarm_rate": false_alarm_rate,
                    "miss_rate": miss_rate,
                }
            )

        low_mask = label_flat < threshold
        low_count = int(np.sum(low_mask))

        result["normal_temp"] = {
            "sample_count": low_count,
            "percentage": low_count / total_samples * 100,
            "range": f"< {threshold}°C",
        }

        if low_count > 0:
            pred_l = pred_flat[low_mask]
            label_l = label_flat[low_mask]

            result["low_temp"].append(
                {
                    "threshold": threshold,
                    "sample_count": low_count,
                    "percentage": low_count / total_samples * 100,
                    "rmse": np.sqrt(np.mean((pred_l - label_l) ** 2)),
                    "mae": np.mean(np.abs(pred_l - label_l)),
                    "bias": np.mean(pred_l - label_l),
                    "underestimate_rate": np.mean(pred_l < label_l) * 100,
                    "overestimate_rate": np.mean(pred_l >= label_l) * 100,
                    "hit_rate": np.mean(pred_l < threshold) * 100,
                    "false_alarm_rate": 0.0,
                    "miss_rate": np.mean(pred_l >= threshold) * 100,
                }
            )

    return result


def get_extreme_metrics_per_step(
    predict_epoch,
    label_epoch,
    high_thresholds=[35],
    low_thresholds=None,
    threshold_array=None,
):
    """
    计算每个预测步长的极端值指标

    Args:
        predict_epoch: 预测值数组 [num_samples, num_stations, pred_len]
        label_epoch: 真实值数组 [num_samples, num_stations, pred_len]
        high_thresholds: 高温阈值列表
        low_thresholds: 低温阈值列表
        threshold_array: 逐样本阈值数组 [num_samples, num_stations, pred_len]

    Returns:
        list[dict]: 每个步长的极端值指标字典列表
    """
    pred_len = predict_epoch.shape[2]
    metrics_per_step = []

    for step in range(pred_len):
        # 提取该步长的数据: [num_samples, num_stations]
        pred_step = predict_epoch[:, :, step]
        label_step = label_epoch[:, :, step]

        # 提取该步长的阈值: [num_samples, num_stations]
        thr_step = None
        if threshold_array is not None:
            thr_step = threshold_array[:, :, step]

        # 扩展维度以匹配get_extreme_metrics的输入格式
        # [num_samples, num_stations] -> [num_samples, num_stations, 1]
        pred_step_3d = pred_step[:, :, np.newaxis]
        label_step_3d = label_step[:, :, np.newaxis]
        thr_step_3d = thr_step[:, :, np.newaxis] if thr_step is not None else None

        # 计算该步长的极端值指标
        step_metrics = get_extreme_metrics(
            pred_step_3d,
            label_step_3d,
            threshold_array=thr_step_3d,
        )

        step_metrics["step"] = step + 1
        metrics_per_step.append(step_metrics)

    return metrics_per_step


# 损失函数
criterion = nn.MSELoss()


def train(train_loader, model, optimizer, scheduler, config,
          arch_arg=None, epoch=None, total_epochs=None):
    """
    训练一个epoch

    Args:
        train_loader: 训练数据加载器
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        config: 配置对象
        arch_arg: 架构参数（可选，用于自回归解码器）
        epoch: 当前epoch（可选，用于TF衰减）
        total_epochs: 总epoch数（可选，用于TF衰减）

    Returns:
        train_loss: 训练损失（反标准化后的RMSE）
    """
    model.train()
    train_loss = 0

    use_ar = (
        arch_arg is not None
        and getattr(arch_arg, 'use_ar_decoder', False)
        and hasattr(model, 'ar_decoder')
    )

    if use_ar and epoch is not None and total_epochs is not None:
        model.ar_decoder.update_teacher_forcing_ratio(
            epoch, total_epochs)

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        # 修复：使用if/elif结构
        if not config.use_edge_attr:
            # 不使用边属性
            feature = batch[0].x.to(config.device)
            ta_label = batch[0].y.to(config.device)
            edge_index = batch[0].edge_index.to(config.device)
            if use_ar:
                ta_pred = model(feature, edge_index, ta_label=ta_label)
            else:
                ta_pred = model(feature, edge_index)
        elif config.use_edge_attr:
            # 使用边属性
            feature = batch[0].x.to(config.device)
            ta_label = batch[0].y.to(config.device)
            edge_index = batch[0].edge_index.to(config.device)
            edge_attr = batch[0].edge_attr.to(config.device)
            if use_ar:
                ta_pred = model(feature, edge_index, edge_attr,
                                ta_label=ta_label)
            else:
                ta_pred = model(feature, edge_index, edge_attr)

        loss = criterion(ta_pred, ta_label)

        # NaN检测
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n✗ 检测到异常损失 (Batch {batch_idx}):")
            print(f"  Loss: {loss.item()}")
            print(
                f"  Pred - min: {ta_pred.min().item():.4f}, max: {ta_pred.max().item():.4f}, "
                f"mean: {ta_pred.mean().item():.4f}"
            )
            print(
                f"  Pred - NaN: {torch.isnan(ta_pred).sum().item()}, "
                f"Inf: {torch.isinf(ta_pred).sum().item()}"
            )
            print(
                f"  Label - min: {ta_label.min().item():.4f}, max: {ta_label.max().item():.4f}, "
                f"mean: {ta_label.mean().item():.4f}"
            )
            print(
                f"  Label - NaN: {torch.isnan(ta_label).sum().item()}, "
                f"Inf: {torch.isinf(ta_label).sum().item()}"
            )
            raise ValueError("训练过程中出现NaN或Inf损失，训练终止")

        loss.backward()

        # 梯度裁剪：防止梯度爆炸（对LSTM尤其重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    # 反标准化：转换为实际温度的RMSE
    train_loss = np.sqrt(train_loss * (config.ta_std**2))

    # 更新学习率（非ReduceLROnPlateau调度器在这里更新）
    if scheduler is not None and not isinstance(
        scheduler, optim.lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step()

    return train_loss


def val(val_loader, model, config):
    """
    验证模型

    Args:
        val_loader: 验证数据加载器
        model: 模型
        config: 配置对象

    Returns:
        val_loss: 验证损失（反标准化后的RMSE）
        predict_epoch: 预测值数组 [num_samples, num_stations, pred_len]
        label_epoch: 真实值数组 [num_samples, num_stations, pred_len]
        time_epoch: 时间索引数组 [num_samples]
    """
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    val_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # 修复：使用if/elif结构
            if not config.use_edge_attr:
                feature = batch[0].x.to(config.device)
                ta_label = batch[0].y.to(config.device)
                edge_index = batch[0].edge_index.to(config.device)
                ta_pred = model(feature, edge_index)
            elif config.use_edge_attr:
                feature = batch[0].x.to(config.device)
                ta_label = batch[0].y.to(config.device)
                edge_index = batch[0].edge_index.to(config.device)
                edge_attr = batch[0].edge_attr.to(config.device)
                ta_pred = model(feature, edge_index, edge_attr)

            loss = criterion(ta_pred, ta_label)
            val_loss += loss.item()

            # 时间索引
            time_arr = batch[1]

            # 反标准化
            ta_pred_val = (
                ta_pred.cpu().detach().numpy() * config.ta_std + config.ta_mean
            )
            ta_label_val = (
                ta_label.cpu().detach().numpy() * config.ta_std + config.ta_mean
            )

            predict_list.append(ta_pred_val)
            label_list.append(ta_label_val)
            time_list.append(time_arr.cpu().detach().numpy())

    val_loss /= len(val_loader)
    val_loss = np.sqrt(val_loss * (config.ta_std**2))

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)

    # Reshape: [total_nodes, pred_len] → [num_samples, num_stations, pred_len]
    # PyG Batch将图节点展平,需要恢复为 [samples, stations, pred_len] 格式
    num_samples = len(time_epoch)
    num_stations = config.node_num
    pred_len = config.pred_len

    predict_epoch = predict_epoch.reshape(num_samples, num_stations, pred_len)
    label_epoch = label_epoch.reshape(num_samples, num_stations, pred_len)

    return val_loss, predict_epoch, label_epoch, time_epoch


def test(test_loader, model, config, arch_arg=None):
    """
    测试模型

    Args:
        test_loader: 测试数据加载器
        model: 模型
        config: 配置对象
        arch_arg: 架构参数对象 (用于可解释性配置)

    Returns:
        test_loss: 测试损失（反标准化后的RMSE）
        predict_epoch: 预测值数组 [num_samples, num_stations, pred_len]
        label_epoch: 真实值数组 [num_samples, num_stations, pred_len]
        time_epoch: 时间索引数组 [num_samples]
        attention_records: 注意力记录 (如果启用可解释性模式)
    """
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0

    # 可解释性模式
    interpretability_mode = False
    if arch_arg is not None:
        interpretability_mode = getattr(arch_arg, "interpretability_mode", False)

    attention_records = None
    if interpretability_mode:
        from interpretability import AttentionRecorder

        attention_records = AttentionRecorder()

    # 检查模型是否支持返回注意力
    supports_attention = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            feature = batch[0].x.to(config.device)
            ta_label = batch[0].y.to(config.device)
            edge_index = batch[0].edge_index.to(config.device)

            if config.use_edge_attr:
                edge_attr = batch[0].edge_attr.to(config.device)
            else:
                edge_attr = None

            # 前向传播
            if interpretability_mode and supports_attention:
                if edge_attr is not None:
                    ta_pred, attn_dict = model(
                        feature, edge_index, edge_attr, return_attention=True
                    )
                else:
                    ta_pred, attn_dict = model(
                        feature, edge_index, return_attention=True
                    )

                # 记录注意力
                for key, value in attn_dict.items():
                    attention_records.record(
                        key,
                        value,
                        metadata={
                            "batch": batch_idx,
                            "time": batch[1].item()
                            if batch[1].numel() == 1
                            else batch[1][0].item(),
                        },
                    )
            else:
                if edge_attr is not None:
                    ta_pred = model(feature, edge_index, edge_attr)
                else:
                    ta_pred = model(feature, edge_index)

            time_arr = batch[1]
            loss = criterion(ta_pred, ta_label)
            test_loss += loss.item()

            # 反标准化
            ta_pred_val = (
                ta_pred.cpu().detach().numpy() * config.ta_std + config.ta_mean
            )
            ta_label_val = (
                ta_label.cpu().detach().numpy() * config.ta_std + config.ta_mean
            )

            predict_list.append(ta_pred_val)
            label_list.append(ta_label_val)
            time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= len(test_loader)
    test_loss = np.sqrt(test_loss * (config.ta_std**2))

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)

    # Reshape: [total_nodes, pred_len] → [num_samples, num_stations, pred_len]
    # PyG Batch将图节点展平,需要恢复为 [samples, stations, pred_len] 格式
    num_samples = len(time_epoch)
    num_stations = config.node_num
    pred_len = config.pred_len

    predict_epoch = predict_epoch.reshape(num_samples, num_stations, pred_len)
    label_epoch = label_epoch.reshape(num_samples, num_stations, pred_len)

    # 保存注意力权重
    if interpretability_mode and attention_records is not None:
        save_path = getattr(arch_arg, "attention_save_path", "attention_weights/")
        from pathlib import Path

        Path(save_path).mkdir(parents=True, exist_ok=True)
        attention_records.save(f"{save_path}/attention_weights.npz")
        print(f"注意力权重已保存到: {save_path}/attention_weights.npz")

    if interpretability_mode:
        return test_loss, predict_epoch, label_epoch, time_epoch, attention_records

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_exp_info(config):
    """
    生成实验信息字符串

    Args:
        config: 配置对象

    Returns:
        exp_info: 实验信息字符串
    """
    exp_info = (
        "============== Train Info ==============\n"
        + f"Dataset: {config.dataset_num}\n"
        + f"Model: {config.exp_model}\n"
        + f"Train: {config.train_start} --> {config.train_end}\n"
        + f"Val: {config.val_start} --> {config.val_end}\n"
        + f"Test: {config.test_start} --> {config.test_end}\n"
        + f"Stations: {config.node_num}\n"
        + f"Input Dimension: {config.in_dim}\n"
        + f"Batch Size: {config.batch_size}\n"
        + f"Epochs: {config.epochs}\n"
        + f"History Length: {config.hist_len} days\n"
        + f"Prediction Length: {config.pred_len} days\n"
        + f"Learning Rate: {config.lr}\n"
        + f"Weight Decay: {config.weight_decay}\n"
        + f"Early Stop: {config.early_stop}\n"
        + f"Device: {config.device}\n"
        + "========================================\n"
    )
    return exp_info
