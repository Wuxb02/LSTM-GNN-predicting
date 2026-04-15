"""
损失函数模块 - 用于改进夏季气温预测

包含的损失函数：
1. WeightedTrendMSELoss - 加权趋势损失（论文方法，推荐）

参考文献:
刘旭, 杨昊, 梁潇云, 等. 基于注意力机制与加权趋势损失的风速订正方法.
应用气象学报, 2025, 36(3): 316-327.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedTrendMSELoss(nn.Module):
    """
    🔥  自适应加权趋势损失函数

    适用场景:
        - 核心逻辑:
            1. 仅对高温(Heat)进行不对称惩罚 (漏报惩罚 >> 误报惩罚)
            2. 结合趋势约束 (Trend Constraint)

    优化说明:
        - 权重计算: 基于反标准化后的真实温度 (保持物理意义)
        - 梯度计算: 基于标准化后的数据 (防止梯度爆炸)
    """

    def __init__(
        self,
        alert_temp=35.0,
        c_under=4.0,
        c_over=2.0,
        delta=0.1,
        trend_weight=0,
        ta_mean=None,
        ta_std=None,
        threshold_map=None,
        use_station_day_threshold=False,
    ):
        super().__init__()
        self.alert_temp = alert_temp
        self.c_under = c_under
        self.c_over = c_over
        self.delta = delta
        self.trend_weight = trend_weight
        self.ta_mean = ta_mean
        self.ta_std = ta_std
        self.use_station_day_threshold = use_station_day_threshold

        if threshold_map is not None:
            self.register_buffer(
                "threshold_map", torch.as_tensor(threshold_map, dtype=torch.float32)
            )
        else:
            self.threshold_map = None

        # 检查必要的统计量
        if self.ta_mean is None or self.ta_std is None:
            raise ValueError(
                "针对广州数据，必须提供 ta_mean 和 ta_std 以正确还原物理温度进行判定"
            )

    def _compute_weights(self, pred_actual, label_actual, threshold):
        """
        计算高温关注权重。
        threshold 必须与 label_actual 形状相同（由 forward 保证）。

        两种加权情况：
          Case 1 真漏报(FN)：label >= T 且 pred < T → c_under 惩罚（最严重）
          Case 2 误报(FP)：label < T 且 pred >= T → c_over 惩罚
          其余情况（正常温度、高温命中、高温区间内低估）权重保持 1.0
        """
        # 断言形状匹配（_gather_dynamic_thresholds 已保证，标量阈值由 full_like 保证）
        assert threshold.shape == label_actual.shape, (
            f"threshold shape {threshold.shape} != label_actual shape {label_actual.shape}"
        )

        weights = torch.ones_like(label_actual)

        # Case 1：真正漏报(FN) — 实际高温但预测低于阈值 -> 最严重的错误
        under_mask = (label_actual >= threshold) & (pred_actual < threshold)
        if under_mask.any():
            diff = label_actual[under_mask] - threshold[under_mask]
            weights[under_mask] += self.c_under * (diff + self.delta)

        # Case 2：误报(FP) — 实际非高温但预测超阈值 -> 次要错误
        over_mask = (label_actual < threshold) & (pred_actual >= threshold)
        if over_mask.any():
            diff = pred_actual[over_mask] - threshold[over_mask]
            weights[over_mask] += self.c_over * (diff + self.delta)

        return weights

    def _compute_trend_loss(self, pred, label):
        """计算趋势损失 (基于标准化数据)"""
        if pred.shape[1] <= 1:
            return torch.tensor(0.0, device=pred.device)

        # 一阶差分: 捕捉升温/降温速率
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_label = label[:, 1:] - label[:, :-1]

        return F.mse_loss(diff_pred, diff_label)

    def _gather_dynamic_thresholds(self, doy_indices, pred_shape):
        """
        根据 doy_indices 从 threshold_map 中查出对应的阈值张量���

        Args:
            doy_indices: [batch_size, pred_len]
            pred_shape: pred_actual 的形状 [batch_size * num_stations, pred_len]

        Returns:
            threshold: 与 pred_actual 形状相同的阈值张量 [batch_size * num_stations, pred_len]
        """
        map_device = self.threshold_map.device
        num_stations = self.threshold_map.shape[1]

        # 将 doy_indices 移到 threshold_map 所在设备进行查表
        doy_on_map = doy_indices.to(map_device)

        # doy_on_map: [batch_size, pred_len] -> [batch_size, 1, pred_len]
        doy_expanded = doy_on_map.unsqueeze(1).expand(-1, num_stations, -1)

        # station_indices: [num_stations] -> [1, num_stations, 1]
        # -> 广播到 [batch_size, num_stations, pred_len]
        station_indices = torch.arange(num_stations, device=map_device).view(
            1, num_stations, 1
        )

        # 查表: [batch_size, num_stations, pred_len]
        threshold = self.threshold_map[doy_expanded, station_indices]

        # reshape 到 pred_shape，并移回 pred_actual 所在设备
        threshold = threshold.reshape(pred_shape).to(doy_indices.device)

        return threshold

    def forward(self, pred, label, doy_indices=None):
        """
        Args:
            pred: 模型输出的标准化预测值 [batch_nodes, pred_len]
            label: 标准化的真实标签 [batch_nodes, pred_len]
            doy_indices: 每个样本每个预测步对应的 doy 索引 [batch_size, pred_len]
        """
        # 1. 反标准化: 还原为摄氏度，用于判断是否超过阈值
        with torch.no_grad():
            pred_actual = pred.detach() * self.ta_std + self.ta_mean
            label_actual = label.detach() * self.ta_std + self.ta_mean

        # 2. 确定阈值
        if (
            self.use_station_day_threshold
            and doy_indices is not None
            and self.threshold_map is not None
        ):
            threshold = self._gather_dynamic_thresholds(doy_indices, pred_actual.shape)
        else:
            if self.use_station_day_threshold and self.threshold_map is not None:
                warnings.warn(
                    "站点-日内动态阈值已启用，但 doy_indices 为 None。"
                    "回退到固定阈值模式。请确保 DataLoader 正确传递 doy_indices。",
                    UserWarning,
                )
            threshold = torch.full_like(label_actual, self.alert_temp)

        # 3. 计算物理权重
        pixel_weights = self._compute_weights(pred_actual, label_actual, threshold)

        # 4. 计算 Loss (在标准化数值上���行，保证数值稳定性)
        weighted_mse = torch.mean(pixel_weights * (pred - label) ** 2)

        # Trend MSE (仅在 trend_weight > 0 时计算)
        if self.trend_weight > 0:
            trend_loss = self._compute_trend_loss(pred, label)
        else:
            trend_loss = torch.tensor(0.0, device=pred.device)

        # 5. 总损失
        total_loss = weighted_mse + self.trend_weight * trend_loss

        return total_loss
