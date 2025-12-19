"""
损失函数模块 - 用于改进夏季气温预测

包含的损失函数：
1. WeightedTrendMSELoss - 加权趋势损失（论文方法，推荐）

参考文献:
刘旭, 杨昊, 梁潇云, 等. 基于注意力机制与加权趋势损失的风速订正方法.
应用气象学报, 2025, 36(3): 316-327.
"""

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

    def __init__(self, 
                 alert_temp=35.0,       # 默认高温阈值 (广州常选35度或37度)
                 c_under=3.0,           # 漏报高温的惩罚系数 (建议设大，因为漏报后果严重)
                 c_over=1.0,            # 误报高温的惩罚系数
                 delta=0.1,             # 缓冲项
                 trend_weight=0.5,      # 趋势项权重 (alpha)
                 ta_mean=None,          # [必须] 训练集温度均值
                 ta_std=None):          # [必须] 训练集温度标准差
        super().__init__()
        self.alert_temp = alert_temp
        self.c_under = c_under
        self.c_over = c_over
        self.delta = delta
        
        self.trend_weight = trend_weight
        self.ta_mean = ta_mean
        self.ta_std = ta_std

        # 检查必要的统计量
        if self.ta_mean is None or self.ta_std is None:
            raise ValueError("针对广州数据，必须提供 ta_mean 和 ta_std 以正确还原物理温度进行判定")


    def _compute_weights(self, pred_actual, label_actual, threshold):
        """
        计算高温关注权重 (广州模式: 只关注高温)
        """
        weights = torch.ones_like(label_actual)
        
        # 1. 漏报高温 (实际 >= 阈值, 但预测值 < 实际值) -> ⚠️ 最严重的错误
        # 逻辑: 实际是38度，你报了34度，不仅数值有误，而且漏掉了高温信号
        under_mask = (label_actual >= threshold) & (pred_actual < label_actual)
        if under_mask.any():
            diff = label_actual[under_mask] - threshold
            weights[under_mask] += self.c_under * (diff + self.delta)

        # 2. 误报高温 (实际 < 阈值, 但预测值 >= 阈值) -> ⚠️ 次要错误
        # 逻辑: 实际33度，你报了36度。虽然报高了，但至少起到了警示作用。
        # 使用 detach() 确保我们不通过降低权重来“作弊”
        over_mask = (label_actual < threshold) & (pred_actual >= threshold)
        if over_mask.any():
            diff = pred_actual[over_mask].detach() - threshold
            weights[over_mask] += self.c_over * (diff + self.delta)

        # 3. 正确命中高温 (实际 >= 阈值, 且 预测值 >= 实际值) -> ✅ 保持高关注
        # 逻辑: 实际38度，你报了39度。虽然有误差，但正确捕捉了高温事件。
        valid_high_mask = (label_actual >= threshold) & (pred_actual >= label_actual)
        if valid_high_mask.any():
            diff = label_actual[valid_high_mask] - threshold
            weights[valid_high_mask] += 1 * (diff + self.delta)

        return weights

    def _compute_trend_loss(self, pred, label):
        """计算趋势损失 (基于标准化数据)"""
        if pred.shape[1] <= 1:
            return 0.0
        
        # 一阶差分: 捕捉升温/降温速率
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_label = label[:, 1:] - label[:, :-1]
        
        return F.mse_loss(diff_pred, diff_label)

    def forward(self, pred, label):
        """
        Args:
            pred: 模型输出的标准化预测值 (Normalized)
            label: 标准化的真实标签 (Normalized)
        """
        # 1. 反标准化: 还原为摄氏度，用于判断是否超过 35°C 阈值
        # 使用 no_grad 节省显存，只用于生成权重系数
        with torch.no_grad():
            pred_actual = pred.detach() * self.ta_std + self.ta_mean
            label_actual = label.detach() * self.ta_std + self.ta_mean
        
        # 2. 确定阈值 (固定 35/37 或 自适应)
        current_threshold = self.alert_temp
        
        # 3. 计算物理权重
        pixel_weights = self._compute_weights(pred_actual, label_actual, current_threshold)
        
        # 4. 计算 Loss (在标准化数值上进行，保证数值稳定性)
        # Weighted MSE
        weighted_mse = torch.mean(pixel_weights * (pred - label) ** 2)
        
        # Trend MSE
        trend_loss = self._compute_trend_loss(pred, label)
        
        # 5. 总损失
        total_loss = weighted_mse + self.trend_weight * trend_loss
        
        return total_loss



