"""
搜索空间定义模块

定义三种预设搜索模式（Quick / Default / Comprehensive），
包含参数范围、采样类型和约束处理逻辑。
"""

from __future__ import annotations

from typing import Any, Dict

import optuna


# ============================================================
# 参数定义：名称 → (采样方法, 参数, 所属类别)
# ============================================================

# ---------- 类别 A：高敏感度（所有模式都搜索） ----------
CATEGORY_A: Dict[str, Dict[str, Any]] = {
    "lr": {
        "sampler": "log_uniform",
        "low": 1e-4,
        "high": 5e-3,
    },
    "weight_decay": {
        "sampler": "log_uniform",
        "low": 1e-5,
        "high": 1e-2,
    },
    "hid_dim": {
        "sampler": "categorical",
        "choices": [16, 32, 64, 128],
    },
    "GAT_layer": {
        "sampler": "int",
        "low": 1,
        "high": 3,
    },
    "heads": {
        "sampler": "int",
        "low": 1,
        "high": 8,
    },
    "intra_drop": {
        "sampler": "uniform",
        "low": 0.0,
        "high": 0.5,
    },
    "inter_drop": {
        "sampler": "uniform",
        "low": 0.0,
        "high": 0.5,
    },
    "fusion_num_heads": {
        "sampler": "categorical",
        "choices": [1, 2, 4, 8],
    },
    "c_under": {
        "sampler": "uniform",
        "low": 1.0,
        "high": 6.0,
    },
    "c_over": {
        "sampler": "uniform",
        "low": 0.5,
        "high": 3.0,
    },
}

# ---------- 类别 B：中等敏感度（Default+ 模式搜索） ----------
CATEGORY_B: Dict[str, Dict[str, Any]] = {
    "lstm_num_layers": {
        "sampler": "int",
        "low": 1,
        "high": 3,
    },
    "lstm_dropout": {
        "sampler": "uniform",
        "low": 0.0,
        "high": 0.4,
    },
    "MLP_layer": {
        "sampler": "int",
        "low": 1,
        "high": 3,
    },
    "batch_size": {
        "sampler": "categorical",
        "choices": [16, 32, 64],
    },
    "top_neighbors": {
        "sampler": "int",
        "low": 3,
        "high": 10,
    },
    "hist_len": {
        "sampler": "categorical",
        "choices": [7, 14, 21, 28],
    },
    "trend_weight": {
        "sampler": "uniform",
        "low": 0.0,
        "high": 0.5,
    },
}

# ---------- 类别 C：低敏感度（Comprehensive 模式搜索） ----------
CATEGORY_C: Dict[str, Dict[str, Any]] = {
    "AF": {
        "sampler": "categorical",
        "choices": ["ReLU", "GELU", "LeakyReLU"],
    },
    "optimizer": {
        "sampler": "categorical",
        "choices": ["Adam", "AdamW"],
    },
    "scheduler": {
        "sampler": "categorical",
        "choices": ["ReduceLROnPlateau", "CosineAnnealingLR", "StepLR", "None"],
    },
    "use_skip_connection": {
        "sampler": "categorical",
        "choices": [True, False],
    },
    "fusion_use_pre_ln": {
        "sampler": "categorical",
        "choices": [True, False],
    },
}

# ---------- 模式 → 参数集合 ----------
MODE_PARAMS: Dict[str, set] = {
    "quick": set(CATEGORY_A.keys()),
    "default": set(CATEGORY_A.keys()) | set(CATEGORY_B.keys()),
    "comprehensive": set(CATEGORY_A.keys())
    | set(CATEGORY_B.keys())
    | set(CATEGORY_C.keys()),
}

# ---------- 固定值（不搜索） ----------
FIXED_VALUES: Dict[str, Any] = {
    # 损失函数固定
    "use_station_day_threshold": True,
    "threshold_percentile": 90,
    # 架构固定
    "norm_type": "LayerNorm",
    "lstm_bidirectional": False,
}


def _suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    """根据 spec 从 trial 中采样一个参数值。"""
    sampler = spec["sampler"]
    if sampler == "log_uniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    elif sampler == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    elif sampler == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    elif sampler == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"未知采样器: {sampler}")


def _apply_constraints(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    对采样后的参数施加硬性约束。

    约束列表:
        1. heads 必须能整除 hid_dim
        2. fusion_num_heads 必须能整除 hid_dim
        3. lstm_dropout 仅在 lstm_num_layers > 1 时生效
        4. c_under >= c_over
    """
    p = dict(params)

    # 约束1: heads | hid_dim
    hid_dim = p["hid_dim"]
    valid_heads = [h for h in [1, 2, 4, 8] if hid_dim % h == 0]
    p["heads"] = valid_heads[
        min(range(len(valid_heads)), key=lambda i: abs(valid_heads[i] - p["heads"]))
    ]

    # 约束2: fusion_num_heads | hid_dim
    valid_fusion = [h for h in [1, 2, 4, 8] if hid_dim % h == 0]
    p["fusion_num_heads"] = valid_fusion[
        min(
            range(len(valid_fusion)),
            key=lambda i: abs(valid_fusion[i] - p["fusion_num_heads"]),
        )
    ]

    # 约束3: lstm_num_layers == 1 → lstm_dropout = 0
    if p.get("lstm_num_layers", 1) == 1:
        p["lstm_dropout"] = 0.0

    # 约束4: c_under >= c_over
    if p["c_under"] < p["c_over"]:
        p["c_under"], p["c_over"] = p["c_over"], p["c_under"]

    return p


class SearchSpaceFactory:
    """搜索空间工厂：根据模式名称返回可搜索的参数名集合和固定值。"""

    @staticmethod
    def get_param_names(mode: str) -> list:
        """返回指定模式下需要搜索的参数名列表。"""
        mode = mode.lower()
        if mode not in MODE_PARAMS:
            raise ValueError(f"未知模式: {mode}，可选: {list(MODE_PARAMS.keys())}")
        return sorted(MODE_PARAMS[mode])

    @staticmethod
    def suggest_all(trial: optuna.Trial, mode: str) -> Dict[str, Any]:
        """
        在 trial 中采样所有属于该模式的参数，并施加约束。

        Returns:
            约束处理后的参数字典。
        """
        param_names = SearchSpaceFactory.get_param_names(mode)
        all_specs = {**CATEGORY_A, **CATEGORY_B, **CATEGORY_C}

        raw_params: Dict[str, Any] = {}
        for name in param_names:
            if name not in all_specs:
                continue
            raw_params[name] = _suggest_param(trial, name, all_specs[name])

        # 施加约束
        constrained = _apply_constraints(raw_params)

        # 合并固定值
        constrained.update(FIXED_VALUES)
        return constrained

    @staticmethod
    def get_fixed_values() -> Dict[str, Any]:
        """返回不参与搜索的固定参数值。"""
        return dict(FIXED_VALUES)
