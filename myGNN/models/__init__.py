"""
myGNN模型模块

包含所有GNN模型实现:
- GAT.py: 图注意力网络 + LSTM
- GSAGE.py: GraphSAGE + LSTM
- LSTM.py: LSTM基线模型
- GAT_Pure.py: 纯图注意力网络（无LSTM）
- GAT_SeparateEncoder.py: GAT + 分离式编码器（静态/动态分离）
- GSAGE_SeparateEncoder.py: GraphSAGE + 分离式编码器（静态/动态分离）
"""

from .GAT import GAT_LSTM
from .GSAGE import GSAGE_LSTM
from .LSTM import LSTM_direct
from .GAT_Pure import GAT_Pure
from .GAT_SeparateEncoder import GAT_SeparateEncoder
from .GSAGE_SeparateEncoder import GSAGE_SeparateEncoder


__all__ = [
    # 基础模型
    'GAT_LSTM',
    'GSAGE_LSTM',
    'LSTM_direct',
    'GAT_Pure',
    # 分离式编码模型
    'GAT_SeparateEncoder',
    'GSAGE_SeparateEncoder',
]
