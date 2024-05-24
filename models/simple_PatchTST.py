__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


from layers.RevIN import RevIN
from layers.TGTSF import text_encoder, text_temp_cross_block, positional_encoding, TS_encoder


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model

        input_len = configs.seq_len

        dropout = configs.dropout

        patch_len = configs.patch_len
        stride = configs.stride
        self.TS_encoder = TS_encoder(embedding_dim=d_model, layers=n_layers, num_heads=n_heads, dropout=dropout, patch_len=patch_len, stride=stride, causal=False, input_len=configs.seq_len)
        patch_num=(input_len-patch_len)//stride+1
        self.head = nn.Linear(patch_num*d_model, configs.pred_len)

    def forward(self, x, text, _, mask):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        x = self.TS_encoder(x)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
        x = self.head(x)
        return x*torch.sqrt(x_var) + x_mean
