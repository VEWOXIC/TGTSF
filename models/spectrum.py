
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import TSTiEncoder, Flatten_Head
from layers.PatchTST_layers import series_decomp
from layers.RevIN import RevIN
from layers.TGTSF import text_encoder, text_temp_cross_block, positional_encoding
import torch_dct as dct

class patch_dct(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0,2,1).reshape(B*L, C)
        x = dct.dct(x, norm='ortho')
        x = x.reshape(B, L, N).permute(0,2,1)
        return x