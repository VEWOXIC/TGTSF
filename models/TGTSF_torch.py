__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


from layers.RevIN import RevIN
from layers.TGTSF_torch import text_encoder, text_temp_cross_block, positional_encoding, TS_encoder


class Model(nn.Module):
    def __init__(self, configs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        self.pred_len = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model

        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        self.patch_len = patch_len
        stride = configs.stride
        self.stride = stride
        padding_patch = configs.padding_patch
        
        self.revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        self.out_attn_weights = configs.out_attn_weights
        
        self.TS_encoder = TS_encoder(embedding_dim=d_model, layers=n_layers, num_heads=n_heads, dropout=dropout, patch_len=patch_len, stride=stride, causal=True, input_len=configs.seq_len)
        
        self.text_encoder = text_encoder(cross_layer=configs.cross_layers, self_layer=configs.self_layers, embedding_dim=configs.text_dim, num_heads=configs.n_heads, dropout=configs.dropout, pred_len = configs.pred_len, stride=stride)

        # position embedding for text
        # self.dropout = nn.Dropout(dropout)

        self.dropout = dropout

        q_len = context_window

        self.mixer = text_temp_cross_block(text_embedding_dim=configs.text_dim, temp_embedding_dim=d_model, num_heads=configs.n_heads, dropout=0.0, self_layer=configs.mixer_self_layers)

        patch_num = int((context_window - patch_len)/stride + 1)
        self.patch_num = patch_num
        self.total_length = patch_len*2 + (int((self.pred_len - patch_len)/stride + 1) - 1) * stride
        # if padding_patch == 'end': # can be modified to general case
        #     self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        #     patch_num += 1

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.individual = individual

        # assert stride == patch_len, 'stride should be equal to patch_len for token_decoder head'
        self.head = nn.Linear(d_model, patch_len)
    
    
    def forward(self, x, news, description, news_mask):           # x: [Batch, Input length, Channel] news: [Batch, l, news_num, text_dim] description: [b, l, c, d]
        
        if self.revin:
            x_mean = torch.mean(x, dim=1, keepdim=True)
            x = x - x_mean
            x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
            # print(x_var)
            x = x / torch.sqrt(x_var)

        x = self.TS_encoder(x)    # x: [bs x nvars x d_model x patch_num]

        t = self.text_encoder(news, description, news_mask) # t: [bs, l, nvars, d_model] # 添加positional embedding!

        x, mix_weights = self.mixer(t, x) # x: [bs, patch_num, nvars, d_model]

        x = self.head(x) # x: [bs, patch_num, nvars, patch_len]
        x = x.permute(0,2,1,3)
        # x = x.reshape(x.shape[0], x.shape[1], -1) # x: [bs, nvars, patch_num*patch_len]
        # x = x.permute(0,2,1) # x: [bs, patch_num*patch_len, nvars]

        B, C, N, L = x.shape

        output = torch.zeros(B, C, self.total_length).to(x.device)
        count = torch.zeros(B, C, self.total_length).to(x.device)
        for i in range(N):
            start = i* self.stride
            end = start + self.patch_len
            output[:, :, start:end] += x[:, :, i, :]
            count[:, :, start:end] += 1
        output = output[:, :, :self.pred_len] # [bs, nvars, pred_len]
        count = count[:, :, :self.pred_len]

        output = output / count

        x = output[:, :, :self.pred_len] # [bs, nvars, pred_len]
        # print(x.shape, output.shape, x_var.shape)
        x = x.permute(0, 2, 1)
        
        if self.revin:
            # denorm
            x= x * torch.sqrt(x_var) + x_mean

        if self.out_attn_weights:
            return x, mix_weights
        else:
            return x