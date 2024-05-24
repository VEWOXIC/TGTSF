import torch 
from torch import nn
# from sentence_transformers import SentenceTransformer
import numpy as np
from layers.PatchTST_layers import *
from einops import rearrange, repeat
import torch.nn.functional as F
import torch.nn.init as init



def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    
    # reshape W_pos as q_len, 1, d_model
    W_pos = W_pos.unsqueeze(1)
    return nn.Parameter(W_pos, requires_grad=learn_pe)

def initialize_weights(m):
    """Custom weight initialization for linear layers."""
    if isinstance(m, nn.Linear):
        print('find linear, initialize it')
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


class text_temp_cross_block(nn.Module):
    def __init__(self, text_embedding_dim=384, temp_embedding_dim=128 , num_heads=8, self_layer=3, cross_layer=3, dropout=0.0):
        super(text_temp_cross_block, self).__init__()
        # self.att=nn.MultiheadAttention(temp_embedding_dim, num_heads, dropout, True, batch_first=True, kdim=text_embedding_dim, vdim=text_embedding_dim) # automatic align the dimension of key and value by reshaping the projection matrix
        cross_encoder_layer = nn.TransformerDecoderLayer(d_model=temp_embedding_dim,
                                                    nhead=num_heads,
                                                    dropout=dropout,
                                                    dim_feedforward=temp_embedding_dim*4,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True)
        norm_layer = nn.LayerNorm(temp_embedding_dim, eps=1e-5)
        self.cross_encoder = nn.TransformerDecoder(cross_encoder_layer, cross_layer, norm=norm_layer)

        self_encoder_layer = nn.TransformerDecoderLayer(d_model=text_embedding_dim,
                                                    nhead=num_heads,
                                                    dropout=dropout,
                                                    dim_feedforward=text_embedding_dim*4,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True)
        self_norm_layer = nn.LayerNorm(text_embedding_dim, eps=1e-5)
        self.self_encoder = nn.TransformerDecoder(self_encoder_layer, self_layer, norm=self_norm_layer)

        for p in self.cross_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.self_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, text_emb, temp_emb):
        # text_emb:         [b, l, c, d]    Embed the news according to the description of each timestamp and each channel
        # temp_emb:         [b, n, c, d]    embedding of temporal transformers

        B, L_text, C, D_text = text_emb.shape
        _, L_temp, _, D_temp = temp_emb.shape

        # print(D_text, D_temp)

        # permute the text_emb and temp_emb
        text_emb=text_emb.permute(0, 2, 1, 3)    # [b, c, l, d]
        temp_emb=temp_emb.permute(0, 2, 1, 3)    # [b, c, n, d]

        # reshape the text_emb and temp_emb
        text_emb=text_emb.reshape(B*C, L_text, D_text)
        # print(temp_emb.shape)
        temp_emb=temp_emb.reshape(B*C, L_temp, D_temp)

        # cross attention
        # result = self.cross_encoder(tgt=temp_emb, memory=text_emb) # [b*c, l, d]
        result = self.cross_encoder(tgt=text_emb, memory=temp_emb) # [b*c, l, d] text as query

        result = self.self_encoder(tgt=result, memory=result) # [b*c, l, d]

        # reshape the result
        # attn_weights = result[1]
        result=result.view(B, C, -1, D_temp)

        # result = result[:, :, -L_temp:, :]

        # permute the result
        result=result.permute(0, 2, 1, 3)

        return result, 0
    
        

class text_encoder(nn.Module):
    def __init__(self, cross_layer=2, self_layer=3, embedding_dim=384, num_heads=8, dropout=0.0, pred_len=96, stride=24):
        super(text_encoder, self).__init__()
        
        cross_encoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim,
                                                    nhead=num_heads,
                                                    dropout=dropout,
                                                    dim_feedforward=embedding_dim*4,
                                                    activation='gelu',
                                                    batch_first=True,
                                                    norm_first=True)
        
        encoder_norm = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.cross_encoder = nn.TransformerDecoder(cross_encoder_layer, cross_layer, norm=encoder_norm)

        # self_encoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim,
        #                                             nhead=num_heads,
        #                                             dropout=dropout,
        #                                             dim_feedforward=embedding_dim*4,
        #                                             activation='gelu',
        #                                             batch_first=True,
        #                                             norm_first=True)
        # self_norm_layer = nn.LayerNorm(embedding_dim, eps=1e-5)
        # self.self_encoder = nn.TransformerDecoder(self_encoder_layer, self_layer, norm=self_norm_layer)

        #self.W_pos = positional_encoding('zeros', True, (pred_len-stride)//stride+1, embedding_dim)
        self.W_pos = positional_encoding('zeros', True, int(np.ceil(pred_len/stride)), embedding_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        for p in self.cross_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, news_emb, description_emb, news_mask):
        # news_emb:        [b, l, n, d]    As the key and value, the embedding of news on the segment, each day may have different number of news pad it to the same length n (in the dataloader)
        # description_emb: [b, l, c, d]    As the query, the embedding of description of each channels
        # N and C are order invariant, no positional embedding needed

        # cross layers
        B, L, C, D = description_emb.shape

        # reshape the news_emb
        news_emb=news_emb.view(B*L, news_emb.shape[2], D) # [b*l, n, d]
        news_mask=news_mask.view(B*L, news_mask.shape[2]) # [b*l, n]

        # reshape the description_emb
        description_emb=description_emb.view(B*L, description_emb.shape[2], D) # [b*l, c, d]

        text_emb=description_emb

        text_emb=self.cross_encoder(tgt=text_emb, memory=news_emb, memory_key_padding_mask=news_mask)

        # reshape the text_emb
        text_emb=text_emb.view(B, L, C, D)

        # add positional encoding
        x = rearrange(text_emb, 'b l c d -> (b c) l d', b=B, c=C)
        x = x + self.W_pos.permute(1, 0, 2)

        x = self.dropout_layer(x)

        # x = self.self_encoder(tgt=x, memory=x)

        text_emb = rearrange(x, '(b c) l d -> b l c d', b=B, c=C)

        return text_emb
    

class TS_encoder(nn.Module): # 

    def __init__(self, embedding_dim=384, num_heads=8, dropout=0.0, layers=3,
                 patch_len=8, stride=8, causal=False, input_len=60):
        super(TS_encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layers = layers
        self.patch_len = patch_len
        self.stride = stride
        self.causal = causal

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, dim_feedforward=embedding_dim*4, activation='gelu', batch_first=True, norm_first=True)
        norm_layer = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.attentions = nn.TransformerEncoder(encoder_layer, layers, norm=norm_layer)
        self.input_encoder = nn.Linear(patch_len, embedding_dim)

        self.dropout_layer = nn.Dropout(dropout)

        for p in self.attentions.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


        self.W_pos = positional_encoding('zeros', True, (input_len-patch_len)//stride+1, embedding_dim)

    def forward(self, x):
        B, L, C = x.shape
        x = x.permute(0, 2, 1)
        x = x.unfold(-1, self.patch_len, self.stride) # [B, C, patch_num, patch_len]

        x = self.input_encoder(x) # [B, C, patch_num, embedding_dim]

        x = rearrange(x, 'b c n d -> (b c) n d', b=B, c=C)

        # print(x.shape, self.W_pos.shape)

        x = x + self.W_pos.permute(1, 0, 2)

        x = self.dropout_layer(x)

        # attn_mask = torch.triu(torch.full((x.shape[1], x.shape[1]), -np.inf), 1).to(x.device)

        x = self.attentions(x)

        x = rearrange(x, '(b c) n d -> b n c d', b=B, c=C)

        return x


class spectrum_encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass