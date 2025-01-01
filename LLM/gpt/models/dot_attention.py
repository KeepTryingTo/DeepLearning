"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-19:28
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import numpy as np

import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_v, d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        ##
        # 计算每个Q与K的分数，计算出来的大小是 [batch_size, n_heads, len_q, len_q]
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 把被mask的地方置为无限小，softmax之后基本就是0，也就对q不起作用
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # 注意力后的大小 [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, v)
        return context, attn