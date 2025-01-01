"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-19:48
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from torch import nn
from models.dot_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, seq_len, d_model]
        # k: [batch_size, seq_len, d_model]
        # v: [batch_size, seq_len, d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        ##
        # 记录原始值, 后续计算残差
        residual, batch_size = q, q.size(0)
        # 先映射 q、k、v, 然后后分头
        # q: [batch_size, n_heads, len_q, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k: [batch_size, n_heads, len_k, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v: [batch_size, n_heads, len_v(=len_k), d_v]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 点积注意力分数计算，  [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention(self.d_k)(q, k, v, attention_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 还原为原始大小
        output = self.fc(context)
        # LN + 残差计算
        return self.layernorm(output + residual), attn