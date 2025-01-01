"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-19:55
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import numpy as np

import torch
from torch import nn
from models.multihead_attention import MultiHeadAttention
from models.mlp import PoswiseFeedForwardNet
from models.position import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v):
        super(DecoderLayer, self).__init__()
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # 前馈神经网络层
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len, d_model]
        # attention_mask: [batch_size, seq_len, seq_len]
        ##
        # outputs: [batch_size, seq_len, d_model]
        # self_attn: [batch_size, n_heads, seq_len, seq_len]
        outputs, self_attn = self.attention(inputs, inputs, inputs, attention_mask)
        # [batch_size, seq_len, d_model]
        outputs = self.pos_ffn(outputs)
        return outputs, self_attn

"""
对于 Transformer Decoder 结构，模型在解码时应该是自回归的，每次都是基于之前的信息预测下一个Token，
这意味着在生成序列的第 i 个元素时，模型只能看到位置 i 之前的信息。因此在训练时需要进行遮盖，
防止模型看到未来的信息，遮盖的操作也非常简单，可以构建一个上三角掩码器。
原始注意力分数矩阵（无掩码）:
[[q1k1, q1k2, q1k3, q1k4],
 [q2k1, q2k2, q3k3, q3k4],
 [q3k1, q3k2, q3k3, q3k4],
 [q4k1, q4k2, q4k3, q4k4]]

上三角掩码器:
[[0, 1, 1, 1],
 [0, 0, 1, 1],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]

应用掩码后的分数矩阵:
[[q1k1, -inf, -inf, -inf],
 [q2k1, q2k2, -inf, -inf],
 [q3k1, q3k2, q3k3, -inf],
 [q4k1, q4k2, q4k3, q4k4]]
"""

def get_attn_subsequence_mask(seq, device):
    # 注意力分数的大小是 [batch_size, n_heads, len_seq, len_seq]
    # 所以这里要生成 [batch_size, len_seq, len_seq] 大小
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 生成一个上三角矩阵
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask

def get_attn_pad_mask(attention_mask):
    batch_size, len_seq = attention_mask.size()
    attention_mask = attention_mask.data.eq(0).unsqueeze(1)
    # 注意力分数的大小是 [batch_size, n_heads, len_q, len_q]
    # 所以这里要转换成 [batch_size, len_seq, len_seq] 大小
    return attention_mask.expand(batch_size, len_seq, len_seq)


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(Decoder, self).__init__()
        self.device = device
        # 将Token转为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_pos, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])

    def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len]
        ##
        # [batch_size, seq_len, d_model]
        outputs = self.embedding(inputs) + self.pos_encoding(inputs)
        # 上三角掩码，防止看到未来的信息， [batch_size, seq_len, seq_len]
        subsequence_mask = get_attn_subsequence_mask(inputs, self.device)
        if attention_mask is not None:
            # pad掩码 [batch_size, seq_len, seq_len]
            attention_mask = get_attn_pad_mask(attention_mask)
            # [batch_size, seq_len, seq_len]
            attention_mask = torch.gt((attention_mask + subsequence_mask), 0)
        else:
            attention_mask = subsequence_mask.bool()
        # 计算每一层的结果
        self_attns = []
        for layer in self.layers:
            # outputs: [batch_size, seq_len, d_model],
            # self_attn: [batch_size, n_heads, seq_len, seq_len],
            outputs, self_attn = layer(outputs, attention_mask)
            self_attns.append(self_attn)
        return outputs, self_attns