"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-20:00
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from torch import nn
from models.decoder import Decoder

class GPTModel(nn.Module):
    def __init__(self,
                 d_model, n_heads,
                 d_ff, d_k,
                 d_v, vocab_size,
                 max_pos, n_layers,
                 device
                 ):
        super(GPTModel, self).__init__()
        # 解码器
        self.decoder = Decoder(d_model, n_heads,
                               d_ff, d_k,
                               d_v, vocab_size,
                               max_pos, n_layers,
                               device)
        # 映射为词表大小
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, attention_mask=None):
        ##
        # inputs: [batch_size, seq_len]
        ##
        # outputs: [batch_size, seq_len, d_model]
        # self_attns: [n_layers, batch_size, n_heads, seq_len, seq_len]
        outputs, self_attns = self.decoder(inputs, attention_mask)
        # [batch_size, seq_len, vocab_size]
        logits = self.projection(outputs)
        return logits.view(-1, logits.size(-1)), self_attns

