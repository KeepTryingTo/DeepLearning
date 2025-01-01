"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-19:56
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_pos, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_pos, d_model)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        # [seq_len] -> [batch_size, seq_len]
        pos = pos.unsqueeze(0).expand_as(inputs)
        return self.pos_embedding(pos)