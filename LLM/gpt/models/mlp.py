"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-19:54
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from torch import nn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        ##
        # inputs: [batch_size, seq_len, d_model]
        ##
        residual = inputs
        output = self.fc(inputs)
        # # LN + 残差计算, [batch_size, seq_len, d_model]
        return self.layernorm(output + residual)