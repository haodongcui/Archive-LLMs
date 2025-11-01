from torch.nn import Module, Softmax
import math

class MyScaledDotProductAttention(Module):
    '''
    attention is weights, output is weighted sum, output represents the information of V
    '''
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=-1)  # the last dim is matmul with V
    
    def forward(self, Q, K, V, mask=None):
        d_k = V.size(-1)
        attention = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # (b,len,d) @ (b,len,d)^T / sqrt(d) -> (b,len,len)

        if mask is not None:
            attention = attention.masked_fill(mask==0, float('-1e9')) # pos if 'True' will be replaced by '-inf'

        attention = self.softmax(attention)  # (b,len,len) -> (b,len,len_s)
        output = attention @ V     # (b,len,len_s) @ (b,len,d) -> (b,len,d)
        return output