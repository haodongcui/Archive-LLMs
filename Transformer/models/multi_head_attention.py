from torch.nn import Module, Linear
from models.scaled_dot_product_attention import MyScaledDotProductAttention

class MyMultiHeadAttention(Module):
    '''
    (len, d) -> (Q, K, V) -> (len, d)
    '''
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.attention = MyScaledDotProductAttention()
        
    def forward(self, Q, K, V, mask=None):
        Q, K, V = self.w_q(Q), self.w_k(K), self.w_v(V)
        Q, K, V = self._split_heads(Q), self._split_heads(K), self._split_heads(V)
        output = self.attention(Q, K, V, mask)
        output = self._concat_heads(output)
        output = self.w_o(output)
        return  output
    
    def _split_heads(self, tensor):
        B, L, D = tensor.size()
        N = self.num_heads
        D_H = D // N
        tensor = tensor.view(B, L, N, D_H).transpose(1, 2)
        return tensor
    
    def _concat_heads(self, tensor):
        B, N, L, D_H = tensor.size()
        D = N * D_H
        tensor = tensor.transpose(1, 2).reshape(B, L, D)
        return tensor