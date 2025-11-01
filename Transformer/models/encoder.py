from torch.nn import Module, LayerNorm, ModuleList

from models.embedding import MyEmbedding
from models.multi_head_attention import MyMultiHeadAttention
from models.feed_orward import MyFeedForward


# Encoder Layer
class MyEncoderLayer(Module):
    def __init__(self, d_model, num_heads, ffn_hidden_dim, ffn_dropout_rate):
        super().__init__()
        self.multi_head_attention = MyMultiHeadAttention(d_model, num_heads)
        self.feed_forward = MyFeedForward(d_model, ffn_hidden_dim, ffn_dropout_rate)
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        _x = x
        x = _x + self.multi_head_attention(Q=x, K=x, V=x, mask=mask)
        x = self.norm(x + _x)
        _x = x
        x = self.feed_forward(x)
        x = self.norm(x + _x)
        return x

# Encoder Module
class MyEncoder(Module):
    '''
    id_list -> (seq_len, d_model)
    '''
    def __init__(self,
                vocab_size,
                d_model,
                max_len,
                num_heads,
                ffn_hidden_dim,
                ffn_dropout_rate,
                num_encoder_layers,
                batch_size
                ):
        super().__init__()
        self.embedding = MyEmbedding(vocab_size, d_model, max_len, batch_size)
        self.encoder_layers = ModuleList(
            [MyEncoderLayer(d_model, num_heads, ffn_hidden_dim, ffn_dropout_rate)
            for _ in range(num_encoder_layers)]
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x