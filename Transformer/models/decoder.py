from torch.nn import Module, LayerNorm, ModuleList, Linear, Softmax

from models.embedding import MyEmbedding
from models.multi_head_attention import MyMultiHeadAttention
from models.feed_orward import MyFeedForward


# Decoder Layer
class MyDecoderLayer(Module):
    def __init__(self, d_model, num_heads, ffn_hidden_dim, ffn_dropout_rate):
        super().__init__()
        self.masked_multi_head_attention = MyMultiHeadAttention(d_model, num_heads)
        self.cross_multi_head_attention = MyMultiHeadAttention(d_model, num_heads)
        self.feed_forward = MyFeedForward(d_model, ffn_hidden_dim, ffn_dropout_rate)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.layer_norm_3 = LayerNorm(d_model)

    def forward(self, dec_x, enc_x, self_mask, cross_mask ):
        _x = dec_x
        dec_x = self.masked_multi_head_attention(Q=dec_x, K=dec_x, V=dec_x, mask=self_mask)
        dec_x = self.layer_norm_1(dec_x + _x)

        _x = dec_x
        x = self.cross_multi_head_attention(Q=dec_x, K=enc_x, V=enc_x, mask=cross_mask)
        x = self.layer_norm_2(x + _x)

        _x = x
        x = self.feed_forward(x)
        x = self.layer_norm_3(x + _x)
        return x

# Decoder Module
class MyDecoder(Module):
    '''
    (input_emb, output_ids, self_mask, cross_mask) -> (vocab_size)
    '''
    def __init__(self,
                vocab_size,
                d_model,
                max_len,
                num_heads,
                ffn_hidden_dim,
                ffn_dropout_rate,
                num_decoder_layers,
                batch_size
                ):
        super().__init__()
        self.embedding = MyEmbedding(vocab_size, d_model, max_len, batch_size)
        self.decoder_layers = ModuleList(
            [MyDecoderLayer(d_model, num_heads, ffn_hidden_dim, ffn_dropout_rate)
            for _ in range(num_decoder_layers)]
        )
        self.linear = Linear(d_model, vocab_size)
        self.softmax = Softmax(dim=-1)

    def forward(self, dec_ids, enc_x, self_mask, cross_mask):
        x = self.embedding(dec_ids)
        for layer in self.decoder_layers:
            x = layer(x, enc_x, self_mask, cross_mask)
        x = self.linear(x)
        x = self.softmax(x)
        return x