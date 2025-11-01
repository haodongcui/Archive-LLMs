from torch.nn import Module
import torch

from models.encoder import MyEncoder
from models.decoder import MyDecoder

class MyTransformer(Module):
    '''
    (input_ids, output_ids) -> predicted probability list
    '''
    def __init__(self,
                vocab_size,
                d_model,
                max_len,
                num_heads,
                ffn_hidden_dim,
                ffn_dropout_rate,
                num_encoder_layers,
                num_decoder_layers,
                batch_size,
                ):
        super().__init__()
        self.encoder = MyEncoder(
            vocab_size,
            d_model,
            max_len,
            num_heads,
            ffn_hidden_dim,
            ffn_dropout_rate,
            num_encoder_layers,
            batch_size
        )
        self.decoder = MyDecoder(
            vocab_size,
            d_model,
            max_len,
            num_heads,
            ffn_hidden_dim,
            ffn_dropout_rate,
            num_decoder_layers,
            batch_size
        )

    def forward(self, input_ids, output_ids):
        self_mask = self._get_self_mask(output_ids)
        cross_mask = self._get_cross_mask(input_ids, output_ids)
        enc_x = self.encoder(input_ids)
        output = self.decoder(output_ids, enc_x, self_mask, cross_mask)
        return output
    
    def _get_self_mask(self, output_ids):
        seq_len = output_ids.size(1)    # (b, len)
        mask = torch.tril(
            torch.ones((seq_len, seq_len), device=output_ids.device)
        ).bool()
        return mask.unsqueeze(0)
    
    def _get_cross_mask(self, input_ids, output_ids):
        input_len = input_ids.size(1)
        output_len = output_ids.size(1)
        mask = torch.zeros((input_len, output_len), device=input_ids.device).bool()
        return mask.unsqueeze(0)