from torch.nn import Module, Parameter, init
import torch


# Input Embedding
class MyTokenEmbedding(Module):
    '''
    id_list -> emb_matrix[id_list]
    '''
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb_matrix=Parameter(torch.empty(num_embeddings, embedding_dim))
        # init.normal_(self.emb_matrix, mean=0, std=0.1)
        # init.xavier_uniform_(self.emb_matrix)
        init.kaiming_normal_(self.emb_matrix, mode='fan_in', nonlinearity='relu')
    
    def forward(self, ids):
        return self.emb_matrix[ids]


# Positional Embedding
class MyPositionalEmbedding(Module):
    '''
    tokens_matrix -> tokens_matrix + pos_matrix
    '''
    def __init__(self, max_len, d_model, batch_size):
        super().__init__()
        
        pos = torch.arange(0, max_len).unsqueeze(1)
        _2i = -2 * torch.arange(0, d_model, step=2).float()
        div = torch.exp( _2i/d_model * torch.log(torch.tensor(10000.0)) )

        pos_emb = torch.zeros(max_len, d_model)
        pos_emb[:, 0::2] = torch.sin(pos * div)
        pos_emb[:, 1::2] = torch.cos(pos * div)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        # self.pos_emb = pos_emb
        self.register_buffer('pos_emb', pos_emb)    # 缓存
    
    def forward(self, x):
        seq_len = x.size(1) # (batch_size, seq_len, d_model)
        x = x + self.pos_emb[:, :seq_len, :]
        return x


# Embedding Layer
class MyEmbedding(Module):
    '''
    id_list -> (seq_len, d_model)
    '''
    def __init__(self, vocab_size, d_model, max_len, batch_size):
        super().__init__()
        self.token_embedding = MyTokenEmbedding(vocab_size, d_model)
        self.positional_embedding = MyPositionalEmbedding(max_len, d_model, batch_size)
    
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        return x