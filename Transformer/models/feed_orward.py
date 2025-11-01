from torch.nn import Module, Linear, Sequential, ReLU, Dropout

class MyFeedForward(Module):
    def __init__(self, d_model, ffn_hidden_dim, ffn_dropout_rate):
        super().__init__()
        hidden_dim = 4 * d_model if ffn_hidden_dim is None else ffn_hidden_dim
        self.ffn = Sequential(
            Linear(d_model, hidden_dim),
            ReLU(),
            Dropout(p=ffn_dropout_rate),
            Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.ffn(x)