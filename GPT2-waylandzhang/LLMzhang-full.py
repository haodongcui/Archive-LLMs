import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tiktoken
import os
import math
import requests


# Hyperparameters
batch_size = 4      # x个批次并行训练
context_size = 16   # 输入序列长度
max_len = 5000
d_model = 64
num_heads = 4       # 头数
num_layers = 4      # Transformer Block 数量
learning_rate = 1e-3
epochs = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load training data
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt'
    response = requests.get(url)
    with open('sales_textbook.txt', 'w') as f:
        f.write(response.text)

with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize text by using tiktoken (OpenAI's tokenizer, same as GPT-3)
enc = tiktoken.get_encoding("o200k_base")   # tokenize text to tokens
tokenized_text = enc.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)
vocab_size = max(tokenized_text) + 1        # Token vacabulary size

# Create training data
split_idx = int(0.9 * len(tokenized_text))
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]


# ----------------- Model components -----------------

# Token embedding
class TokenEmbedding(nn.Module):
    def __init__(self):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=d_model)

    def forward(self, x):
        return self.embedding(x)

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, batch_size=batch_size, d_model=d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.expand(batch_size, -1, -1)                    # (max_len, d_model) -> (1, max_len, d_model)
        self.register_buffer('pe', pe)          # 将 pe 注册为缓冲区，模型训练时不会被更新

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]          # (batch_size, seq_len, d_model)
        return x

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model):
        super(ScaledDotProductAttention, self).__init__()
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        # self.register_buffer('tril', torch.tril(torch.ones((self.seq_len, self.seq_len))))  # Lower triangular mask

    def forward(self, x):                       # (batch_size, seq_len, head_size)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        seq_len = x.size(1)                              # 自适应长度的 mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))  # Create mask

        weights = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = self.softmax(weights)
        output = weights @ V
        return output

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        attention = ScaledDotProductAttention(d_model=self.head_size)
        self.heads = nn.ModuleList([attention for _ in range(self.num_heads)])
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_size)

        output = [self.heads[i](x[:, i, :, :]) for i in range(self.num_heads)] # list of (batch_size, seq_len, head_size)
        output = torch.cat(output, dim=-1) # (batch_size, seq_len, d_model)
        output = self.w_o(output) # (batch_size, seq_len, d_model)
        return output

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        hidden_dim = d_model * 4
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        residual = x
        x = self.multi_head_attention(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size):
        super(Transformer, self).__init__()
        self.token_embedding = TokenEmbedding()
        self.positional_encoding = PositionalEncoding(batch_size=batch_size, d_model=d_model, max_len=max_len)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x_idx):
        x = self.token_embedding(x_idx)
        x = self.positional_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
            x = self.layer_norm(x)
        logits = self.fc_out(x)
        return logits
    
    def generate(self, x_idx, max_new_tokens=10):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(x_idx)  # (batch_size, seq_len, vocab_size)
            logits = logits[:, -1, :]     # (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (batch_size, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            x_idx = torch.cat([x_idx, next_token], dim=1)  # Append to sequence
        return x_idx



# ----------------- Processing -----------------

# Generate training and validating batches
def get_batch(status: str):
    data = train_data if status == 'train' else val_data
    idx = torch.randint(0, len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in idx])
    y = torch.stack([data[i+1:i+context_size+1] for i in idx])
    return x, y

model = Transformer(d_model=d_model, num_heads=num_heads, num_layers=num_layers, vocab_size=vocab_size).to(device)
optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0) # 为什么设置 ignore_index=0 呢


for epoch in range(epochs):
    
    # train status
    model.train()
    x, y = get_batch('train')
    logits = model(x)  # (batch_size, context_size, vocab_size)
    
    logits = logits.view(-1, vocab_size)
    targets = y.view(-1)
    loss = criterion(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # val status
    model.eval()
    x_val, y_val = get_batch('val')
    with torch.no_grad():
        logits = model(x_val)  # (batch_size, context_size, vocab_size)
        logits = logits.view(-1, vocab_size)
        targets = y_val.view(-1)
        val_loss = criterion(logits, targets)

    print(f'Epoch: {epoch}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')



# Generate text
model.eval()
x = get_batch('val')[0]  # (batch_size, context_size)[0]
gen_tokens = model.generate(x, max_new_tokens=20)
gen_text = [enc.decode([token]) for token in gen_tokens[0].tolist()]
sentence = ''.join(gen_text)
print('GEN:\n', gen_text)
print('GEN:\n', sentence)

