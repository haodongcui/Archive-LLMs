from torch.optim import Adam, lr_scheduler
from torch.nn import CrossEntropyLoss
# from torch.utils.data import DataLoader

from utils.data_loader import MyDataset
from utils.tokenizer import MyBPETokenizer
from models.transformer import MyTransformer
from config import *


# Initialize Tokenizer
tokenizer = MyBPETokenizer()
tokenizer.load(tokenizer_load_path)
vocab_size = len(tokenizer)

# Define dataset
# with open(train_data_path, 'r', encoding='utf-8') as f:
#     train_text = f.read()
# with open(valid_data_path, 'r', encoding='utf-8') as f:
#     valid_text = f.read()

# train_data = tokenizer.encode(train_text)
# valid_data = tokenizer.encode(valid_text)

with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokenized_text = tokenizer.encode(text)
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)
split_idx = int(len(tokenized_text)*0.8)
train_data = tokenized_text[:split_idx]
valid_data = tokenized_text[split_idx:]

def get_batch(status:str):
    data = train_data if status == 'train' else valid_data
    idx = torch.randint(0, len(data)-max_len, (batch_size,))
    x = torch.stack([data[i:i+max_len] for i in idx])
    y = torch.stack([data[i+1:i+1+max_len] for i in idx])
    return x, y

# Initialize Model
model = MyTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    max_len=max_len,
    num_heads=num_heads,
    ffn_hidden_dim=ffn_hidden_dim,
    ffn_dropout_rate=ffn_dropout_rate,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    batch_size=batch_size
).to(device)

optimizer = Adam(params=model.parameters(), lr=learning_rate)
criterion = CrossEntropyLoss(ignore_index=0)    # 为什么设置 ignore_index=0 呢
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Train
model.train()
model_save_path = 'checkpoints/model_{}.pt'
epoch_losss = 0
for epoch in range(epochs):

    # train status
    model.train()
    input_ids, output_ids = get_batch('train')
    logits = model(input_ids, output_ids)
    
    logits = logits.view(-1, logits.shape[-1])
    targets = output_ids.view(-1)
    loss = criterion(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # valid status
    model.eval()
    x_val, y_val = get_batch('valid')
    with torch.no_grad():
        logits = model(x_val, y_val)
        logits = logits.view(-1, logits.shape[-1])
        targets = y_val.view(-1)
        val_loss = criterion(logits, targets)
        # scheduler.step(loss)

    print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Valid Loss: {val_loss.item():.4f}')

    if (epoch+1) % 1000 == 0:
        torch.save(model.state_dict(), model_save_path.format(epoch+1))