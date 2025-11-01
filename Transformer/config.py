import torch

# random seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer settings
vocab_size = 1000
special_tokens = ['<|BOS|>', '<|EOS|>', '<|PAD|>']
tokenizer_save_path = f'checkpoints/tokenizer_{vocab_size}.json'
tokenizer_load_path = f'checkpoints/tokenizer_16384.json'
tokenizer_data_path_list = [
    'data/tokenizer-train-cn.txt',
    'data/tokenizer-train-en.txt',
]

# model settings
d_model = 64
max_len = 100

num_encoder_layers = 2
num_decoder_layers = 2
num_heads = 4
ffn_hidden_dim = 128
ffn_dropout_rate = 0.1

# trainning settings
batch_size = 4
epochs = 10000
learning_rate = 1e-4
checkpoints_dir = 'checkpoints'

train_data_path = 'data/TinyStoriesV2-GPT4-train.txt'
valid_data_path = 'data/TinyStoriesV2-GPT4-valid.txt'