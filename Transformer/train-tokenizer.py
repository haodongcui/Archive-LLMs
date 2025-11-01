from utils.tokenizer import MyBPETokenizer
from config import *

# Load Data
with open('data/tokenizer-train-cn.txt', 'r', encoding='utf-8') as f:
    text_cn = f.read()
with open('data/tokenizer-train-en.txt', 'r', encoding='utf-8') as f:
    text_en = f.read()
text = text_cn + text_en

for path in tokenizer_data_path_list:
    with open(path, 'r', encoding='utf-8') as f:
        text += f.read() + '\n'

# Train Tokenizer
tokenizer = MyBPETokenizer()
tokenizer.train(text, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.save(tokenizer_save_path)