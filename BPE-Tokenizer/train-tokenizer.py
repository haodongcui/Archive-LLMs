from models.tokenizer import BPETokenizer


# parameters
vocab_size = 16384        # 2^15=32768, 2^14=16384
special_tokens = ['<|BOS|>', '<|EOS|>', '<|PAD|>']
save_path = f'outputs/tokenizer_{vocab_size}.json'


# load dataset
with open('dataset/train-cn.txt', 'r', encoding='utf-8') as f:
        text_cn = f.read()
with open('dataset/train-en.txt', 'r', encoding='utf-8') as f:
        text_en = f.read()
text = text_cn + text_en

# train
tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size, special_tokens)
tokenizer.save(save_path)