from models.tokenizer import BPETokenizer

# parameters
load_path = f'outputs/tokenizer_16384.json'


# load tokenizer
tokenizer = BPETokenizer()
tokenizer.load(load_path)

# test
text = '''
<|BOS|>system
you are a helper assistant<|EOS|>
<|BOS|>user
今天的天气怎么样<|EOS|>
<|BOS|>assistant
今天天气很好, 嘿嘿, Good Weather!<EOS>'''
print(text)

ids = tokenizer.encode(text)
s = tokenizer.decode(ids)
print(s)