import json
from tqdm import trange

class BPETokenizer:
    def __init__(self):
        self.merges = {}    # (int, int) -> int
        self.vocab = {}     # int -> bytes
        self.special_tokens = {}    # str -> int

    def _get_stats(self, ids):
        '''
        calculate pair frequency
        (int, int) -> int
        where int is ASCLL_int
        '''
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, id):
        '''
        merge pair with id in ids
        [..., int, int, ...] -> [..., int, ...]
        where int is ASCLL_int
        '''
        new_ids = []
        i = 0
        while i < len(ids):
            if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:  # check boundary, then merge pair to new id
                new_ids.append(id)
                i +=2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids


    def train(self, text, vocab_size=5000, special_tokens=[]):
        assert vocab_size - len(special_tokens) >= 256
        num_merges = vocab_size - len(special_tokens) - 256 

        # 基本字符集
        self.vocab = {i:bytes([i]) for i in range(256)}     # 256个基本字符? 为什么都跟数字有对应?
        
        # 获取语料最基本的 tokens, ASCLL_int格式
        text_bytes = text.encode('utf-8')     # .encode -> bytes, bytes存储格式是 ASCLL_int_list, 所以迭代展开是 ASCLL_int
        ids = list(text_bytes)  # list of ASCLL_int

        for i in trange(num_merges):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)    # type of pair is ASCLL_int_tuple
            id = 256 + i    # id must be out of ASCLL_int, i.e. id>255
            ids = self._merge(ids, pair, id)
            self.merges[pair] = id
            self.vocab[id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            # print(f'merge {i+1}/{num_merges}: {pair} -> {id}  |  counts {self.vocab[id]}: {stats[pair]}')
        self.add_special_tokens(special_tokens)

    def encode(self, text):
        '''
        str -> int list
        '''
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        while len(ids)>=2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))   # default is +inf
            if pair not in self.merges: break
            ids = self._merge(ids, pair, id=self.merges[pair])
        return ids
    
    def decode(self, ids):
        '''
        int list -> str
        '''
        text_bytes_list = [self.vocab[id] for id in ids]
        text_bytes = b''.join(text_bytes_list)
        text = text_bytes.decode('utf-8')
        return text
    
    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            id = len(self.vocab)    # len(self.vocab) varies with special_tokens adding into vocab
            self.special_tokens[token] = id
            self.vocab[id] = token.encode('utf-8')

    def save(self, path='tokenizer.json'):
        data = {
            'vocab_size':len(self.vocab),
            'special_tokens': self.special_tokens,
            'vocab': {},
            'merges': {},
        }
        # edit json format
        for k, v in self.vocab.items():
            try: data['vocab'][k] = v.decode('utf-8')
            except: data['vocab'][k] = v.hex()
        for (p0, p1), merge_id in self.merges.items():
            data['merges'][f'{p0},{p1}'] = merge_id
        # save json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'Successfully saved tokenizer to {path}')
    
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = {}
        self.merges = {}
        self.special_tokens = data['special_tokens']
        # load data from json
        for k, v in data['vocab'].items():
            try:
                self.vocab[int(k)] = v.encode('utf-8')
            except UnicodeEncodeError:
                self.vocab[int(k)] = bytes.fromhex(v)
        for pair_str, merge_id in data['merges'].items():
            pair = tuple(int(i) for i in pair_str.split(','))
            self.merges[pair] = merge_id
        print(f'Successfully loaded tokenizer from {path}')

    def __len__(self):
        return len(self.vocab)




if __name__ == '__main__':
    # load dataset
    with open('dataset/train-cn.txt', 'r', encoding='utf-8') as f:
        text_cn = f.read()
    with open('dataset/train-en.txt', 'r', encoding='utf-8') as f:
        text_en = f.read()
    text = text_cn + text_en

    # train
    tokenizer = BPETokenizer()
    tokenizer.train(text=text, vocab_size=5000)
    tokenizer.add_special_tokens(['<BOS>', '<EOS>', '<PAD>'])
    tokenizer.save('models/tokenizer_5000.json')


    # load
    tokenizer = BPETokenizer()
    tokenizer.load('models/tokenizer_5000.json')
    print('vocab_size:', len(tokenizer))

    # encode
    text = '<|BOS|>system\nyou are a helper assistant\n<|EOS|>\n<|BOS|>user\n今天的天气\n<|EOS|><|BOS|>assistant\n<EOS>'
    ids = tokenizer.encode(text)
    print('encode:', ids, tokenizer.decode(ids))

    # decode sample
    # ids = [256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267]
    # s = tokenizer.decode(ids)
    # print('decode:', s)