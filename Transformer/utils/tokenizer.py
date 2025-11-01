from tqdm import trange
import json

class BaseTokenizer:
    '''
    Basic functions for tokenizers
    '''
    def __init__(self):
        self.vocab = {}             # int -< str
        self.merges = {}            # (int, int) -> str
        self.special_stokens = {}   # str -> int
    
    def save(self, path):
        data = {
            'vocab_size': len(self.vocab),
            'special_stokens': self.special_stokens,
            'vocab': {},
            'merges': {},
        }
        # format to json
        for k, v in self.vocab.items():
            try: data['vocab'][k] = v.decode('utf-8')
            except: data['vocab'][k] = v.hex()
        for (p0, p1), merged_id in self.merges.items():
            data['merges'][f'{p0},{p1}'] = merged_id
        # save json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'Successfully save tokenizer to {path}, \nvocab_size: {len(self.vocab)}')

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.special_stokens = data['special_tokens']
        for k, v in data['vocab'].items():
            try: self.vocab[int(k)] = v.encode('utf-8')
            except: self.vocab[int(k)] = bytes.fromhex(v)
        for pair_str, merged_id in data['merges'].items():
            pair = tuple(int(i) for i in pair_str.split(','))
            self.merges[pair] = merged_id
        print(f'Successfully load tokenizer from {path} tokens,\nvocab size: {len(self.vocab)}')

    def __len__(self):
        return len(self.vocab)



class MyBPETokenizer(BaseTokenizer):
    '''
    BPE Tokenizer
    '''
    def __init__(self):
        super().__init__()
        self.vocab = {}             # int -< str
        self.merges = {}            # (int, int) -> str
        self.special_stokens = {}   # str -> int

    def train(self, text, vocab_size=5000, special_tokens=[]):
        assert vocab_size > 256 + len(special_tokens), "vocab_size too small."
        num_merges = vocab_size - 256 - len(special_tokens)

        self.vocab = {i:bytes([i]) for i in range(256)} # 基本字符集

        text_bytes = text.encode('utf-8')   # .encode(): str -> bytes
        ids = list(text_bytes)              # bytes -> List[ascii_int]

        for i in trange(num_merges):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            id = 256 + i
            self.vocab[id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            # self.vocab[pair[0]] -= stats[pair]
            # self.vocab[pair[1]] -= stats[pair]
            self.merges[pair] = id
            ids = self._merge_pair(ids, pair, id)

        self._add_special_tokens(special_tokens)
    
    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge_pair(self, ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _add_special_tokens(self, special_tokens):
        for token in special_tokens:
            id = len(self.vocab)
            self.vocab[id] = token.encode('utf-8')
            self.special_stokens[token] = id
    
    def encode(self, text):
        '''
        str -> list[int]
        '''
        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)
        while len(ids) > 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges: break
            ids = self._merge_pair(ids, pair, new_id=self.merges[pair])
        return ids
    
    def decode(self, ids):
        '''
        list[int] -> str
        '''
        text_bytes_list = [self.vocab[id] for id in ids]
        text_bytes = b''.join(text_bytes_list)
        text = text_bytes.decode('utf-8')
        return text