

## BPE Tokenizer


### 参考资料
- [minibpe](https://github.com/karpathy/minbpe)
- [B站视频 bpe-tokenizer实现(比minibpe繁琐一些)](https://www.bilibili.com/video/BV1SZ42177SH/?spm_id_from=333.337.search-card.all.click&vd_source=acbdb8fe72197a35f3556cf0747ef4e4)


### 大致思路

词表和语料是两个概念，首先将基本字符集作为token加入词表，保证最基本的所有字符都有出现可能性。
其次，对语料进行分词和迭代计算相邻字符出现频次，每次均频次最高的相邻字符，合并为一个token并加入词表。

#### 疑惑

加入词表的顺序相当于是

基本单字符token -> 2字符token -> 3字符token -> 4字符token ...

在完整的单词token加入词表之前，已经有大量零散的“半词”token加入词表了，词表就会很大。

例如，'hello'一词，
先加入基本字符'h''e''l''o'，再依次加入'he', 'hel', 'hell', 'hello'

也就是一个词'hello'会有许多冗余token加入词表。

这个问题现在是否已被解决了呢？

#### 疑惑
```python
A = 'aaa'.encode('utf-8')   # 按照utf-8标准转换为字节 bytes
print(A)                # 表面上是 b'aaa', 实际上是以int_list格式存储 [97, 97, 97]
for i in A: print(i)    # 输出 97 97 97, 是因为将bytes迭代展开了
for i in A: print(bytes([i]))   # 输出 b'a' b'a' b'a', 将展开的int转换成bytes
```

#### 疑惑
'''python
self.vocab = {i:bytes([i]) for i in range(256)}
'''
256个基本字符? 为什么跟前256个数字有对应?

答：bytes([i])表示的是，创建一个长度为1的bytes对象，它的ASCII码为i，所以是全部的基本字符。