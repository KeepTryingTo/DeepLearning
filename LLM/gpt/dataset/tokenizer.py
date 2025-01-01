"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-20:02
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import json

class Tokenizer():

    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as r:
            vocab = r.read()
            if not vocab:
                raise Exception("词表读取为空！")
        vocab = json.loads(vocab)
        self.word2id = vocab["word2id"]
        self.id2word = vocab["id2word"]
        self.pad_token = self.word2id["<pad>"]
        self.unk_token = self.word2id["<unk>"]
        self.sep_token = self.word2id["<sep>"]

    def encode(self, text, text1=None, max_length=128, pad_to_max_length=False):
        #TODO 获得输入文本的token
        tokens = [self.word2id[word] if word in self.word2id else self.unk_token for word in text]
        #TODO 词与词之间的额间隔标志
        tokens.append(self.sep_token)
        if text1:
            tokens.extend([self.word2id[word] if word in self.word2id else self.unk_token for word in text1])
            tokens.append(self.sep_token)
        #TODO 根据文本token生成对应的掩码
        att_mask = [1] * len(tokens)
        #TODO 是否需要将文本填充至指定大小
        if pad_to_max_length:
            #TODO 如果大于指定长度就进行丢弃一部分
            if len(tokens) > max_length:
                tokens = tokens[0:max_length]
                att_mask = att_mask[0:max_length]
            #TODO 否则根据pad_token对文本token进行填充
            elif len(tokens) < max_length:
                tokens.extend([self.pad_token] * (max_length - len(tokens)))
                att_mask.extend([0] * (max_length - len(att_mask)))
        return tokens, att_mask

    def decode(self, token):
        #TODO 对token进行解码操作
        if type(token) is tuple or type(token) is list:
            return [self.id2word[n] for n in token]
        else:
            return self.id2word[token]

    def get_vocab_size(self):
        return len(self.id2word)

if __name__ == '__main__':
    tokenizer = Tokenizer(vocab_path="data/vocab.json")
    encode, att_mask = tokenizer.encode("你好,小毕超", "你好,小毕超", pad_to_max_length=True)
    decode = tokenizer.decode(encode)
    print("token lens: ", len(encode))
    print("encode: ", encode)
    print("att_mask: ", att_mask)
    print("decode: ", decode)
    print("vocab_size", tokenizer.get_vocab_size())