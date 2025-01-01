"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-20:01
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import json

def build_vocab(file_path):
    # 读取所有文本
    texts = []
    with open(file_path, 'r', encoding='utf-8') as r:
        for line in r:
            if not line:
                continue
            line = json.loads(line)
            question = line["question"]
            answer = line["answer"]
            texts.append(question)
            texts.append(answer)
    # 拆分 Token
    words = set()
    for t in texts:
        if not t:
            continue
        for word in t.strip():
            words.add(word)
    words = list(words)
    words.sort()
    # 特殊Token
    # pad 占位、unk 未知、sep 结束
    word2id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    # 构建词表
    word2id.update({word: i + len(word2id) for i, word in enumerate(words)})
    id2word = list(word2id.keys())
    vocab = {"word2id": word2id, "id2word": id2word}
    vocab = json.dumps(vocab, ensure_ascii=False)
    with open('data/vocab.json', 'w', encoding='utf-8') as w:
        w.write(vocab)
    print(f"finish. words: {len(id2word)}")

if __name__ == '__main__':
    # build_vocab("./data/train.json")
    pass