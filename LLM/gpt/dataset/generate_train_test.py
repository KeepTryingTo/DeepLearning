"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-20:07
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import json
from tokenizer import Tokenizer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

def get_num_tokens(file_path, tokenizer):
    input_num_tokens = []
    with open(file_path, "r", encoding="utf-8") as r:
        for line in r:
            line = json.loads(line)
            question = line["question"]
            answer = line["answer"]
            tokens, att_mask = tokenizer.encode(question, answer)
            input_num_tokens.append(len(tokens))
    return input_num_tokens

def count_intervals(num_tokens, interval):
    max_value = max(num_tokens)
    intervals_count = {}
    for lower_bound in range(0, max_value + 1, interval):
        upper_bound = lower_bound + interval
        count = len([num for num in num_tokens if lower_bound <= num < upper_bound])
        intervals_count[f"{lower_bound}-{upper_bound}"] = count
    return intervals_count

def main():
    train_data_path = "data/train.json"
    tokenizer = Tokenizer("data/vocab.json")
    input_num_tokens = get_num_tokens(train_data_path, tokenizer)
    intervals_count = count_intervals(input_num_tokens, 20)
    print(intervals_count)
    x = [k for k, v in intervals_count.items()]
    y = [v for k, v in intervals_count.items()]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, y)
    plt.title('训练集Token分布情况')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')
    plt.show()

if __name__ == '__main__':
    main()