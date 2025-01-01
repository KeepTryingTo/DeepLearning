"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-20:10
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

from torch.utils.data import Dataset
import torch
import json
import numpy as np


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    question = json_line["question"]
                    answer = json_line["answer"]
                    self.data.append({
                        "question": question,
                        "answer": answer
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, question, answer):
        encode, att_mask = self.tokenizer.encode(question,
                                                 answer,
                                                 max_length=self.max_length,
                                                 pad_to_max_length=True)
        input_ids = encode[:-1]
        att_mask = att_mask[:-1]
        labels = encode[1:]
        return input_ids, att_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, att_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(att_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)