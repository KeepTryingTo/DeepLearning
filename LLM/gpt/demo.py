"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-20:12
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import torch
from models.gpt import GPTModel
from dataset.tokenizer import Tokenizer


def generate(model, tokenizer, text, max_length, device):
    input, att_mask = tokenizer.encode(text)
    input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)
    stop = False
    input_len = len(input[0])
    while True:
        if len(input[0]) - input_len > max_length:
            next_symbol = tokenizer.sep_token
            input = torch.cat([
                    input.detach(),
                    torch.tensor([[next_symbol]], dtype=input.dtype, device=device)],
                    dim = -1
            )
            break
        #TODO 模型输出预测结果
        projected, self_attns = model(input)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        #TODO 解码获得概率词对应索引
        next_word = prob.data[-1]
        #TODO 作为下一个词的预测输入
        next_symbol = next_word
        #TODO 是否预测结束
        if next_symbol == tokenizer.sep_token:
            stop = True
        #TODO 当前词的预测和输入拼接预测下一个词
        input = torch.cat([
            input.detach(),
            torch.tensor([[next_symbol]], dtype=input.dtype, device=device)],
            dim = -1
        )
    #TODO 根据预测的结果索引解码得到对应的词
    decode = tokenizer.decode(input[0].tolist())
    decode = decode[len(text):]
    return "".join(decode)


def main():
    model_path = "outputs/best.pt"
    vocab_path = "./dataset/data/vocab.json"  # 词表位置
    max_length = 128  # 最大长度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载分词器
    tokenizer = Tokenizer(vocab_path)
    # 模型参数
    model_param = {
        "d_model": 768,  # 嵌入层大小
        "d_ff": 2048,  # 前馈神经网络大小
        "d_k": 64,  # K 的大小
        "d_v": 64,  # V 的大小
        "n_layers": 6,  # 解码层的数量
        "n_heads": 8,  # 多头注意力的头数
        "max_pos": 1800,  # 位置编码的长度
        "device": device,  # 设备
        "vocab_size": tokenizer.get_vocab_size(),  # 词表大小
    }
    model = GPTModel(**model_param)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    while True:
        text = input("请输入：")
        if not text:
            continue
        if text == "q":
            break
        res = generate(model, tokenizer, text, max_length, device)
        print("AI: ", res)


if __name__ == '__main__':
    main()