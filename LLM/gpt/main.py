"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/12/31-19:41
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

"""
reference to: https://mp.weixin.qq.com/s/hha33dv5yISvlV_cFd5baA
"""

import torch
from models.gpt import GPTModel

# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.tokenizer import Tokenizer
from models.gpt import GPTModel
from dataset.dataset import QADataset
from tqdm import tqdm
import time, sys, os


def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs,
        model_output_dir,
        writer
):
    batch_step = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs, dec_self_attns = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            writer.add_scalar('Loss/train', loss, batch_step)
            batch_step += 1
            # 100轮打印一次 loss
            if index % 100 == 0 or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss)} ; lr: {optimizer.param_groups[0]['lr']} ;each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, criterion, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"val loss: {val_loss} , epoch: {epoch}")
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_output_dir, "best.pt")
            print("Save Best Model To ", best_model_path, ", epoch: ", epoch)
            torch.save(model.state_dict(), best_model_path)
        # 保存当前模型
        last_model_path = os.path.join(model_output_dir, "last.pt")
        print("Save Last Model To ", last_model_path, ", epoch: ", epoch)
        torch.save(model.state_dict(), last_model_path)

def validate_model(model, criterion, device, val_loader):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            outputs, dec_self_attns = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.view(-1))
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    train_json_path = "./dataset/data/train.json"  # 训练集
    val_json_path = "./dataset/data/val.json"  # 验证集
    vocab_path = "./dataset/data/vocab.json"  # 词表位置
    max_length = 120  # 最大长度
    epochs = 150 # 迭代周期
    train_batch_size = 32  # 训练一个批次的大小
    test_batch_size = 2
    lr = 1e-4  # 学习率
    model_output_dir = "outputs"  # 模型保存目录
    logs_dir = "logs"  # 日志记录目标
    # 设备
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
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
    print("Start Load Train Data...")
    train_params = {
        "batch_size": train_batch_size,
        "shuffle": True,
        "num_workers": 8,
    }
    training_set = QADataset(train_json_path, tokenizer, max_length)
    training_loader = DataLoader(training_set, **train_params)
    print("Start Load Validation Data...")
    val_params = {
        "batch_size": test_batch_size,
        "shuffle": False,
        "num_workers": 8,
    }
    val_set = QADataset(val_json_path, tokenizer, max_length)
    val_loader = DataLoader(val_set, **val_params)
    # 日志记录
    writer = SummaryWriter(logs_dir)
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    model = model.to(device)
    # 开始训练
    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer
    )
    writer.close()


if __name__ == '__main__':
    main()