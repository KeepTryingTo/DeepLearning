"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './output/chargpt'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt2'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    TODO text8: text8：也来源于Wikipedia文本，但是所有的XML都被删除了，并且被小写到只有26个字符
    TODO enwik8 benchmark ("Hutter Prize"),enwik8基准（“Hutter Prize”），维基百科XML转储的前100M字节，包含205个唯一的令牌（英语加上空格）
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        #TODO 字符：下标
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        #TODO 下标：字符
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        #TODO  从数据中获取一个（block_size + 1）字符块
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # TODO 编码每一个字符到整数（对应字符下标）
        dix = [self.stoi[s] for s in chunk]
        # TODO 返回tensor类型的token
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

def main():
    # get default config and overrides from the command line, if any
    config = get_config()
    # config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # TODO TODO 构建数据集
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # TODO 构建模型
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # TODO 训练器
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                # sample from the model...
                context = "O God, O God!"
                x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])
                print(completion)
            # save the latest model
            print("saving model")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
# -----------------------------------------------------------------------------


def demo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config()

    # TODO TODO 构建数据集
    text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)
    config.model.vocab_size = train_dataset.get_vocab_size()  # TODO 这里的词汇表就是0-9，因为数字都是0-9之间的数随机组成的
    config.model.block_size = train_dataset.get_block_size()

    model = GPT(config.model).to(device)

    weight_path = r'output/chargpt/model.pt'
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    def chat(model, context):
        model.eval()
        with torch.no_grad():
            x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(device)
            y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
            completion = ''.join([train_dataset.itos[int(i)] for i in y])
            print('gpt: {}'.format(completion))

    while True:
        context = input("user: ")
        chat(model, context)

if __name__ == '__main__':
    # main()
    demo()
    pass