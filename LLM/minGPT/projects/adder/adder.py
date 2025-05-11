"""
Trains a GPT to add n-digit numbers.
"""

import os
import sys
import json

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
    C.system.work_dir = './outputs/adder'

    # data
    C.data = AdditionDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt2'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "85 50 531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.ndigit = 2
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # train/test

        # TODO 将所有加法问题拆分为训练数据或测试数据
        ndigit = self.config.ndigit
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        """
            假设每个加数是一个ndigit位十进制数（范围0到10^ndigit - 1）
            加法组合总数为 (10^ndigit) * (10^ndigit) = 10^(2*ndigit)
            （例如ndigit=2时，num=10000种组合）
        """
        num = (10**ndigit)**2 # TODO 整数可能的加法问题的总数
        rng = torch.Generator()
        rng.manual_seed(1337) # TODO 固定随机种子保证可复现性
        perm = torch.randperm(num, generator=rng) # TODO 生成0到num-1的随机排列
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        """
            由于潜在的进位溢出，a,b，a+b和+1，但还有-1，因为最后一个数字永远不会插入，
            因为没有显式的<EOS>令牌可以预测，这是隐含的
        """
        return 3*self.config.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.config.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # TODO calculate the "label" of the addition problem a + b
        c = a + b
        # TODO encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # TODO reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # TODO 转换字符到对应的token convert each character to its token index
        # TODO x作为编码器的输入，y作为期望的输出，如果了解基于transformer的机器翻译就知道，解码器部分和编码器之间的关系
        #  x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)#TODO 去掉最后一个元素
        y = torch.tensor(dix[1:], dtype=torch.long)  # TODO 去掉第一个元素 predict the next token in the sequence
        #TODO
        y[:ndigit*2-1] = -1 # TODO 我们将只在输出地点进行训练。-1将把损失掩盖为零
        return x, y

# -----------------------------------------------------------------------------


def main():
    #TODO  get default config and overrides from the command line, if any
    config = get_config()
    # config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # TODO 构建训练集和测试集
    train_dataset = AdditionDataset(config.data, split='train')
    test_dataset  = AdditionDataset(config.data, split='test')

    # TODO 构建模型和词汇表 construct the model
    config.model.vocab_size = train_dataset.get_vocab_size() #TODO 这里的词汇表就是0-9，因为数字都是0-9之间的数随机组成的
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # TODO construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        ndigit = config.data.ndigit
        results = []
        mistakes_printed_already = 0

        factors = torch.tensor([[10**i for i in range(ndigit+1)][::-1]]).to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # TODO 获得操作数1和操作数2 isolate the first two digits of the input sequence alone
            d1d2 = x[:, :ndigit*2]
            # TODO 输入操作数1和操作数2 ，让后进行生成结果，让模型对序列的其余部分进行采样
            d1d2d3 = model.generate(d1d2, ndigit+1, do_sample=False) # using greedy argmax, not sampling
            #TODO  分离采样序列的最后ndigit位数，也就是去最后生成的数字
            d3 = d1d2d3[:, -(ndigit+1):]
            d3 = d3.flip(1) # reverse the digits to their "normal" order
            # TODO 从字符串解码操作数1和操作数2  decode the integers from individual digits
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(dim = 1)
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(dim = 1)
            #TODO 对生成的结果进行解码
            d3i_pred = (d3 * factors).sum(1)
            d3i_gt = d1i + d2i # manually calculate the ground truth
            # TODO 验证生成的结果和真实值是否相同 evaluate the correctness of the results in this batch
            correct = (d3i_pred == d3i_gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # TODO 只打印最多5个错误
                    mistakes_printed_already += 1
                    print("GPT claims that %d + %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        nonlocal top_score
        #TODO
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            train_max_batches = {1: None, 2: None, 3: 5}[config.data.ndigit] # if ndigit=2 we can afford the whole train set, ow no
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            score = train_score + test_score
            # TODO save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()


def demo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config()

    train_dataset = AdditionDataset(config.data, split='train')
    config.model.vocab_size = train_dataset.get_vocab_size()  # TODO 这里的词汇表就是0-9，因为数字都是0-9之间的数随机组成的
    config.model.block_size = train_dataset.get_block_size()
    ndigit = int(config.data.ndigit)
    model = GPT(config.model).to(device)

    weight_path = r'outputs/adder/model.pt'
    checkpoint = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    def add(a, b, ndigit):
        c = a + b
        # TODO encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit + 1}d' % c)[::-1]  # TODO reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render]  # TODO 转换字符到对应的token convert each character to its token index
        # TODO x作为编码器的输入，y作为期望的输出，如果了解基于transformer的机器翻译就知道，解码器部分和编码器之间的关系
        #  x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)  # TODO 去掉最后一个元素
        y = torch.tensor(dix[1:], dtype=torch.long)  # TODO 去掉第一个元素 predict the next token in the sequence
        # TODO
        y[:ndigit * 2 - 1] = -1  # TODO 我们将只在输出地点进行训练。-1将把损失掩盖为零

        return x,y
    #TODO [100,10,1]
    factors = torch.tensor([[10 ** i for i in range(ndigit + 1)][::-1]]).to(device)
    while True:
        a = int(input("请输入a的值: "))
        b = int(input("请输入b的值: "))
        x, y = add(a,b,ndigit)
        x = x.unsqueeze(dim = 0).to(device)

        # TODO 获得操作数1和操作数2 isolate the first two digits of the input sequence alone
        d1d2 = x[:, :ndigit * 2]
        # TODO 输入操作数1和操作数2 ，让后进行生成结果，让模型对序列的其余部分进行采样
        d1d2d3 = model.generate(d1d2, ndigit + 1, do_sample=False)  # using greedy argmax, not sampling
        # TODO  分离采样序列的最后ndigit位数，也就是去最后生成的数字
        d3 = d1d2d3[:, -(ndigit + 1):]
        d3 = d3.flip(1)  # reverse the digits to their "normal" order
        # TODO 从字符串解码操作数1和操作数2  decode the integers from individual digits
        d1i = (d1d2[:, :ndigit] * factors[:, 1:]).sum(dim=1)
        d2i = (d1d2[:, ndigit:ndigit * 2] * factors[:, 1:]).sum(dim=1)
        # TODO 对生成的结果进行解码
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i  # manually calculate the ground truth

        print(f'{a} + {b} prediciton = {d3i_pred.item()} and gt is {d3i_gt.item()}')

if __name__ == '__main__':
    num = (10 ** 2) ** 2  # TODO 整数可能的加法问题的总数
    rng = torch.Generator()
    rng.manual_seed(1337)  # TODO 固定随机种子保证可复现性
    perm = torch.randperm(num, generator=rng)  # TODO 生成0到num-1的随机排列
    num_test = min(int(num * 0.2), 500)  # 20% of the whole dataset, or only up to 500

    # main()
    demo()