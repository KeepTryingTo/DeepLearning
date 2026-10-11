import time
import copy
from networks import *
from utils.meters import *
import random

class Trainer(object):
    def __init__(self, args, model, criterion):
        super(Trainer, self).__init__()
        self.model = model
        self.device = 'cpu' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.args = args

    def train(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_meta_train = AverageMeter()
        losses_meta_test = AverageMeter()
        metaLR = optimizer.param_groups[0]['lr']

        #TODO 得到划分的域数量，默认为4
        source_count = len(data_loaders)
        end = time.time()
        #TODO 每一个epoch，训练集都要迭代400次
        for i in range(train_iters):
            # with torch.autograd.set_detect_anomaly(True):
            # TODO divide source domains into meta_tr and meta_te  根据划分的域，得到划分域的索引
            data_loader_index = [i for i in range(source_count)]  ## 0 2
            random.shuffle(data_loader_index) #对其索引打乱操作
            #TODO 根据指定域的数量得到source_count个图像数据集作为batch
            batch_data = [data_loaders[i].next() for i in range(source_count)]
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            # self_param = list(self.model.parameters())
            for p in self.model.parameters():
                #TODO 判断模型中参数的梯度情况
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            # meta train(inner loop) begins
            loss_meta_train = 0.
            loss_meta_test = 0.
            # TODO meta-train on samples from multiple meta train domains 分别对num_domain个域的batch进行预测
            for t in range(source_count):  # 0 ... num_domain - 1
                inner_model = copy.deepcopy(self.model)
                #定义优化器
                inner_opt = torch.optim.Adam(inner_model.parameters(),
                                             lr=metaLR, weight_decay=self.args.weight_decay)

                data_time.update(time.time() - end)
                # TODO process inputs for meta train 根据当前遍历域的索引得到得到打乱之后集合域的一个batch
                traininputs = batch_data[data_loader_index[t]]
                #TODO 得到域对应的索引
                trainid = data_loader_index[t]
                #TODO 如果遍历到最后一个域，则
                if t == len(data_loader_index)-1:
                    testinputs = batch_data[data_loader_index[0]]
                    testid = data_loader_index[0]
                else:
                    testinputs = batch_data[data_loader_index[t+1]]
                    testid = data_loader_index[t+1]
                inputs, targets = self._parse_data(traininputs)[2:]

                pred_mtr, sim_loss, sim_loss2, orth_loss = inner_model.train_forward(inputs, trainid)
                loss_mtr = self.criterion(pred_mtr, targets) + torch.sum(sim_loss) + sim_loss2 + orth_loss
         
                loss_meta_train += loss_mtr

                inner_opt.zero_grad()
                loss_mtr.backward()
                inner_opt.step()

                #TODO 使用当前训练的inner_model梯度来更新原始model的梯度
                for p_tgt, p_src in zip(self.model.parameters(), inner_model.parameters()):
                    if p_src.grad is not None:
                        p_tgt.grad.data.add_(p_src.grad.data / source_count)
                #TODO testInputs数据集是源数据进行增强之后的结果
                testInputs, testMaps = self._parse_data(testinputs)[:2]
                # TODO meta test begins
                pred_mte, sim_loss, sim_loss2, orth_loss  = inner_model.train_forward(testInputs, testid)

                loss_mte = self.criterion(pred_mte, testMaps) + torch.sum(sim_loss) + sim_loss2 + orth_loss
                loss_meta_test += loss_mte
                #TODO 用于计算给定输出相对于给定输入的梯度
                grad_inner_j = torch.autograd.grad(loss_mte, inner_model.parameters(), allow_unused=True)

                #TODO 再一次进行梯度的更新
                for p, g_j in zip(self.model.parameters(), grad_inner_j):
                    if g_j is not None:
                        p.grad.data.add_(1.0 * g_j.data / source_count)


            loss_final = loss_meta_train + loss_meta_test
            losses_meta_train.update(loss_meta_train.item())
            losses_meta_test.update(loss_meta_test.item())

            optimizer.step()

            losses.update(loss_final.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Total loss {:.3f} ({:.3f})\t'
                      'Loss {:.3f}({:.3f})\t'
                      'LossMeta {:.3f}({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              losses_meta_train.val, losses_meta_train.avg,
                              losses_meta_test.val, losses_meta_test.avg))

    def _parse_data(self, inputs):
        imgs, dens, imgs2, dens2 = inputs
        return imgs.to(device), dens.to(device), imgs2.to(device), dens2.to(device)
