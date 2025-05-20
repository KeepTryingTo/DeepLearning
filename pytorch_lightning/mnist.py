"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/5/18-18:42
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
from typing import Any

import torch
from torch import nn
from torch import optim, nn, utils, Tensor
from torch.optim import Optimizer
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as L
from pytorch_lightning import loggers
from torch.utils.data import DataLoader

# define any number of nn.Modules (or use your current ones)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self,x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28)
        )
    def forward(self,x):
        return self.decoder(x)


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self,
                 encoder,
                 decoder,
                 save_dir : str = None
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.save_dir = save_dir

    def on_train_start(self) -> None:
        if os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def on_train_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pass

    def on_train_end(self) -> None:
        pass

    def on_train_epoch_end(self, outputs) -> None:
        # TODO 在训练周期结束时调用，并输出所有训练步骤的输出。
        #  如果您需要对每个training_step的所有输出执行某些操作，请使用此方法。
        self.scheduler.step()

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.encoder.parameters() + self.decoder.parameters(),
                                         lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",  # 或 "step"
                "frequency": 1,  # 每1个interval调用一次
            },
            "monitor": "loss"
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x,_ = val_batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def backward(self, loss: Tensor, optimizer: Optimizer, optimizer_idx: int, *args, **kwargs) -> None:
        loss.backward()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


# TODO init the autoencoder
encoder = Encoder()
decoder = Decoder()
autoencoder = LitAutoEncoder(encoder, decoder)


# setup data
train_dataset = MNIST(root='./data', download=True, transform=ToTensor())
val_dataset = MNIST(root='./data', download=True, transform=ToTensor())
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 4
)
val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = 4
)

#TODO 第一点 保存模型
save_callback = L.callbacks.ModelCheckpoint(
    monitor='val_loss',#TODO 监控的指标为平均绝对误差最小的,这一点和on_validation_epoch_end日志记录的指标是呼应的
    save_top_k=1, #TODO 这里的1，表示保存的模型中，只保存前4个最好结果模型权重文件
    mode='min',#TODO 表示保存当前误差最小的模型
    filename='{epoch}-{val_mae:.2f}'#TODO 保存模型格式
)
#TODO 早停机制
early_stopping = L.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.1,
    patience=5,
    verbose=True,#TODO 冗长模式
    mode='min'
)

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
logger_ten = L.loggers.TensorBoardLogger("lightning_logs", name="mnist")
logger_mlf = L.loggers.MLFlowLogger(experiment_name='mnist')
#TODO 以yaml和CSV格式记录到本地文件系统。
logger_csv = L.loggers.CSVLogger(save_dir="save path")
logger_comet  = L.loggers.CometLogger(api_key="")
logger_wand = L.loggers.WandbLogger(name="mnist")
logger_ = L.loggers.NeptuneLogger()

trainer = L.Trainer(
    limit_train_batches=100,
    max_epochs=10,
    callbacks=[save_callback],
    val_check_interval=5
)
trainer.fit(
    model=autoencoder,
    train_dataloader=train_loader,
    val_dataloaders=val_loader
)

# TODO 加载模型 load checkpoint
checkpoint = r"D:\conda3\Transfer_Learning\PyTorchLighting\lightning_logs\version_0\checkpoints\epoch=9-step=999.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(
    checkpoint,
    encoder=encoder,
    decoder=decoder
)

# TODO 选择编码器 choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# TODO 初始化一张图像 embed 4 fake images!
fake_image_batch = torch.rand(4, 28 * 28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)

# train on 4 GPUs
# trainer = L.Trainer(
#     devices=1,
#     accelerator="gpu",
#  )

# # train 1TB+ parameter models with Deepspeed/fsdp
# trainer = L.Trainer(
#     devices=4,
#     accelerator="gpu",
#     strategy="deepspeed_stage_2",
#     precision=16
#  )
#
# # 20+ helpful flags for rapid idea iteration
# trainer = L.Trainer(
#     max_epochs=10,
#     min_epochs=5,
#     overfit_batches=1
#  )

class myDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.split = split
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class customModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        pass

    def training_step(self, batch, batch_dix):
        pass


train_dataset = myDataset(split='train')
test_dataset = myDataset(split='test')

train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

model = customModel()

train_ = L.Trainer(
    max_epochs=10,
    devices=[0],
    check_val_every_n_epoch=1
)

train_.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)

