import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from datasets.my_dataset import MyDataSet
from models.TBConv import resnet50_tbc
from utils import read_split_data, create_lr_scheduler, get_params_groups, train_one_epoch, evaluate


def main():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    print(f"using {device} device.")

    # train_path = r'E:\conda_3\PyCharm\Transer_Learning\classificationDataset\ImageNet1k-code\ImageNet_1k_0_999_test'
    # val_path = r'E:\conda_3\PyCharm\Transer_Learning\classificationDataset\ImageNet1k-code\ImageNet_1k_0_999_test'
    train_path = r'E:\conda_3\PyCharm\Transer_Learning\classificationDataset\flower_photos'
    val_path = r'E:\conda_3\PyCharm\Transer_Learning\classificationDataset\flower_photos'
    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

    img_size = 224
    init_lr = 5e-4
    wd = 5e-2
    epochs = 1000
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(root_dir=train_path,transform=data_transform['train']).ImageFold()

    print(train_dataset.classes)
    print(train_dataset.class_to_idx)
    # 实例化验证数据集
    val_dataset = MyDataSet(root_dir=val_path,transform=data_transform['val']).ImageFold()


    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0)

    model = resnet50_tbc().to(device)


    # pg = [p for p in model.parameters() if p.requires_grad]
    pg = get_params_groups(model, weight_decay=wd)
    optimizer = optim.AdamW(pg, lr=init_lr, weight_decay=wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs,
                                       warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                epochs = epochs,
                                                lr_scheduler=lr_scheduler)
        print('[{}/{}] --------- train_loss: {} -------- train_acc: {}'.format(epoch,epochs,train_loss,train_acc))
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        print('[{}/{}] --------- train_loss: {} -------- train_acc: {}'.format(epoch, epochs, val_loss, val_acc))
        if best_acc < val_acc:
            best_acc = val_acc
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc':best_acc
            }
            torch.save(state, f"./weights/{round(best_acc,2)}_best_model.pth")



if __name__ == '__main__':
    main()
