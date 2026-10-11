import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

# from models.net import vgg16, vgg16_bn
from models.update_net import vgg16_bn
from models.resnet_yolo import resnet50, resnet18
from models.darknet import Yolov1
from yoloLoss import yoloLoss
from dataset.dataset import yoloDataset

# from visualize import Visualizer
import numpy as np

use_gpu = torch.cuda.is_available()

file_root = '/data1/KTG/myDataset/VOC/Image07_12/'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.0003
num_epochs = 500
batch_size = 8
use_resnet = False
use_vgg = True
if use_resnet:
    net = resnet50()
    resnet = models.resnet50(pretrained=True)
    new_state_dict = resnet.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and not k.startswith('fc'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    print('backbone used is resne50...')
elif use_vgg:
    net = vgg16_bn()
    vgg = models.vgg16_bn(pretrained=True)
    new_state_dict = vgg.state_dict()
    dd = net.state_dict()
    for k in new_state_dict.keys():
        if k in dd.keys() and k.startswith('features'):
            dd[k] = new_state_dict[k]
    net.load_state_dict(dd)
    print('backbone used is vgg16...')
else:
    net = Yolov1(in_channels=3,split_size = 14,
                num_boxes = 2,num_classes = 20)
    print('backbone used is darknet...')

print('load pre-trined model')


if False:
    net.load_state_dict(torch.load('best.pth'))
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

criterion = yoloLoss(S=7,B = 2, l_coord=5,l_noobj=0.5, device = device)
net = net.to(device)

net.train()
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

# train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
train_dataset = yoloDataset(root=file_root,
                            list_file=['save/voc2012.txt','save/voc2007.txt'],
                            train=True,
                            transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,num_workers=8)
# test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
test_dataset = yoloDataset(root=file_root, list_file='save/voc2007test.txt',
                           train=False, transform = [transforms.ToTensor()])
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
# vis = Visualizer(env='xiong')
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    # if epoch == 1:
    #     learning_rate = 0.0005
    # if epoch == 2:
    #     learning_rate = 0.00075
    # if epoch == 3:
    #     learning_rate = 0.001
    if epoch == 30:
        learning_rate=0.0001
    if epoch == 40:
        learning_rate=0.00001
    # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.to(device),target.to(device)
        
        pred = net(images)
        loss = criterion(pred,target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1
            # vis.plot_train_val(loss_train=total_loss/(i+1))

    #validation
    validation_loss = 0.0
    net.eval()
    for i,(images,target) in enumerate(test_loader):
        images = Variable(images,volatile=True)
        target = Variable(target,volatile=True)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        validation_loss += loss.item()
    validation_loss /= len(test_loader)
    # vis.plot_train_val(loss_val=validation_loss)
    
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(net.state_dict(),'./weights/vgg_epoch_{}_best.pth'.format(epoch))
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
    logfile.flush()      
    torch.save(net.state_dict(),'./weights/yolo.pth')
    

