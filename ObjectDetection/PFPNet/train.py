import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from data import *
from utils.trainutils import LossCal, Timer, CaffeSGD, CaffeScheduler
from layers.modules import RefineDetMultiBoxLoss
from valid import _valid


def _train(model, args, cfg, dataset,device):

    lr, lr2 = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias"):
            lr2.append(param)
        else:
            lr.append(param)

    param_groups = [{'params': lr, 'weight_decay': args.weight_decay, 'lr': args.lr, 'lr_mult': 1.},
                    {'params': lr2, 'weight_decay': 0., 'lr': args.lr, 'lr_mult': 2.}]

    optimizer = CaffeSGD(param_groups, lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(num_classes=2, overlap_thresh=0.5,
                                          prior_for_matching=True, bkg_label=0,
                                          neg_mining=True, neg_pos=3,
                                          neg_overlap=0.5, encode_target=False, device = device)
    odm_criterion = RefineDetMultiBoxLoss(cfg['num_classes'],
                                          overlap_thresh=0.5, prior_for_matching=True,
                                          bkg_label=0, neg_mining=True,
                                          neg_pos=3, neg_overlap=0.5,
                                          encode_target=False, device = device, use_ARM=True)

    scheduler = CaffeScheduler(optimizer=optimizer, milestones=cfg['lr_steps'],
                               gamma=args.gamma, base_lr=args.lr)

    writer = SummaryWriter()
    iters = 0

    if args.resume:
        print('Resume trainig, loading {}'.format(args.resume))
        state = torch.load(args.resume)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        iters = state['iter']

    else:
        def weights_init(m):
            def xavier(param):
                init.xavier_uniform_(param)

            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                try:
                    m.bias.data.zero_()
                except AttributeError:
                    pass

        vgg_weights = torch.load(args.basenet)
        model.vgg.load_state_dict(vgg_weights)

        print('Initialize weights')
        model.arm_loc.apply(weights_init)
        model.arm_conf.apply(weights_init)
        model.odm_loc.apply(weights_init)
        model.odm_conf.apply(weights_init)
        model.msca.apply(weights_init)
        model.fppool.apply(weights_init)
        model.pyramid_head.apply(weights_init)

    print('Load the dataset')
    train_loader = data.DataLoader(dataset[0],
                                  args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    print('Training the model')
    num_epoch = int(cfg['max_iter'] * args.batch_size / len(train_loader)) + 1
    print('Max %d epochs, Max iter %d' % (num_epoch, cfg['max_iter']))

    arm_iter = LossCal()
    odm_iter = LossCal()
    timer = Timer('m')
    model.train()

    timer.tic()
    mini_iter = 0
    # model.phase = 'test'
    # map = _valid(model, args, iters, dataset[1],device)


    for epoch in range(num_epoch):
        for batch_data in train_loader:
            mini_iter += 1
            images, targets = batch_data
            images = images.to(device)
            targets = [ann.to(device) for ann in targets]

            if mini_iter % args.iter_size == 1:
                optimizer.zero_grad()

            out = model(images)

            arm_l, arm_c = arm_criterion(out, targets)
            odm_l, odm_c = odm_criterion(out, targets)

            arm_loss = arm_c + arm_l
            odm_loss = odm_c + odm_l

            loss = arm_loss + odm_loss

            loss.backward()

            arm_iter.stack(arm_l.item(), arm_c.item())
            odm_iter.stack(odm_l.item(), odm_c.item())

            if mini_iter % args.iter_size == 0:
                scheduler.step()
                optimizer.step()
                mini_iter = 0
                iters += 1
                if iters % args.print_freq == 0:
                    lrs = []
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] not in lrs:
                            lrs.append(param_group['lr'])
                    al, ac = arm_iter.pop()
                    ol, oc = odm_iter.pop()
                    print("%d/%d Time: %.2f" %(iters, cfg['max_iter'], timer.toc()), "lr: ", lrs,
                          "ARM_L: %.3f ARM_C: %.3f ODM_L: %.3f ODM_C: %.3f" %(al,ac,ol,oc))
                    timer.tic()
                    writer.add_scalar('ARM_loss/loc', al, iters)
                    writer.add_scalar('ARM_loss/conf', ac, iters)
                    writer.add_scalar('ODM_loss/loc', ol, iters)
                    writer.add_scalar('ODM_loss/conf', oc, iters)
                    arm_iter.reset()
                    odm_iter.reset()

                if iters % args.valid_freq == 0:
                    model.phase = 'test'
                    map = _valid(model, args, iters, dataset[1],device)
                    writer.add_scalar('mAP', map, iters)
                    model.phase = 'train'

                if iters % args.save_freq == 0:
                    print('Saving state, iter:', iters)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter': iters
                    }, args.save_folder + '/Net_%s_%d.pth'%(args.input_size, iters))

                if iters == cfg['max_iter']:
                    break
        if iters == cfg['max_iter']:
            print('Saving state:', iters)
            torch.save({
                'model': model.state_dict()
            }, args.save_folder + '/Net_%s.pth' % (args.input_size))
            break
