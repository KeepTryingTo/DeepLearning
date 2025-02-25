
import os.path as osp
from scipy import linalg
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist

from codes.lib.utils import transf_to_CLIP_input, dummy_context_mgr
from codes.lib.utils import mkdir_p, get_rank
from codes.lib.datasets import prepare_data
from codes.models.inception import InceptionV3



############   GAN   ############
def train(dataloader, netG, netD, netC, text_encoder, image_encoder,
          optimizerG, optimizerD, scaler_G, scaler_D, args):
    batch_size = args.batch_size
    device = args.device
    epoch = args.current_epoch
    max_epoch = args.max_epoch
    z_dim = args.z_dim
    netG, netD, netC, image_encoder = netG.train(), netD.train(), netC.train(), image_encoder.train()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=len(dataloader))
    for step, data in enumerate(dataloader, 0):
        ##############
        # TODO Train D   首先更新判别器
        ##############
        optimizerD.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            # prepare_data TODO 对图像，文本描述，文本token，描述句子嵌入向量，词嵌入向量以及文件名
            real, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            real = real.requires_grad_()
            sent_emb = sent_emb.requires_grad_()
            # words_embs = words_embs.requires_grad_()
            # predict real TODO CLIP对图像进行编码；返回局部特征向量和图像编码结果
            CLIP_real,real_emb = image_encoder(real)
            #TODO 判断
            real_feats = netD(CLIP_real)
            #TODO 返回预测的真实性和误差；对于图像和文本匹配的给予更小的误差
            pred_real, errD_real = predict_loss(netC, real_feats, sent_emb, negtive=False)
            # predict mismatch TODO 对于不匹配的句子嵌入向量给予更大的误差
            mis_sent_emb = torch.cat((sent_emb[1:], sent_emb[0:1]), dim=0).detach()
            _, errD_mis = predict_loss(netC, real_feats, mis_sent_emb, negtive=True)
            # synthesize fake images TODO 得到随机的高斯噪声
            noise = torch.randn(batch_size, z_dim).to(device)
            #TODO 根据噪声和文本描述，生成假的图像
            fake = netG(noise, sent_emb)
            #TODO 对图像CLIP编码
            CLIP_fake, fake_emb = image_encoder(fake)
            fake_feats = netD(CLIP_fake.detach())
            #TODO 对于生成的假图像给予更大的误差
            _, errD_fake = predict_loss(netC, fake_feats, sent_emb, negtive=True)
        # MA-GP TODO 混合精度
        if args.mixed_precision:
            errD_MAGP = MA_GP_MP(CLIP_real, sent_emb, pred_real, scaler_D)
        else:
            errD_MAGP = MA_GP_FP32(CLIP_real, sent_emb, pred_real)
        # whole D loss
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            errD = errD_real + (errD_fake + errD_mis)/2.0 + errD_MAGP
        # update D
        if args.mixed_precision:
            scaler_D.scale(errD).backward()
            scaler_D.step(optimizerD)
            scaler_D.update()
            if scaler_D.get_scale()<args.scaler_min:
                scaler_D.update(16384.0)
        else:
            errD.backward()
            optimizerD.step()
        ##############
        # Train G  TODO 更新生成器
        ##############
        optimizerG.zero_grad()
        with torch.cuda.amp.autocast() if args.mixed_precision else dummy_context_mgr() as mpc:
            fake_feats = netD(CLIP_fake)
            #TODO 对于生成图像，在句子向量的条件下判断真假性
            output = netC(fake_feats, sent_emb)
            #TODO 计算生成图像编码向量和句子向量之间的相似度
            text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
            errG = -output.mean() - args.sim_w*text_img_sim
        if args.mixed_precision:
            scaler_G.scale(errG).backward()
            scaler_G.step(optimizerG)
            scaler_G.update()
            if scaler_G.get_scale()<args.scaler_min:
                scaler_G.update(16384.0)
        else:
            errG.backward()
            optimizerG.step()
        # update loop information
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            loop.update(1)
            loop.set_description(f'Train Epoch [{epoch}/{max_epoch}]')
            loop.set_postfix()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        loop.close()


def test(dataloader, text_encoder, netG, PTM, device,
         m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    FID, TI_sim = calculate_FID_CLIP_sim(
        dataloader, text_encoder, netG, PTM,
        device, m1, s1, epoch, max_epoch,
        times, z_dim, batch_size
    )
    return FID, TI_sim


def save_model(netG, netD, netC, optG, optD, epoch, multi_gpus, step, save_path):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        state = {'model': {'netG': netG.state_dict(), 'netD': netD.state_dict(), 'netC': netC.state_dict()}, \
                'optimizers': {'optimizer_G': optG.state_dict(), 'optimizer_D': optD.state_dict()},\
                'epoch': epoch}
        torch.save(state, '%s/state_epoch_%03d_%03d.pth' % (save_path, epoch, step))


#########   MAGP   ########
def MA_GP_MP(img, sent, out, scaler):
    grads = torch.autograd.grad(outputs=scaler.scale(out),
                            inputs=(img, sent),
                            grad_outputs=torch.ones_like(out),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    inv_scale = 1./(scaler.get_scale()+float("1e-8"))
    #inv_scale = 1./scaler.get_scale()
    grads = [grad * inv_scale for grad in grads]
    with torch.cuda.amp.autocast():
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0,grad1),dim=1)                        
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def MA_GP_FP32(img, sent, out):
    grads = torch.autograd.grad(outputs=out,
                            inputs=(img, sent),
                            grad_outputs=torch.ones(out.size()).cuda(),
                            retain_graph=True,
                            create_graph=True,
                            only_inputs=True)
    grad0 = grads[0].view(grads[0].size(0), -1)
    grad1 = grads[1].view(grads[1].size(0), -1)
    grad = torch.cat((grad0,grad1),dim=1)                        
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp =  2.0 * torch.mean((grad_l2norm) ** 6)
    return d_loss_gp


def sample(dataloader, netG, text_encoder,
           save_dir, device, multi_gpus,
           z_dim, stamp):
    netG.eval()
    for step, data in enumerate(dataloader, 0):
        ######################################################
        # (1) Prepare_data
        ######################################################
        real, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
        ######################################################
        # (2) Generate fake images
        ######################################################
        batch_size = sent_emb.size(0)
        with torch.no_grad():
            #TODO 生成高斯噪声
            noise = torch.randn(batch_size, z_dim).to(device)
            #TODO 生成假的图像
            fake_imgs = netG(noise, sent_emb, eval=True).float()
            fake_imgs = torch.clamp(fake_imgs, -1., max=1.)
            #TODO 是否使用多GPU
            if multi_gpus==True:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', str('gpu%d'%(get_rank())), 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            else:
                batch_img_name = 'step_%04d.png'%(step)
                batch_img_save_dir  = osp.join(save_dir, 'batch', 'imgs')
                batch_img_save_name = osp.join(batch_img_save_dir, batch_img_name)
                batch_txt_name = 'step_%04d.txt'%(step)
                batch_txt_save_dir  = osp.join(save_dir, 'batch', 'txts')
                batch_txt_save_name = osp.join(batch_txt_save_dir, batch_txt_name)
            mkdir_p(batch_img_save_dir)
            vutils.save_image(fake_imgs.data, batch_img_save_name, nrow=8, value_range=(-1, 1), normalize=True)
            mkdir_p(batch_txt_save_dir)
            txt = open(batch_txt_save_name,'w')
            for cap in captions:
                txt.write(cap+'\n')
            txt.close()
            for j in range(batch_size):
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                ######################################################
                # (3) Save fake images
                ######################################################      
                if multi_gpus==True:
                    single_img_name = 'batch_%04d.png'%(j)
                    single_img_save_dir  = osp.join(save_dir, 'single', str('gpu%d'%(get_rank())), 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)
                else:
                    single_img_name = 'step_%04d.png'%(step)
                    single_img_save_dir  = osp.join(save_dir, 'single', 'step%04d'%(step))
                    single_img_save_name = osp.join(single_img_save_dir, single_img_name)   
                mkdir_p(single_img_save_dir)   
                im.save(single_img_save_name)
        if (multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('Step: %d' % (step))


def calculate_FID_CLIP_sim(dataloader, text_encoder, netG, CLIP, device,
                           m1, s1, epoch, max_epoch, times, z_dim, batch_size):
    """ Calculates the FID """
    clip_cos = torch.FloatTensor([0.0]).to(device)
    # prepare Inception V3
    dims = 2048
    """
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = dist.get_world_size()
    dl_length = dataloader.__len__()
    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            ######################################################
            # (1) Prepare_data TODO 记载数据
            ######################################################
            imgs, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, text_encoder, device)
            ######################################################
            # (2) Generate fake images
            ######################################################
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                #TODO 根据噪声和句子向量条件，生成假的图像
                noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb,eval=True).float()
                # norm_ip(fake_imgs, -1, 1) TODO 缩放到指定范围
                fake_imgs = torch.clamp(fake_imgs, -1., 1.)
                #TODO 如果图像中存在nan,使用-1.0代替
                """
                nan (float, optional): 用于替换 NaN 的值。默认为 0.0。
                posinf (float, optional): 用于替换正无穷大的值。默认为 float('inf')。
                neginf (float, optional): 用于替换负无穷大的值。默认为 float('-inf')。
                """
                fake_imgs = torch.nan_to_num(fake_imgs, nan=-1.0, posinf=1.0, neginf=-1.0)
                #TODO 对文本和向量token使用CLIP编码之后计算相似性
                clip_sim = calc_clip_sim(CLIP, fake_imgs, CLIP_tokens, device)
                clip_cos = clip_cos + clip_sim
                #TODO 归一化操作
                fake = norm(fake_imgs)
                #TODO 计算inception输出结果
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                # concat pred from multi GPUs TODO 如果存在多个GPU，将多个GPU上的数据收集
                output = list(torch.empty_like(pred) for _ in range(n_gpu))
                dist.all_gather(output, pred)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            # update loop information
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                if epoch==-1:
                    loop.set_description('Evaluating]')
                else:
                    loop.set_description(f'Eval Epoch [{epoch}/{max_epoch}]')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    # CLIP-score TODO 如果存在多个GPU，将多个GPU上的计算余弦相似度数据收集
    CLIP_score_gather = list(torch.empty_like(clip_cos) for _ in range(n_gpu))
    dist.all_gather(CLIP_score_gather, clip_cos)
    clip_score = torch.cat(CLIP_score_gather, dim=0).mean().item()/(dl_length*times)
    # FID
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value,clip_score


def calc_clip_sim(clip, fake, caps_clip, device):
    ''' calculate cosine similarity between fake and text features,
    '''
    # Calculate features
    fake = transf_to_CLIP_input(fake)
    fake_features = clip.encode_image(fake)
    text_features = clip.encode_text(caps_clip)
    text_img_sim = torch.cosine_similarity(fake_features, text_features).mean()
    return text_img_sim


def sample_one_batch(noise, sent, netG, multi_gpus,
                     epoch, img_save_dir, writer):
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        netG.eval()
        with torch.no_grad():
            B = noise.size(0)
            #TODO 根据噪声和句子向量生成训练图像
            fixed_results_train = generate_samples(noise[:B//2], sent[:B//2], netG).cpu()
            torch.cuda.empty_cache()
            #TODO 生成测试图像
            fixed_results_test = generate_samples(noise[B//2:], sent[B//2:], netG).cpu()
            torch.cuda.empty_cache()
            fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, value_range=(-1, 1), normalize=True)


def generate_samples(noise, caption, model):
    with torch.no_grad():
        fake = model(noise, caption, eval=True)
    return fake


def predict_loss(predictor, img_feature, text_feature, negtive):
    #TODO 判断图像的真假是在文本描述的条件下
    output = predictor(img_feature, text_feature)
    err = hinge_loss(output, negtive)
    return output,err


def hinge_loss(output, negtive):
    if negtive==False:
        err = torch.mean(F.relu(1. - output))
    else:
        err = torch.mean(F.relu(1. + output))
    return err


def logit_loss(output, negtive):
    batch_size = output.size(0)
    real_labels = torch.FloatTensor(batch_size,1).fill_(1).to(output.device)
    fake_labels = torch.FloatTensor(batch_size,1).fill_(0).to(output.device)
    output = nn.Sigmoid()(output)
    if negtive==False:
        err = nn.BCELoss()(output, real_labels)
    else:
        err = nn.BCELoss()(output, fake_labels)
    return err

# TODO FID 的计算过程
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    #TODO atleast_1d方法是NumPy库中一个非常实用且易于理解的函数，用于确保输入至少是一维数组
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    '''
    print('&'*20)
    print(sigma1)#, sigma1.type())
    print('&'*20)
    print(sigma2)#, sigma2.type())
    '''
    # Product might be almost singular
    #TODO linalg.sqrtm 通常指的是计算一个矩阵的平方根的一种函数，特别地是当这个矩阵可能是非对称或者没有简单的实数特征值分解时。这里的“m”可能指的是“matrix”（矩阵）的缩写。
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component TODO 如果是复数情况，则取其实部即可
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
