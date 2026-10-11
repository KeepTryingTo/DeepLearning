import torch
import torch.nn as nn

import numpy as np
from codes.miscc.config import cfg

from codes.models.GlobalAttention import func_attention


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code,
              rnn_code,
              labels,
              class_ids,
              batch_size,
              eps=1e-8):
    # ### Mask mis-match samples  ###
    # TODO that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    #TODO 对CNN提取的特征向量和文本编码向量L2归一化
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # TODO scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # TODO --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features,
               words_emb,
               labels,
               cap_lens,
               class_ids,
               batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    #TODO 针对batch 中每一张图像和句子进行遍历
    for i in range(batch_size):
        #TODO 判断当前的图像的真实id和
        if class_ids is not None:
            #TODO 根据当前遍历的batch 索引得到对应的mask（和one-hot差不多一样形式）
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description TODO 得到当前图像对应描述句子长度
        words_num = cap_lens[i]
        # -> 1 x nef x words_num TODO 得到当前句子的文本编码向量
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num TODO 在batch维度上复制
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        #TODO 获得编码的上下文和注意力图
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num TODO 对应论文中的公式R(Ci,Ei)，
        # TODO 计算词向量和编码图像之后的向量之间计算余弦相似性
        row_sim = cosine_similarity(word, weiContext)
        # TODO --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # TODO Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # TODO batch_size x batch_size
    similarities = torch.cat(similarities, dim=1)
    if class_ids is not None:
        masks = np.concatenate(masks, axis=0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()
    #todo 对应公式11
    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# ##################Loss for G and Ds##############################
def discriminator_loss(netD,
                       real_imgs,
                       fake_imgs,
                       conditions,
                       real_labels,
                       fake_labels):
    """
    conditions: 表示经过LSTM之后输出的句子编码向量
    """
    # Forward TODO 判断真实的图像和假的图像特征向量
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #TODO 在句子编码向量的调价下的判断结果
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)

    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #TODO 对于mismatch的样本计算损失
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)],
                                       conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits,
                                   fake_labels[1:batch_size])

    #TODO 默认采用了非条件判断
    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    return errD


def generator_loss(netsD,
                   image_encoder,
                   fake_imgs,
                   real_labels,
                   words_embs,
                   sent_emb,
                   match_labels,
                   cap_lens,
                   class_ids):
    #TODO 判别器的数量
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    #TODO 针对生成的多尺度图像进行判别
    for i in range(numDs):
        #TODO 判别器提取假图像特征
        features = netsD[i](fake_imgs[i])
        #TODO 在句子编码向量的条件下判别结果
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        #TODO 默认采用了非条件判断
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.data[0]
        logs += 'g_loss%d: %.2f ' % (i, g_loss.data.item())

        # Ranking loss TODO 最后一个判别器
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef TODO 对于最后一个判别器，也就是对最后生成的图像编码
            #TODO 得到每一个区域的特征向量和最后经过卷积的编码向量
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features,
                                             words_embs,
                                             match_labels,
                                             cap_lens,
                                             class_ids,
                                             batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.data[0]
            #TODO 句子方面级的损失
            s_loss0, s_loss1 = sent_loss(cnn_code,
                                         sent_emb,
                                         match_labels,
                                         class_ids,
                                         batch_size)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.data[0]

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.data.item(), s_loss.data.item())
    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
