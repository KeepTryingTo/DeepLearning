import torch
from torch.nn import Module
from .bregman_pytorch import sinkhorn

class OT_Loss(Module):
    def __init__(self, c_size, stride,
                 norm_cood, device,
                 num_of_iter_in_ot=100,
                 reg=10.0):
        """
        裁剪图像大小，
        下采样比率，
        计算距离是否归一化坐标，
        加载设备，
        sinkhorn算法迭代次数，
        在sinkhorn算法中使用熵正则化
        """
        super(OT_Loss, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.norm_cood = norm_cood #TODO 计算距离是否归一化坐标
        self.num_of_iter_in_ot = num_of_iter_in_ot #TODO sinkhorn算法迭代次数
        self.reg = reg #TODO 在sinkhorn算法中使用熵正则化

        # TODO coordinate is same to image space, set to constant since crop size is same
        #TODO 根据裁剪大小和下采样步长得到一个[0,stide,2stride,3stride,...,crop_size] + stride / 2
        # =>[stride / 2, 1.5stride, 2.5stride,...]（就认为是将原图按照网格进行了划分，同时中心点在网格中心）
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.density_size = self.cood.size(0)
        self.cood.unsqueeze_(0) # [1, #cood]
        #TODO 对坐标进行归一化[-1,1]之间
        if self.norm_cood:
            self.cood = self.cood / c_size * 2 - 1 # map to [-1, 1]
        #TODO 网格的大小
        self.output_size = self.cood.size(1)


    def forward(self, normed_density, unnormed_density, points):
        """
        normed_density：网络输出经过归一化的密度图
        unnormed_density：未经过归一化的密度图
        points： 真实点标注（二维列表）
        """
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.output_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0 # TODO wasserstain distance
        #TODO 遍历一个batch的标注点列表
        for idx, im_points in enumerate(points):
            #TODO 首先判断当前图像对应的点标注是否大于0（该图像是否包含人群，其实这个条件基本是满足的）
            if len(im_points) > 0:
                # TODO compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                if self.norm_cood: #TODO 是否需要进行归一化操作
                    im_points = im_points / self.c_size * 2 - 1 # map to [-1, 1]
                #TODO 获得实际标注的X和Y坐标列表
                x = im_points[:, 0].unsqueeze_(1)  # [#gt, 1]
                y = im_points[:, 1].unsqueeze_(1)
                #TODO (x - cood)^2和(y - cood)^2 => [num_gt,num_cood]（cood: [1, #cood]）
                x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood # [#gt, #cood]
                y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
                y_dis.unsqueeze_(2) #TODO [num_gt, num_cood, 1]
                x_dis.unsqueeze_(1) #TODO [num_gt, 1, num_cood]
                dis = y_dis + x_dis #TODO [num_gt, num_cood, num_cood]
                dis = dis.view((dis.size(0), -1)) # TODO [num_gt, num_cood, num_cood] => [num_gt，num_cood * num_cood]

                #TODO 将当前的密度图并归一化转换为一维向量，作为源分布
                #TODO normed_density是经过归一化之后，也就是求和为1
                source_prob = normed_density[idx][0].view([-1]).detach()
                #TODO 作为目标分布，一维向量，并且是均匀分布的，求和为1
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
                # TODO use sinkhorn to solve OT, compute optimal beta.
                #TODO 这里的C=dis表示目标分布中心和源分布中心的距离，也就是“代价矩阵”
                #TODO https://zhuanlan.zhihu.com/p/675645310   https://blog.51cto.com/u_15876949/6406917
                P, log = sinkhorn(a=target_prob, b=source_prob,
                                  C=dis, reg=self.reg,
                                  maxIter=self.num_of_iter_in_ot,
                                  log=True)
                #TODO 进过sinkhorn算法迭代之后得到β
                beta = log['beta'] # size is the same as source_prob: [#cood * #cood]
                #TODO
                ot_obj_values += torch.sum(
                    normed_density[idx] * \
                    beta.view([1, self.output_size, self.output_size])
                )
                # TODO compute the gradient of OT loss to predicted density (unnormed_density).
                # TODO im_grad = beta / source_count - < beta, source_density> / (source_count)^2
                #TODO 首先计算当前图像预测的密度图人群数，并转换为一维向量（主要为了和后面的β维度一样）
                source_density = unnormed_density[idx][0].view([-1]).detach()
                source_count = source_density.sum()
                #TODO 最优化传输求解梯度的公式
                im_grad_1 = (source_count) / (source_count * source_count+1e-8) * beta # size of [#cood * #cood]
                im_grad_2 = (source_density * beta).sum() / (source_count * source_count + 1e-8) # size of 1

                im_grad = im_grad_1 - im_grad_2
                im_grad = im_grad.detach().view([1, self.output_size, self.output_size])
                # TODO Define loss = <im_grad, predicted density>. The gradient of loss w.r.t prediced density is im_grad.
                #TODO 上面的求导是对预测的密度图求导的，将求导计算的结果和预测的密度图相乘，然后计算求和作为损失值
                loss += torch.sum(unnormed_density[idx] * im_grad) #TODO 反映了调整预测密度图（即网络输出）以优化目标的方向和幅度
                #TODO wasserstain distance = 代价矩阵K * 分配策略P = 最优化传输的代价
                wd += torch.sum(dis * P).item()

        return loss, wd, ot_obj_values


