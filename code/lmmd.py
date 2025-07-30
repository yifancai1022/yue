import torch
import torch.nn as nn

class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """计算高斯核矩阵"""
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        
        return sum(kernel_val)

    def get_loss(self, source, target):
        """计算MMD损失"""
        batch_size = source.size()[0]
        
        # 计算高斯核矩阵
        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, 
                                kernel_num=self.kernel_num, 
                                fix_sigma=self.fix_sigma)
        
        # 分割核矩阵为源域间(SS)、目标域间(TT)和源-目标域间(ST)
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]
        
        # 计算MMD损失
        # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
        loss = torch.mean(SS) + torch.mean(TT) - 2 * torch.mean(ST)
        
        return loss