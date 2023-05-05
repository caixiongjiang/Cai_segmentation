import torch
import torch.nn as nn
from torch.nn import functional as F


class NFD_loss(nn.Module):
    def __init__(self, t_channel, s_channel):
        super(NFD_loss, self).__init__()
        
        self.t_channel = t_channel
        self.s_channel = s_channel
        self.conv1x1 = nn.Conv2d(s_channel, t_channel, 1)
        
    def Normal(self, f):
        # 归一化：在高和宽两个维度上进行归一化
        mean = torch.mean(f, dim=(2, 3), keepdim=True)
        std = torch.std(f, dim=(2, 3), keepdim=True)
        
        return (f-mean)/std
    
    
    def D_L2(self, f1, f2):
        
        return torch.norm(f1-f2)


    def forward(self, f_t, f_s):
        _, _, t_W, t_H = f_t.shape
        _, _, s_W, s_H = f_s.shape
        
        if t_W != s_W or t_H != s_H:
            f_t = F.interpolate(f_t, size=(s_W, s_H), mode='bilinear', align_corners=True)
        
        if self.t_channel != self.s_channel:
            f_s = self.conv1x1(f_s)
            
        f_t = self.Normal(f_t)
        f_s = self.Normal(f_s)
            
        f_t.detach()  # 锁住教师网络的特征防止反向传播
        
        return self.D_L2(f_t, f_s) / (s_W * s_H)


class Logits_loss(nn.Module):
    def __init__(self, temperature=2, alpha=0.5):
        super(Logits_loss, self).__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        
    def pixel_loss(self, output_t, output_s):
        
        return F.mse_loss(output_t, output_s)
            
    
    def KL_loss(self, output_t, output_s):

        # 计算softmax输出
        soft_output_s = F.softmax(output_s / self.temperature, dim=1)
        soft_output_t = F.softmax(output_t / self.temperature, dim=1)
        # 计算KL散度损失
        KL_loss = F.kl_div(soft_output_s.log(), soft_output_t, reduction='batchmean') * (self.temperature ** 2)
        
        return KL_loss
        
    def forward(self, output_t, output_s):
        
        _, _, s_W, s_H = output_s.shape
        _, _, t_W, t_H = output_t.shape
        if s_W != t_W or s_H != t_H:
            output_t = F.interpolate(output_t, (s_W, s_H), mode='bilinear', align_corners=True)
        output_t.detach()
        
        kl_loss = self.KL_loss(output_t, output_s)
        pixel_loss_value = self.pixel_loss(output_t, output_s)
        logits_loss = self.alpha * kl_loss + (1 - self.alpha) * pixel_loss_value
        
        return logits_loss / (s_W * s_H)
    



if __name__ == '__main__':
    f_t = torch.randn(4, 4, 512, 512)
    f_s = torch.randn(4, 4, 224, 224)
    f_t_1 = torch.randn(4, 512, 512, 512)
    f_s_1 = torch.randn(4, 224, 224, 224)
    x = Logits_loss()
    y = NFD_loss(512, 224)
    loss1 = x(f_t, f_s)
    loss2 = y(f_t_1, f_s_1)
    print(loss1)
    print(loss2)
