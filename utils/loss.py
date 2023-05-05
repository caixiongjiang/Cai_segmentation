import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, weight=None, num_classes=19):
        # weight代表每个类别的权重
        super(CrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=num_classes
        )

    def forward(self, input, target):

        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        if h != ht and w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        
        temp_input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_target = target.view(-1)
        
        loss = self.criterion(temp_input, temp_target)

        return loss


class Focal_loss(nn.Module):
    def __init__(self, weight=None, num_classes=19, alpha=0.5, gamma=2):
        # weight代表每个类别的权重
        super(Focal_loss, self).__init__()
    
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=num_classes, reduction='none')
    
    def forward(self, input, target):
        
        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        if h != ht and w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        temp_input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_target = target.view(-1)
        
        logpt = -self.criterion(temp_input, temp_target)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        loss = loss.mean()
        return loss
            

