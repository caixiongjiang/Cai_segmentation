import torch
import torch.nn as nn
import torch.nn.functional as F


from configs import default_config


class CrossEntropy(nn.Module):
    def __init__(self, weight=None, num_classes=19):
        # weight代表每个类别的权重
        super(CrossEntropy, self).__init__()
        self.num_classes = num_classes
        # ignore_index设置成num_classes是为了实现前景分割，如果不是前景分割可以设置为其他值
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=num_classes
        )

    def _forward(self, input, target):

        n, c, h, w = input.size()
        nt, ht, wt = target.size()
        if h != ht and w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
        
        temp_input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_target = target.view(-1)
        
        loss = self.criterion(temp_input, temp_target)

        return loss
    
    def forward(self, input, target):

        if default_config.MODEL.NUM_OUTPUTS == 1:
            input = [input]

        balance_weights = default_config.LOSS.BALANCE_WEIGHTS
        sb_weights = default_config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(input):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, input)])
        elif len(input) == 1:
            return sb_weights * self._forward(input[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


class Focal_loss(nn.Module):
    def __init__(self, weight=None, num_classes=19, alpha=0.5, gamma=2):
        # weight代表每个类别的权重
        super(Focal_loss, self).__init__()
    
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=num_classes, reduction='none')
    
    def _forward(self, input, target):
        
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
    
    def forward(self, input, target):

        if default_config.MODEL.NUM_OUTPUTS == 1:
            input = [input]

        balance_weights = default_config.LOSS.BALANCE_WEIGHTS
        sb_weights = default_config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(input):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, input)])
        elif len(input) == 1:
            return sb_weights * self._forward(input[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")

            

