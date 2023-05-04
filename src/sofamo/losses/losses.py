# %%
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight
from .lovasz_losses import lovasz_softmax
from ..common.onehot import make_onehot


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction="mean"):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight,
                                      ignore_index=ignore_index,
                                      reduction=reduction)

    def forward(self, output, target):
        loss = self.ce(output, target)
        return loss


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma=2,
                 alpha=None,
                 ignore_index=255,
                 size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.ce = nn.CrossEntropyLoss(reduce=False,
                                      ignore_index=ignore_index,
                                      weight=alpha)

    def forward(self, output, target):
        target = target.squeeze(1)  # [bs, H, W]
        target = target.to(dtype=torch.long)
        logpt = self.ce(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self,
                 smooth=1,
                 reduction="mean",
                 ignore_index=255,
                 weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.fn_dice = DiceLoss(ignore_index=ignore_index)
        self.fn_ce = nn.CrossEntropyLoss(weight=weight,
                                         reduction=reduction,
                                         ignore_index=ignore_index)

    def forward(self, output, target):
        ce_loss = self.fn_ce(output, target)
        dice_loss = self.fn_dice(output, target)
        return ce_loss + dice_loss


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        n_classes = output.shape[1]
        target = make_onehot(target, n_classes=n_classes)
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        numerator = intersection + self.smooth
        denominator = output_flat.sum() + target_flat.sum() - \
            intersection + self.smooth
        loss = 1 - numerator / denominator
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        n_classes = output.shape[1]
        target = make_onehot(target, n_classes=n_classes)
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        numerator = 2.0 * intersection + self.smooth
        denominator = output_flat.sum() + target_flat.sum() + self.smooth
        loss = 1 - numerator / denominator
        return loss


# class WeightedDiceLoss(nn.Module):
# 	def __init__(self):
# 		super(MulticlassDiceLoss, self).__init__()

# 	def forward(self, input, target, weights=None):
# 		n_classes = input.shape[1]
# 		target = make_onehot(target, n_classes=n_classes)

# 		# if weights is None:
# 		# 	weights = torch.ones(n_classes)

# 		func_dice = DiceLoss()
# 		loss = 0
# 		for i in range(n_classes):
# 			dice_loss = func_dice(input[:,i,:,:], target[:,i,:,:])
# 			if weights is not None:
# 				dice_loss = weights[i] * dice_loss
# 			loss = loss + dice
# 		return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, output, target):
        n_classes = output.shape[1]
        target = make_onehot(target, n_classes=n_classes)
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (output_flat * target_flat).sum()
        numerator = intersection + self.smooth
        denominator = (
            intersection
            + self.alpha * (output_flat.sum() - intersection)
            + self.beta * (target_flat.sum() - intersection)
            + self.smooth
        )
        loss = 1 - numerator / denominator
        return loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes="present", per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.fn_lovasz_softmax = lovasz_softmax

    def forward(self, predict, target):
        logits = F.softmax(predict, dim=1)
        loss = self.fn_lovasz_softmax(
            logits,
            target,
            classes=self.smooth,
            per_image=self.per_image,
            ignore=self.ignore_index,
        )
        return loss
