## code from https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510

from torch import nn
import torch.nn.functional as F

class HLoss(nn.Module):
    """
        returning the negative entropy of an input tensor
    """
    def __init__(self, is_maximization=False):
        super(HLoss, self).__init__()
        self.is_neg = is_maximization

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        if self.is_neg:
            # b = 1.0 * b.sum()          # summation over batches         
            b = 1.0 * b.sum(dim=1).mean()     # summation over batches, mean over batches       
        else:
            # b = -1.0 * b.sum()
            b = -1.0 * b.sum(dim=1).mean()     # summation over batches, mean over batches
        return b