## code from https://discuss.pytorch.org/t/cross-entropy-for-soft-label/16093 and https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/21

import torch
from torch import nn

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        logsoftmax = nn.LogSoftmax(dim=1)
        
        return torch.mean(torch.sum(- target * logsoftmax(inputs), 1))