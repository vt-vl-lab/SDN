## code from https://github.com/jindongwang/transferlearning/tree/master/code/deep/DANN(RevGrad)
## original paper: Ganin Y, Lempitsky V. Unsupervised domain adaptation by backpropagation. ICML 2015.

from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
