import torch
import math
import warnings
warnings.filterwarnings("ignore")


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, consensus_type, dim):
        
        ctx.shape = input.size()
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        
        if consensus_type == 'avg':
            output = input.mean(dim, keepdim=True)
        elif consensus_type == 'max':
            output = input.max(dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input
        else:
            output = None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None, None


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus.apply(input, self.consensus_type, self.dim)