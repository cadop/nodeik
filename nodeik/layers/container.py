import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, std, logpx=None, rev=False, inds=None):
        if inds is None:
            if rev:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, std, rev=rev)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, std, logpx, rev=rev)
            return x, logpx
