import torch
import torch.nn as nn

import pytorch_lightning as pl

from lib.utils import standard_normal_logprob

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, dataloader):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.iters = 0
        
    def forward(self, x):
        zero = torch.zeros(x.shape[0], 1).to(x)
        x = x[:7]
        c = x[7:]
        return self.model(x,c,zero)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1
        x= batch
        
        zero = torch.zeros(x.shape[0], 1).to(x)

        c = x[:, 7:]
        x = x[:, :7]

        z, delta_logp = self.model(x, c, zero)

        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)

        return {'loss': loss} 
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-5)

    def train_dataloader(self):
        return self.dataloader
