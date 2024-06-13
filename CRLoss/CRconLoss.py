import torch
import torch.nn as nn


class CRconLoss(nn.Module):
    def __init__(self, C=None, class_num=65, d="cuda"):
        super().__init__()
        self.class_num = class_num
        self.C = C
        self.nor = lambda x: x / torch.sqrt(torch.sum(x ** 2, dim=0))
        self.d = d

    def forward(self, log, pre, vat=None):
        t = torch.tensor(0.07)
        C = self.C.clone()
        C = C  # / torch.sum(C, dim=0)
        temp = torch.mm(torch.mm(log, C), pre.T)  # / log_nor / pre_nor
        temp = torch.exp(temp / t)
        return torch.mean(
            -torch.log(
                torch.diag(temp) / (torch.sum(temp, dim=0))
            )
        )