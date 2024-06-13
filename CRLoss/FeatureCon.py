import torch
import torch.nn as nn

class FeatureCompare(nn.Module):
    def __init__(self, t=0.07):
        super().__init__()
        self.t = torch.tensor(t)

    def forward(self, x, y):
        x = (x.T / torch.norm(x, dim=1)).T
        y = (y.T / torch.norm(y, dim=1)).T

        temp = torch.mm(x, y.T)
        temp /= self.t
        temp = torch.exp(temp)
        los = torch.mean(-torch.log(torch.diag(temp) / (torch.sum(temp, dim=0))))
        if torch.isnan(los):
            print("NAN")
            print(torch.diag(temp), torch.sum(temp, dim=0))
            return torch.tensor(0.)
        return los
