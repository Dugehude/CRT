import torch
import torch.nn as nn
class DotLoss(nn.Module):
    def __init__(self, C=None, class_num=65, lamda=0.999):
        super().__init__()
        self.class_num = class_num
        self.C = C.cuda()
        self.nor = lambda x: x / torch.sqrt(torch.sum(x ** 2, axis=0))
        self.lamda = lamda

    def forward(self, log, pre, C2=None, count=None):
        C = self.C.cuda()
        B = pre.shape[0]
        nor_pre = self.nor(pre)
        log_nor = torch.sqrt(torch.mm(torch.mm(log, C), log.T).detach())
        pre_nor = torch.sqrt(torch.mm(torch.mm(pre, C), pre.T).detach())
        new_C = torch.mm(torch.transpose(nor_pre, 1, 0), nor_pre)
        return -torch.trace(torch.mm(torch.mm(log, C), pre.T) / log_nor / pre_nor) / B, \
            self.upload(new_C, C2, count).cuda()

    def upload(self, C, C2=None, count=None):
        if True:
            if C2 is not None:
                e = torch.ones(C.shape) * 0.5
                if count is not None:
                    for ie in count:
                        C2[ie, :] = C[ie, :]
                        C2[:, ie] = C[:, ie]
                e = e.cuda()
                C = C.cuda()
                C = C*e + (1-e)*C2.cuda()
                # C += C - torch.eye(C.shape[0]).cuda()
                # VisDa need
                # C = torch.tensor(C, dtype=torch.float)
                C = C.to(torch.float)
                C = C.detach().clone().cuda()
        lamda = self.lamda
        self.C = lamda * self.C + (1 - lamda) * C
        return self.C
