import torch
import torch.nn as nn
class WeakCompareLoss(nn.Module):
    def __init__(self, C=None, class_num=65, conf=0.95):
        super().__init__()
        self.class_num = class_num
        self.C = C.cuda()
        self.nor = lambda x: x / torch.sqrt(torch.sum(x ** 2, dim=0))
        self.conf = conf

    def forward(self, log, pre, labels=None):
        """

        Args:
            labels: None
            log: detach B*Class
            pre:  grad B*Class

        Returns:

        """
        return 0
        if labels is not None:
            pass
        t = torch.tensor(0.07)
        self.class_num = log.shape[1]
        C = self.C.clone()  # / torch.sum(self.C, dim=0)
        f = lambda x, l: x[0, l] / torch.sum(x)
        loss = torch.mean(-torch.log(torch.tensor([
            (f(torch.exp(torch.mm(pre[i:i+1], C) / t), label) + 1e-5)
            if conf > self.conf else 1
            for i, (conf, label) in enumerate(zip(torch.max(log, dim=1)[0], labels))
        ]).cuda()))
        return torch.tensor(0) if torch.isnan(loss) else loss
