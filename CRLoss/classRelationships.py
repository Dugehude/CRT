import torch

def getC(args, netC):
    weight = netC.fc.weight_g * netC.fc.weight_v / torch.norm(netC.fc.weight_v) \
        if args.layer == "wn" else netC.fc.weight
    c_par = torch.concat((weight.cpu().detach().clone(),
                          torch.unsqueeze(netC.fc.bias.cpu().detach().clone(), dim=-1)), dim=1)
    c_par /= torch.sqrt(torch.sum(c_par ** 2, dim=1)).reshape(args.class_num, 1)
    return torch.mm(c_par, c_par.T)

class ClassRelationships:
    def __init__(self, args, netC=None):
        if args.init_C and netC:
            self.C = getC(args, netC)
        else:
            self.C = torch.eye(args.class_num).cuda()
        self.C_fea, self.C_label = None, None
        self.lamda = args.lamda
        self.e = args.e

    def uploadC(self):
        Ct = self.e * self.C_label + (1 - self.e) * self.C_fea
        self.C = self.lamda * self.C + (1 - self.lamda) * Ct

    def set_C_label(self, pre):
        nor = lambda x: x / torch.sqrt(torch.sum(x ** 2, axis=0))
        nor_pre = nor(pre)
        self.C_label = torch.mm(torch.transpose(nor_pre, 1, 0), nor_pre)

    def set_C_fea(self, C_fea):
        self.C_fea = C_fea
