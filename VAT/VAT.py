import torch
import torch.nn.functional as F

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped.detach().clone(), dim=1, keepdim=True) + 1e-8
    return d

def VAT(x, loss, netC, args, move=False, vat=None):
    """
    return Vat samples
    Args:
        x: feature
        loss: loss, lambda x: loss_(x, y)
        netC:  classifier
        args:  args
        move: whether moving
        vat:  max_d

    Returns: Vat Sample Features

    """
    if move:
        d = torch.rand(x.shape).to(x.device).sub(0.5)
    else:
        d = torch.zeros(x.shape).to(x.device)

    d.requires_grad_()
    l = loss(F.softmax(netC(x.detach().clone() + d*args.vat_move), dim=1))
    l.backward()

    d = d.grad
    r_dav = _l2_normalize(d) * (args.vat_e if vat is None else vat)

    act = x + r_dav.detach().clone()

    return act

