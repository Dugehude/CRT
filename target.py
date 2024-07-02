import argparse
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

import loss
import network
from CRLoss import CRconLoss, ClassRelationships
from autoaugment import CIFAR10Policy as ImageNetPolicy
from data_list import ImageList_train, ImageList_test
from VAT import VAT


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    transforms_train = {
        "train": [None]*2
    }
    transform_weak = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    # To Teacher

    transform_strong = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        normalize
    ])
    # To student

    transforms_train["train"][0] = transform_weak
    transforms_train["train"][1] = transform_strong

    return transforms_train["train"]


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def update_ema_variables(model, ema_model, alpha, global_step=float("inf")):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()
        for entry in ema_model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()
            new_param = (ema_param * alpha) + (param * (1. - alpha))
            ema_model_state_dict[entry] = new_param
        ema_model.load_state_dict(ema_model_state_dict)


def Cholesky(matrix):
    w = matrix.shape[0]
    G = torch.zeros((w, w))
    for i in range(w):
        G[i, i] = (matrix[i, i] - torch.dot(G[i, :i], G[i, :i].T)) ** 0.5
        for j in range(i + 1, w):
            G[j, i] = (matrix[j, i] - torch.dot(G[j, :i], G[i, :i].T)) / G[i, i]
    return G


def train_target(args):
    C_fea, C_label = None, None
    lamda = args.lamda
    dset_loaders, train_size = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        TnetF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
        TnetF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()

    TnetB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                    bottleneck_dim=args.bottleneck).cuda()
    TnetC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    modelpath = args.output_dir_src + '/source_F.pt'
    TnetF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    TnetB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    TnetC.load_state_dict(torch.load(modelpath))
    TnetF.eval()
    TnetB.eval()
    TnetC.eval()
    CR = ClassRelationships(args, netC)

    for net in [netC, TnetF, TnetB, TnetC]:
        for k, v in net.named_parameters():
            v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    tol_cls_loss, tol_im_loss, tol_cp_loss = 0, 0, 0

    while iter_num < max_iter:
        try:
            inputs_weak, inputs_strong, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_weak, inputs_strong, _, tar_idx = iter_test.next()

        if inputs_weak.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            TnetF.eval()
            TnetB.eval()
            TnetC.eval()

            mem_label, mem_conf, C_fea, count = obtain_label(dset_loaders['test'], TnetF, TnetB, TnetC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            mem_conf = mem_conf.cuda()
            CR.set_C_fea(C_fea)

        inputs_weak, inputs_strong = inputs_weak.cuda(), inputs_strong.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_stu = netF(inputs_weak)
        features_stu = netB(features_stu)
        outputs_stu = netC(features_stu)

        features_tea = TnetF(inputs_strong)
        features_tea = TnetB(features_tea)
        outputs_tea = TnetC(features_tea)

        if args.cls_par > 0:
            pred = mem_label[tar_idx].long()
            conf = mem_conf[tar_idx]
            max_p, _ = torch.max(F.softmax(outputs_stu, dim=1), dim=1)
            if args.cross:
                vat_cp_loss = lambda x: (CRconLoss(CR)(F.softmax(outputs_tea, dim=1).detach().clone(), x)) * args.cp_par
                vat_cp_f = VAT(x=features_stu, loss=vat_cp_loss, netC=netC, args=args, vat=args.vat_e)
                vat_cp_p = netC(vat_cp_f)
            else:
                vat_cp_p = outputs_stu
            classifier_loss = (
                    nn.CrossEntropyLoss(reduction="none", label_smoothing=args.ls)(vat_cp_p, pred) *
                    (conf >= args.max_dic) * (max_p <= args.cls_conf)
            ).mean()
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA":
                classifier_loss *= 0

        else:
            classifier_loss = torch.tensor(0.0).cuda()

        cls_loss = classifier_loss.detach().clone()

        if args.ent:
            im_loss = 0
            softmax_out = nn.Softmax(dim=1)(outputs_stu)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent_par > 0:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= args.gent_par * gentropy_loss  # * max(1 - iter_num/max_iter*2, 0)
            im_loss += entropy_loss * args.ent_par

        else:
            im_loss = 0.
        classifier_loss += im_loss

        pre = F.softmax(outputs_tea, dim=1).detach().clone()
        CR.set_C_label(pre)
        CR.uploadC()

        if args.cp_par > 0:
            if args.cross:
                vat_cls_loss = lambda x: (
                                                 nn.CrossEntropyLoss(reduction="none", label_smoothing=args.ls)(x, pred) *
                                                 (max_p <= args.cls_conf)  # * conf
                                         ).mean() * args.cls_par
                vat_cls_f = VAT(features_stu, vat_cls_loss, netC, args, vat=min(5, args.vat_e))
                vat_cls_p = F.softmax(netC(vat_cls_f), dim=1)
            else:
                vat_cls_p = F.softmax(outputs_stu, dim=1)

            l = CRconLoss(CR)
            cp_loss = l(F.softmax(outputs_tea, dim=1).detach().clone(), vat_cls_p,
                        label=None if iter_num < interval_iter and args.dset == "VISDA" else mem_label[
                            tar_idx].long()) * args.cp_par
        else:
            cp_loss = 0

        classifier_loss += cp_loss

        optimizer.zero_grad()
        classifier_loss.backward()

        optimizer.step()

        update_ema_variables(netF, TnetF, args.ema_alpha, iter_num)
        update_ema_variables(netB, TnetB, args.ema_alpha, iter_num)
        update_ema_variables(netC, TnetC, args.ema_alpha, iter_num)

        tol_cls_loss += cls_loss
        tol_im_loss += im_loss
        tol_cp_loss += cp_loss

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            if args.vat_e < args.max_e:
                args.vat_e += args.vat_e_add
            args.max_dic = min(args.max_dic - args.mid_dic, args.max_max_dic)
            if args.dset == 'VISDA':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], TnetF, TnetB, TnetC, True)
                tol_cls_loss, tol_im_loss, tol_cp_loss = \
                    tol_cls_loss/interval_iter, tol_im_loss/interval_iter, tol_cp_loss/interval_iter
                log_str = 'Tea Task: {}, Iter:{}/{}; Accuracy = {:.2f}% \n cls: {} im: {} cp: {} '.format(
                    args.name, iter_num, max_iter, acc_s_te, tol_cls_loss, tol_im_loss, tol_cp_loss) + \
                          '\n' + acc_list
                tol_cls_loss, tol_im_loss, tol_cp_loss = 0, 0, 0
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], TnetF, TnetB, TnetC, False)
                try:
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% \n cls: {} im: {} cp: {}'.format(
                        args.name, iter_num, max_iter, acc_s_te, tol_cls_loss, tol_im_loss, tol_cp_loss)
                except UnboundLocalError:
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% \n cls: {}'.format(
                        args.name, iter_num, max_iter, acc_s_te, tol_cls_loss)
                tol_cls_loss, tol_im_loss, tol_cp_loss = 0, 0, 0

            netF.eval()
            netB.eval()
            if args.dset == 'VISDA':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str += '\nStu Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter,
                                                                                   acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                try:
                    log_str += '\nStu Accuracy = {:.2f}%'.format(acc_s_te)
                except UnboundLocalError:
                    pass
            netF.train()
            netB.train()
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            # torch.save(C, f"{args.s}-{args.t}=={iter_num}.pth")

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        print(osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))

    return acc_s_te if args.dset!="VISDA" else acc_list, netF, netB, netC


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped.detach().clone(), dim=1, keepdim=True) + 1e-8
    return d


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC, args, topK=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    dd = torch.tensor(dd)

    if topK is None:
        topK = int(dd.shape[0] * args.max_dic)
    mdd = torch.min(dd, dim=1)[0]

    conf = torch.zeros(dd.shape[0])
    conf[torch.topk(mdd, topK, largest=False)[1]] = 1
    b_topK = int(dd.shape[0] / 65 / 4)
    conf[torch.topk(dd, b_topK, dim=0, largest=False)[1]] = 1
    conf += (1 - conf) * 0.1
    cd = torch.tensor(cdist(initc, initc, args.distance)) / 2
    C_fea = torch.exp(-cd * args.feature_delay).cuda()
    for i in range(args.class_num):
        C_fea[i, i] = 1.
    count = np.where(cls_count == args.threshold)[0]

    return predict.astype('int'), conf, C_fea, count

def data_load(args):
    # prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

            new_tar = []
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_train(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    if args.eval_aug == 'weak':
        dsets["eval"] = ImageList_test(txt_tar, transform=image_train()[0])
    else:
        dsets["eval"] = ImageList_test(txt_tar, transform=image_train()[1])
    dset_loaders["eval"] = DataLoader(dsets["eval"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    train_size = len(dsets["target"])
    print("train size:", train_size)

    dsets["test"] = ImageList_test(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders, train_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument("--cross", type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.5)
    parser.add_argument('--ent_par', type=float, default=1.5)
    parser.add_argument('--gent_par', type=float, default=1.5)
    parser.add_argument("--cp_par", type=float, default=1.0)
    parser.add_argument("--vat_move", type=float, default=0.1)
    parser.add_argument('--vat_e', type=float, default=20.)
    parser.add_argument("--vat_e_add", type=float, default=2.)
    parser.add_argument("--max_e", type=float, default=20.)
    parser.add_argument('--vat_ent_par', type=float, default=0.)
    parser.add_argument('--init_C', type=bool, default=True)
    parser.add_argument("--cls_conf", type=float, default=1.0)
    parser.add_argument('--ema_alpha', type=float, default=0.99)
    parser.add_argument('--alpha', default=0.1)
    parser.add_argument('--max_dic', type=float, default=0.9)
    parser.add_argument('--mid_dic', type=float, default=-0.04)
    parser.add_argument('--max_max_dic', type=float, default=0.9)
    parser.add_argument('--ls', type=float, default=0.)
    parser.add_argument('--feature_delay', type=float, default=2.)
    parser.add_argument('--lamda', type=float, default=1.0)

    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/target/')
    parser.add_argument('--output_src', type=str, default='san/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=False)

    parser.add_argument('--eval_aug', type=str, default='weak',
                        help="types of augmented features in memory bank (this is not important in this version)")

    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    elif args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    elif args.dset == 'VISDA':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset,
                                   names[args.s][0].upper() + names[args.t][0].upper() + str(time.time()))
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
