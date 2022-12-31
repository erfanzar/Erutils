import math
import os
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.cluster import KMeans

from .command_line_interface import fprint, Cp, print_model, attar_print

Any = Union[list, dict, int, float, str]


def arg_creator(arg: list = None, prefix=None):
    '''
    :param arg: args from list to pythonic like *args
    :param prefix: to use for first arg
    :return: args
    '''
    created_args = f''.join(
        (((f'{prefix if prefix is not None else ""},{v},' if i == 0 else f'{v},') if i != len(arg) - 1 else f'{v}') for
         i, v in enumerate(arg)))
    return created_args


def pars_model_v2(cfg, c_req: Union[list[str], str], detail: str = None, print_status: bool = False, sc: int = 3):
    """

    :param cfg: a list of lists contain 4 parameters like [index:[int,list[int ]],number:int,name:str,args:[Any]];
    :param c_req: channel request Models
    :param detail: Full detail log
    :param print_status: Status
    :param sc: Start Channel
    :return:
    """
    saves = []
    model = nn.ModuleList()

    if detail is not None:
        print(detail, end='')
    for i, c in enumerate(cfg):
        f, t, m, arg = c
        if print_status: print_model(m, arg, f, t, i)

        prefix = sc if m in c_req else ''
        arg_ = arg_creator(arg, prefix=prefix)
        model_name = f'{m}({arg_})'
        if not print_status: print(f"Adding : {model_name}")

        sc = arg[0] if m in c_req else sc

        m = eval(model_name)
        model.append(m)
    return model


def pars_model(cfg: list, device='cpu'):
    """
    :param cfg: a list of lists contain 4 parameters like [index:[int,list[int ]],number:int,name:str,args:[Any]];
    :param device: device that module going to build in default set to *'cpu'*;
    :return: Module
    """
    model = nn.ModuleList()
    index, save = 0, []
    fprint(f'{f"[ - {device} - ]":>46}', color=Cp.RED)
    fprint(f'{"From":>10}{"Numbers":>25}{"Model Name":>25}{"Args":>25}\n')
    for c in cfg:
        f, n, t, a = c
        args = arg_creator(a)
        fprint(f'{str(f):>10}{str(n):>25}{str(t):>25}{str(a):>25}')
        for i in range(n):
            string: str = f'{t}{args}'

            m = eval(string)
            m = m.to(device)
            setattr(m, 'f', f)
            model.append(m)
            if f != -1:
                save.append(index % f)

            index += 1

    return model, save


def max_args_to_max_non_tom(tensor: torch.Tensor, device: str):
    return_t = torch.zeros(tensor.shape, dtype=torch.long).to(device)
    for i, b in enumerate(tensor):
        index = max(b)
        return_t[i, int(index)] = 1
    return return_t


def max_args_to_one_arg(tensor: torch.Tensor, device: str):
    shape = list(tensor.shape)
    shape[-1] = 1
    return_t = torch.zeros(shape, dtype=torch.long).to(device)
    for i, b in enumerate(tensor):
        index = max(b)
        return_t[i, 0] = index
    return return_t


def accuracy(pred, target, total, true):
    assert (pred.shape == target.shape), 'Predictions And Targets should have same size'
    for p, t in zip(pred, target):
        if torch.argmax(p.cpu().detach() - 1).numpy().tolist() == torch.argmax(t.cpu().detach(), -1).numpy().tolist():
            true += 1
        total += 1
    acc = (true / total) * 100
    return acc, total, true


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def name_to_layer(name: str, attr: Any = None, in_case_prefix_use=None, prefix: Any = None,
                  form: [list, int] = -1,
                  print_debug: bool = True, nc: int = 4, anchors: list = None):
    if in_case_prefix_use is None:
        in_case_prefix_use = []
    attr = [attr] if isinstance(attr, int) else attr
    t = len(attr) if isinstance(attr, list) else (attr if attr is not None else 0)
    tu = ''.join((f'{v},' if i != t else f'{v}') for i, v in enumerate(attr)) if t != 0 else ''
    pr = '' if prefix is None else (f'{prefix},' if name in in_case_prefix_use else '') if t != 0 else ''
    fr = form if t != 0 else ''
    model_str = f'{name}({pr}{tu}{fr})'
    if print_debug:
        print(f"ADDING : {model_str} ")
    model = eval(model_str)
    return model


def module_creator(backbone, head, print_status, ic_backbone, nc, anchors, in_case_prefix_use):
    model = nn.ModuleList()
    save, sv_bb = [], []
    model_list = backbone + head
    sva, idx = 0, 0
    for i, at in enumerate(model_list):
        form, times, name = at[0], at[1], at[2]
        attr = attr_exist_check_(at, 3)
        ic_backbone = ic_backbone * len(form) if name == 'Concat' else ic_backbone
        for _ in range(times):
            model.append(
                name_to_layer(name=name, attr=attr, prefix=ic_backbone, in_case_prefix_use=in_case_prefix_use,
                              form=form,
                              print_debug=print_status, nc=nc, anchors=anchors))
            if not print_status:
                print_model(name, attr, form=form, rank=times, index=idx)
            if name in in_case_prefix_use:
                ic_backbone = attr[0]
            save.extend(x % idx for x in ([form] if isinstance(form, int) else form) if x != -1)
            idx += 1

    train_able_params, none_train_able_params, total_params, total_layers = 0, 0, 0, 0

    for name, parl in model.named_parameters():
        total_layers += 1
        total_params += parl.numel()
        train_able_params += parl.numel() if parl.requires_grad else 0
        none_train_able_params += parl.numel() if not parl.requires_grad else 0
    total_params, train_able_params, none_train_able_params = str(total_params), str(train_able_params), str(
        none_train_able_params)
    fprint(
        f'Model Created \nTotal Layers {Cp.CYAN}{total_layers}{Cp.RESET}\nNumber Of Route Layers {Cp.CYAN}{len(save)}{Cp.RESET}\n')
    fprint(
        f'Total Params : {Cp.CYAN}{total_params}{Cp.RESET}\nTrain Able Params : {Cp.CYAN}{train_able_params}'
        f'{Cp.RESET}\nNone Train Able Params : {Cp.CYAN}{none_train_able_params}{Cp.RESET}\n')
    return model, save


def attr_exist_check_(attr, index):
    try:
        s = attr[index]
    except IndexError:
        s = []
    return s


def iou(box1, box2):
    print(box1)
    print(box2)
    xma = max(box1[..., 0], box2[..., 0])
    yma = max(box1[..., 1], box2[..., 1])
    xmi = min(box1[..., 2], box2[..., 2])
    ymi = min(box1[..., 3], box2[..., 3])

    i_area = abs(max(xma - xmi, 0) * max(yma - ymi, 0))

    box1_area = abs((box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1]))
    box2_area = abs((box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1]))
    result = i_area / float(box2_area + box1_area - i_area)
    return result


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    print(f'b1_x1 : {b1_x1}')
    print(f'b1_y1 : {b1_y1}')
    print(f'b1_x2 : {b1_x2}')
    print(f'b1_y2 : {b1_y2}')
    print(f'b2_x1 : {b2_x1}')
    print(f'b2_y1 : {b2_y1}')
    print(f'b2_x2 : {b2_x2}')
    print(f'b2_y2 : {b2_y2}')
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


class TorchBaseModule(nn.Module):
    def __init__(self):
        super(TorchBaseModule, self).__init__()
        self.optimizer = None
        self.network = None
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def optim(self):
        """
        Init the Optimizer In this function
        """
        return NotImplementedError

    def model(self):
        """
        Model the Optimizer In this function
        """
        return NotImplementedError

    def set_model(self):
        """
        :param model: any torch.Module Subs
        :return: model to base class for init
        """
        self.network = self.model()
        self.optimizer = self.optim()

    def forward_once(self, x):
        return NotImplementedError

    def forward(self):
        return NotImplementedError

    def jit_save(self, input_size, net, name, **saves):
        model_ckpt = {f'{k}': f"{v}" for k, v in saves.items()}
        di = torch.randn(input_size).to(self.DEVICE)
        j = torch.jit.trace(net, di, check_trace=False)
        s = torch.jit.script(j)
        torch.jit.save(s, name,
                       model_ckpt
                       )

    def load(self, path):
        return NotImplementedError


class M(nn.Module):
    """
    this class is same as nn.Module but have hyper parameters
    """

    def __init__(self):
        super(M, self).__init__()
        self.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.hp = []
        self.hyp = []

    def load_hyp(self, path: Union[str, os.PathLike], log: bool = False):
        assert (path.endswith('.yaml')), f'Selected file should be in yaml format but we got {path[path.index("."):]}'
        data = yaml.safe_load(open(path, 'r'))
        if len(data) != 0:
            for d in data:
                _d = d
                attar_print(_d_=data[d])
                if not hasattr(self, d):
                    setattr(self, d, data[f'{d}'])
                else:
                    nd = str(d) + '_'
                    fprint(
                        f'The attribute that you passed in hyperparameter already exist [Action => Changing name from {d} to {nd}]'
                    )
                    setattr(self, nd, data[nd])
                self.hyp.append({d: data[d]})

    def jit_save(self, input_size: Union[list[int], tuple[int]], net, name: str = 'model.pt', **saves):
        model_ckpt = {f'{k}': f"{v}" for k, v in saves.items()}
        di = torch.randn(input_size).to(self.DEVICE)
        j = torch.jit.trace(net, di, check_trace=False)
        s = torch.jit.script(j)
        torch.jit.save(s, name,
                       model_ckpt
                       )


def anchor_prediction(w: list, h: list, n_clusters: int, original_height: int = 640, original_width: int = 640,
                      c_number: int = 640):
    w = np.asarray(w)
    h = np.asarray(h)
    x = [w, h]
    x = np.asarray(x)
    x = np.transpose(x)

    k_mean = KMeans(n_clusters=n_clusters)
    k_mean.fit(x)
    predicted_anchors = k_mean.predict(x)
    anchors = []
    for idx in range(n_clusters):
        anchors.append(np.mean(x[predicted_anchors == idx], axis=0))
    anchors = np.array(anchors)
    anchors_copy = anchors.copy()
    anchors[..., 0] = anchors_copy[..., 0] / original_width * c_number
    anchors[..., 1] = anchors_copy[..., 1] / original_height * c_number
    anchors = np.rint(anchors)
    anchors.sort(axis=0)
    anchors = anchors.reshape((3, 6))

    return anchors
