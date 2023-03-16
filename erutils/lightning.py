import math
import os
from typing import Union, List, Tuple, Optional,OrderedDict,Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.cluster import KMeans

from .loggers import fprint, Cp, print_model, attar_print

import time

import onnxruntime

import cv2

class_names = ['BOTTOM', 'LEFT', 'RIGHT', 'TOP']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, labels_inc=None):
    mask_img = image.copy()
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        label = labels_inc[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(det_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)
        cv2.putText(det_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(text=name1, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img1, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (0, 115, 255), -1)
    cv2.putText(img1, name1,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    (tw, th), _ = cv2.getTextSize(text=name2, fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                  fontScale=fontsize, thickness=text_thickness)
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(img2, (x1 - offset * 2, y1 + offset),
                  (x1 + tw + offset * 2, y1 - th - offset), (94, 23, 235), -1)

    cv2.putText(img2, name2,
                (x1, y1),
                cv2.FONT_HERSHEY_DUPLEX, fontsize,
                (255, 255, 255), text_thickness)

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img


class ONNXRun:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, labels=None):
        if labels is None:
            labels = ['None']
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.labels = labels
        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, labels_inc=self.labels)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


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


def pars_model_v2(cfg, c_req: Union[list[str], str], detail: str = None, print_status: bool = False, sc: int = 3,
                  imports: Union[list[str], str] = None):
    """

    :param imports: before creating layers or adding neruons to layer you must import them into function
    :param cfg: a list of lists contain 4 parameters like [index:[int,list[int ]],number:int,name:str,args:[Any]];
    :param c_req: channel request Models
    :param detail: Full detail log
    :param print_status: Status
    :param sc: Start Channel
    :return:
    """
    if imports is not None:
        if isinstance(imports, list):
            for imp in imports:
                exec(imp)
        elif isinstance(imports, str):
            exec(imports)

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
        ffa = None if f == -1 else f % i if isinstance(f, int) else [None if k == -1 else k % i for k in f]

        saves.append(ffa)

        m = eval(model_name)
        setattr(m, 'f', f)
        model.append(m)
    return model, saves


def pars_model(cfg: list, device='cpu', imports: Union[list[str], str] = None):
    """
    :param cfg: a list of lists contain 4 parameters like [index:[int,list[int ]],number:int,name:str,args:[Any]];
    :param device: device that module going to build in default set to *'cpu'*;
    :return: Module
    """

    if imports is not None:
        if isinstance(imports, list):
            for imp in imports:
                exec(imp)
        elif isinstance(imports, str):
            exec(imports)

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
                  form: Union[List, int] = -1, imports: Union[list[str], str] = None,
                  print_debug: bool = True, nc: int = 4, anchors: list = None):
    if imports is not None:
        if isinstance(imports, list):
            for imp in imports:
                exec(imp)
        elif isinstance(imports, str):
            exec(imports)

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


def module_creator(backbone, head, print_status, ic_backbone, nc, anchors, in_case_prefix_use,
                   imports: Union[list[str], str] = None):
    if imports is not None:
        if isinstance(imports, list):
            for imp in imports:
                exec(imp)
        elif isinstance(imports, str):
            exec(imports)

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

    def set_optim(self):
        """
        Init the Optimizer In this function
        """
        return NotImplementedError

    def s_model(self):
        """
        Model the Optimizer In this function
        """
        return NotImplementedError

    def set_model(self):
        """
        :return: model to base class for init
        """
        self.network = self.s_model()
        self.optimizer = self.set_optim()

    def __call__(self, batch):
        return self.forward(batch)

    def forward_once(self, x):
        return NotImplementedError

    def forward(self, batch):
        if NotImplemented:
            return self.forward_once(batch[0])
        else:
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


def shape_broadcast_freq(q: Optional[torch.Tensor], f: Optional[torch.Tensor]) -> torch.Tensor:
    ndim = q.ndim
    assert ndim > 1
    assert f.shape == (q.shape[1], q.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(q.shape)]
    return f.view(*shape)


def rotary_embedding(query: Optional[torch.Tensor], key: Optional[torch.Tensor], freq: Optional[torch.Tensor]) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    _query: torch.Tensor = torch.view_as_complex(query.float().view(*query.shape[:-1], -1, 2))
    _key: torch.Tensor = torch.view_as_complex(key.float().view(*key.shape[:-1], -1, 2))
    freq: torch.Tensor = shape_broadcast_freq(_query, freq)
    q: torch.Tensor = torch.view_as_real(_query * freq).flatten(3)
    k: torch.Tensor = torch.view_as_real(_key * freq).flatten(3)
    return q.type_as(query), k.type_as(key)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[..., offset: q.shape[-2] + offset, :]
    sin = sin[..., offset: q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def build_alibi_tensor(number_of_heads:Optional[int],attention_mask:Optional[torch.Tensor],dtype:torch.dtype):
    assert attention_mask.ndim == 2
    batch,seq_len = attention_mask.shape
    p2 = 2 ** math.floor(math.log2(number_of_heads))
    base = torch.tensor(2**(-(2**-(math.log2(p2)-3))),device=attention_mask.device,dtype=dtype)
    powers = torch.arange(1,1+p2,device=attention_mask.device,dtype=dtype)
    slopes = torch.pow(base,powers)
    if p2 != number_of_heads:
        
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * p2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(p2, number_of_heads - p2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)
    arange_tensor = ((attention_mask.cumsum(-1) -1)*attention_mask)[:,None,:]
    alibi  = slopes[...,None] * arange_tensor
    return alibi.reshape(batch*number_of_heads,-1,seq_len)

class ModelOutput(OrderedDict):

    def __post_init__(self):
        class_fields = self

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none:
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for idx, element in enumerate(iterator):
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())
    
    
    
class BaseModelOutput(ModelOutput):
  
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):


    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None