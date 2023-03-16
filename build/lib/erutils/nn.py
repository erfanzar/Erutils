import torch
import torch.nn as nn

from .lightning import M

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import math
from collections import OrderedDict

import torch
from packaging import version
from torch import Tensor, nn
from typing import Any ,Dict,List,Union,Optional



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        """_summary_

        Args:
            dim (_type_): int(head_size * rotary_prc)
            max_position_embeddings (_type_): max_sentecne_length
            base (int, optional): _description_. Defaults to 10000.
            device (_type_, optional): _description_. Defaults to None.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...].to(x.device), self.sin_cached[:seq_len, ...].to(x.device)



class HyperParameters(object):
    def __init__(self, **kwargs):
        self.model_type: Optional[str] = kwargs.pop('model_type', 'PGT-s')
        self.num_embedding: Optional[int] = kwargs.pop('num_embedding', 512)
        self.intermediate_size: Optional[int] = kwargs.pop('intermediate_size', 5)
        self.num_heads: Optional[int] = kwargs.pop('num_heads', 8)
        self.chunk: Optional[int] = kwargs.pop('chunk', 256)
        self.vocab_size: Optional[int] = kwargs.pop('vocab_size', 5000)
        self.num_layers: Optional[int] = kwargs.pop('num_layers', 2)
        self.scale_attn_by_layer_idx: Optional[bool] = kwargs.pop('scale_attn_by_layer_idx', False)
        self.use_mask: Optional[bool] = kwargs.pop('use_mask', True)
        self.attn_dropout: Optional[float] = kwargs.pop('attn_dropout', 0.1)
        self.residual_dropout: Optional[float] = kwargs.pop('residual_dropout', 0.2)
        self.activation: Optional[str] = kwargs.pop('activation', "gelu_new")
        self.embedded_dropout: Optional[float] = kwargs.pop('embedded_dropout', 0.15)
        self.epochs: Optional[int] = kwargs.pop('epochs', 500)
        self.lr: Optional[float] = kwargs.pop('lr', 4e-4)
        self.pad_token_id: Optional[int] = kwargs.pop('pad_token_id', 0)
        self.create_attention_mask: Optional[bool] = kwargs.pop('create_attention_mask', False)
        self.device: Optional[str] = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_decay: Optional[float] = kwargs.pop('weight_decay', 2e-1, )
        for k, v in kwargs.items():
            if k not in self:
                setattr(self, k, v)


class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.
    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            raise ImportError(
                f"You are using torch=={torch.__version__}, but torch>=1.12.0 is required to use "
                "PytorchGELUTanh. Please upgrade torch."
            )

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.gelu(input, approximate="tanh")


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.
    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.silu(input)


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": SiLUActivation,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")


# from hugging face :)


class Conv(M):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1,
                 activation: Union[str, Any] = None,
                 form: int = -1):
        super(Conv, self).__init__()
        self.form = form
        self.to(DEVICE)
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s,
                              padding=p if p is not None else (1 if k == 3 else 0), groups=g).to(DEVICE)
        nn.init.xavier_normal_(self.conv.weight.data)

        self.activation = (
            eval(activation) if isinstance(activation, str) else activation
        ) if activation is not None else nn.SiLU()

        self.batch_norm = nn.BatchNorm2d(c2)

    def forward(self, x) -> torch.Tensor:
        x = self.batch_norm(self.activation(self.conv(x)))
        return x


class Concat(M):
    def __init__(self, dim, form):
        super(Concat, self).__init__()
        self.form = form
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


class Neck(M):
    def __init__(self, c1, c2, e=0.5, shortcut=False, form: int = -1):
        super(Neck, self).__init__()
        c_ = int(c2 * e)
        self.form = form
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.cv2 = Conv(c_, c2, k=3, s=1, p=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        ck = self.cv2(self.cv1(x[self.form]))

        k = x + ck if self.add else ck

        return k


class C3(M):
    def __init__(self, c1, c2, e=0.5, n=1, shortcut=True, form: int = -1):
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.form = form
        self.cv1 = Conv(c1, c_, k=3, s=1, p=1)
        self.cv2 = Conv(c1, c_, k=3, s=1, p=1)
        self.cv3 = Conv(c_ * 2, c2, k=3, p=1)
        self.m = nn.Sequential(*(Neck(c_, c_, shortcut=shortcut, e=0.5) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv2(x)), self.cv1(x)), dim=1))


class C4P(C3):
    def __init__(self, c, e=0.5, n=1, ct=2, form: int = -1):
        super(C4P, self).__init__(c1=c, c2=c, e=e, n=n)
        self.form = form
        self.ct = ct

    def forward(self, x):
        for _ in range(self.ct):
            x = self.cv3(torch.cat((self.m(self.cv2(x)), self.cv1(x)), dim=1)) + x
        return x


class RepConv(M):
    def __init__(self, c, e=0.5, n=3, form: int = -1):
        super(RepConv, self).__init__()
        c_ = int(c * e)
        self.form = form
        self.layer = nn.ModuleList()
        # self.layer.append(
        #     *(Conv(c1=c if i == 0 else c_, c2=c_ if i == 0 else c, kernel_size=3, padding=1, stride=1, batch=False)
        #       for i in range(n)))
        for i in range(n):
            self.layer.append(
                Conv(c1=c if i == 0 else c_, c2=c_ if i == 0 else c, k=3, p=1, s=1))

    def forward(self, x):
        x_ = x
        for layer in self.layer:
            x = layer.forward(x)
        return x_ + x


class ConvSc(RepConv):
    def __init__(self, c, n=4, form: int = -1):
        super(ConvSc, self).__init__(c=c, e=1, n=n)
        self.form = form

    def forward(self, x):
        x_ = x.detach().clone()
        for layer in self.layer:
            x = layer(x) + x
        return x + x_


class ResidualBlock(M):
    def __init__(self, c1, n: int = 4, use_residual: bool = True, form: int = -1):
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.n = n
        self.to(DEVICE)
        self.layer = nn.ModuleList()
        self.form = form

        for _ in range(n):
            self.layer.append(
                nn.Sequential(
                    Conv(c1, c1 * 2, s=1, p=0, k=1),
                    Conv(c1 * 2, c1, s=1, p=1, k=3)
                )
            )

    def forward(self, x) -> torch.Tensor:
        c = x
        for layer in self.layer:
            x = layer(x)
        return x + c if self.use_residual else x


class Detect(M):
    stride = False
    interface = False

    def __init__(self, nc=4, anchors=(), ch=(), form=None):  # detection layer
        super(Detect, self).__init__()
        if form is None:
            form = [-1, -2, -3]
        self.form = form
        self.nc = nc

        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)

        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.training = True

    def forward(self, x):

        z = []  # inference
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.interface:

            for p in x:
                # print(p.shape)
                z.append(p.view(bs, -1, self.no))
            x = torch.cat(z, 1)
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix
        return (box, score)


class CV1(M):
    def __init__(self, c1, c2, e=0.5, n=1, shortcut=False, dim=-3, form: int = -1):
        super(CV1, self).__init__()
        c_ = int(c2 * e)
        if shortcut:
            c2 = c1
        self.c = Conv(c1, c_, k=3, p=1, s=1)
        self.form = form
        self.v = Conv(c1, c_, k=3, p=1, s=1)
        self.m = nn.Sequential(
            *(Conv(c_ * 2 if i == 0 else c2, c2, k=3, s=1, p=1) for i in range(n)))
        self.sh = c1 == c2
        self.dim = dim

    def forward(self, x):
        c = torch.cat((self.c(x), self.v(x)), dim=self.dim)
        return self.m(c) if not self.sh else self.m(
            torch.cat((self.c(x), self.v(x)), dim=self.dim)) + x


class UC1(M):
    def __init__(self, c1, c2, e=0.5, dim=-3, form: int = -1):
        super(UC1, self).__init__()
        self.form = form
        c_ = int(c2 * e)
        self.c = Conv(c1=c1, c2=c_, k=1, s=1)
        self.v = Conv(c1=c1, c2=c_, k=1, s=1)
        self.m = Conv(c1=c_, c2=c2, k=1, s=1)
        self.dim = dim

    def forward(self, x):
        return self.m(torch.cat((self.c(x), self.v(x)), dim=self.dim))


class MP(M):
    def __init__(self, k=2, form: int = -1):
        super(MP, self).__init__()
        self.form = form
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        x = self.m(x)
        return x


class SP(M):
    def __init__(self, k=3, s=1, form: int = -1):
        super(SP, self).__init__()
        self.form = form
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        x = self.m(x)
        return x


class LP(M):
    def __init__(self, dim: int = None):
        super(LP, self).__init__()
        self.dim = dim

    def forward(self, l1, l2, dim_f: int = 1):
        return torch.cat((l1, l2), dim=dim_f if self.dim is None else self.dim)


class UpSample(M):
    def __init__(self, s: int = 2, m: str = 'nearest', form: int = -1):
        super(UpSample, self).__init__()
        self.form = form
        self.u = nn.Upsample(scale_factor=s, mode=m)

    def forward(self, x):
        x = self.u(x)
        return x


class SPPCSPC(M):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        x = self.cv7(torch.cat((y1, y2), dim=1))
        return x


from dataclasses import dataclass
from typing import Union, Optional

try:

    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except:
    print('Downloading Missing Module [pytorch]')
    import subprocess
    import sys

    path = sys.executable
    subprocess.run(f'{path} -m pip install torch')
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

import math

__all__ = ['MultiHeadBlock', 'MultiHeadAttention', 'Head', 'FeedForward', 'Decoder', 'Encoder', 'CasualBlock',
           'PGTBlock', 'Conv1D']


@torch.jit.script  # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class Head(nn.Module):
    def __init__(self, n_embedded: int, head_size: int):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embedded, head_size, bias=False)
        self.query = nn.Linear(n_embedded, head_size, bias=False)
        self.value = nn.Linear(n_embedded, head_size, bias=False)

    def forward(self, k: torch.Tensor, q: torch.Tensor = None, v: torch.Tensor = None, mask=None):
        # if q is not None and v is not None:
        assert k.shape == q.shape and q.shape == v.shape
        b, t, c = k.shape
        key = self.key(k)
        query = self.query(q)
        value = self.value(v)
        attn = query @ key.transpose(-2, -1) * c ** -0.5
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = nn.functional.softmax(attn, dim=-1)
        value = attn @ value
        return value, attn


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, number_of_embedded: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(number_of_embedded, 4 * number_of_embedded),
            nn.ReLU(),
            nn.Linear(4 * number_of_embedded, number_of_embedded),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num, head_size: int, number_of_embedded: int):
        super(MultiHeadAttention, self).__init__()

        self.m = nn.ModuleList([Head(head_size=head_size, n_embedded=number_of_embedded) for _ in range(num)])
        self.proj = nn.Linear(number_of_embedded, number_of_embedded)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.cat([h(x, x, x, torch.tril(torch.ones(x.shape[1], x.shape[1])) if i == 0 else None) for i, h in
                       enumerate(self.m)],
                      dim=-1)
        x = self.dp(self.proj(x))
        return x


class MultiHeadBlock(nn.Module):
    def __init__(self, number_of_head, number_of_embedded: int):
        super(MultiHeadBlock, self).__init__()
        head_size = number_of_embedded // number_of_head
        self.sa = MultiHeadAttention(number_of_head, head_size=head_size,
                                     number_of_embedded=number_of_embedded)
        self.ffwd = FeedForward(number_of_embedded=number_of_embedded)
        self.ln1 = LayerNorm(number_of_embedded)
        self.ln2 = LayerNorm(number_of_embedded)

    def forward(self, x):
        x = x + self.ln1(self.sa(x, x, x))
        x = x + self.ln2(self.ffwd(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_head: int, chunk: int):
        super(CausalSelfAttention, self).__init__()
        assert \
            number_of_embedded % number_of_head == 0, \
            'number_of_embedded % number_of_head == 0 Failed Make' \
            ' Sure that number_of_embedded is equal to number_of_head'
        self.number_of_embedded = number_of_embedded
        self.number_of_head = number_of_head
        self.attn = nn.Linear(number_of_embedded, 3 * number_of_embedded)
        self.proj = nn.Linear(number_of_embedded, number_of_embedded)
        self.register_buffer('bias', torch.tril(torch.ones(chunk, chunk).view(1, 1, chunk, chunk)))
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.2)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.number_of_embedded, dim=2)
        q = q.view(B, T, self.number_of_head, C // self.number_of_head).transpose(1, 2)
        k = k.view(B, T, self.number_of_head, C // self.number_of_head).transpose(1, 2)
        v = v.view(B, T, self.number_of_head, C // self.number_of_head).transpose(1, 2)

        attn = q @ k.transpose(-2, -1) * (1.0 / torch.sqrt(k.size(0)))
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        attn = self.dp1(attn)
        attn = attn @ v
        attn = attn.transpose(2, 1).contiguous().view(B, T, C)
        attn = self.dp2(self.proj(attn))
        return attn


class MLP(nn.Module):
    def __init__(self, number_of_embedded: int):
        super(MLP, self).__init__()
        self.li1 = nn.Linear(number_of_embedded, 4 * number_of_embedded)
        self.li2 = nn.Linear(4 * number_of_embedded, number_of_embedded)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dp(self.li2(new_gelu(self.li1(x))))
        return x


class CasualBlock(nn.Module):
    def __init__(self, number_of_embedded: int, number_of_head: int):
        super(CasualBlock, self).__init__()
        self.ln1 = LayerNorm(number_of_embedded)
        self.sc = CausalSelfAttention(number_of_embedded=number_of_embedded, number_of_head=number_of_head)
        self.ln2 = LayerNorm(number_of_embedded)
        self.mlp = MLP(number_of_embedded=number_of_embedded)

    def forward(self, x):
        x = x + self.sc(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@torch.jit.script  # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@dataclass
class Conf:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Dropout = 0.2


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedded: int):
        super(Embedding, self).__init__()
        self.m = nn.Embedding(vocab_size, embedded)

    def forward(self, x):
        return self.m(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_length: int, embedded: int):
        super(PositionalEncoding, self).__init__()
        tensor = torch.zeros((max_length, embedded))
        self.embedded = embedded
        for pos in range(max_length):
            for i in range(0, embedded, 2):
                tensor[pos, i] = math.sin(pos / (10_000 ** ((2 * i) / embedded)))
                tensor[pos, i + 1] = math.cos(pos / (10_000 ** ((2 * (i + 1)) / embedded)))
        self.register_buffer('tensor', tensor)

    def forward(self, x):
        x = x * math.sqrt(self.embedded)
        # print(x.shape)
        # print(self.tensor.shape)
        max_length = x.size(1)
        x = x + torch.autograd.Variable(self.tensor[:max_length, :], requires_grad=False)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embedded: int, number_of_heads: int):
        super(SelfAttention, self).__init__()
        c = embedded // number_of_heads
        assert (c * number_of_heads == embedded)
        self.c = c
        self.embedded = embedded
        self.number_of_heads = number_of_heads
        self.key = nn.Linear(embedded, embedded, bias=False)
        self.queries = nn.Linear(embedded, embedded, bias=False)
        self.value = nn.Linear(embedded, embedded, bias=False)
        self.fc = nn.Linear(embedded, embedded)
        self.dp = nn.Dropout()

    def forward(self, k, q, v, mask=None):
        b, t, c = k.shape
        k = self.key(k)
        q = self.queries(q)
        v = self.value(v)

        k = k.view(b, t, self.number_of_heads, self.c).transpose(1, 2)
        q = q.view(b, t, self.number_of_heads, self.c).transpose(1, 2)
        v = v.view(b, t, self.number_of_heads, self.c).transpose(1, 2)

        # DotScale
        attn = q @ k.transpose(-2, -1) * (math.sqrt(self.c))
        # print(f'ATTN : {attn.shape} ')
        # print(f'MASK : {mask.shape}')
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        attn = self.dp(attn)

        attn = attn @ v

        attn = self.fc(attn.transpose(1, 2).contiguous().view(b, t, c))
        return attn


class FFD(nn.Module):
    def __init__(self, embedded: int):
        super(FFD, self).__init__()
        self.m = nn.Sequential(
            nn.Linear(embedded, embedded * 4),
            nn.ReLU(),
            nn.Dropout(Conf.Dropout),
            nn.Linear(4 * embedded, embedded)
        )

    def forward(self, x):
        return self.m(x)


class EncoderLayer(nn.Module):
    def __init__(self, embedded: int, number_of_heads: int):
        super(EncoderLayer, self).__init__()
        self.ln1 = LayerNorm(embedded)
        self.attn = SelfAttention(embedded, number_of_heads)
        self.ln2 = LayerNorm(embedded)
        self.dp1 = nn.Dropout(Conf.Dropout)
        self.dp2 = nn.Dropout(Conf.Dropout)
        self.ff = FFD(embedded)

    def forward(self, x, src_mask):
        xl = self.ln1(x)
        ka = self.dp1(self.attn(xl, xl, xl, src_mask))
        # print(f'KA DIM : {ka.shape}')
        x = ka + x
        xl = self.ln2(x)
        x = self.dp2(self.ff(xl)) + x
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embedded: int, number_of_heads: int, number_of_layers: int):
        super(Encoder, self).__init__()
        self.embedded = embedded
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        self.layers = nn.ModuleList([EncoderLayer(embedded, number_of_heads) for _ in range(number_of_layers)])

        self.token = Embedding(vocab_size, embedded)
        self.position = PositionalEncoding(max_length, embedded)
        self.ln = LayerNorm(embedded)

    def forward(self, x, src_mask):
        # print('-' * 20)
        # print(f'INPUT TO DECODER : {x.shape}')
        x = self.position(self.token(x))
        # print(f'TOKENS : {x.shape}')
        # print('-' * 20)
        for i, m in enumerate(self.layers):
            # print(f'RUNNING ENCODER {i} : {x.shape}')
            x = m(x, src_mask)
        return self.ln(x)


class DecoderLayer(nn.Module):
    def __init__(self, embedded: int, number_of_heads: int):
        super(DecoderLayer, self).__init__()
        self.ln1 = LayerNorm(embedded)
        self.ln2 = LayerNorm(embedded)
        self.ln3 = LayerNorm(embedded)

        self.attn1 = SelfAttention(embedded, number_of_heads)
        self.attn2 = SelfAttention(embedded, number_of_heads)

        self.dp1 = nn.Dropout(Conf.Dropout)
        self.dp2 = nn.Dropout(Conf.Dropout)
        self.dp3 = nn.Dropout(Conf.Dropout)
        self.ff = FFD(embedded)

    def forward(self, x, enc_out, src_mask, trg_mask):
        lx = self.ln1(x)
        x = self.dp1(self.attn1(lx, lx, lx, trg_mask)) + x
        lx = self.ln2(x)
        x = self.dp2(self.attn2(lx, enc_out, enc_out, src_mask)) + x
        lx = self.ln3(x)
        x = self.dp3(self.ff(lx)) + x
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, embedded: int, number_of_heads: int, number_of_layers: int,
                 ):
        super(Decoder, self).__init__()
        self.embedded = embedded
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        self.layers = nn.ModuleList([DecoderLayer(embedded, number_of_heads) for _ in range(number_of_layers)])
        self.fc = nn.Linear(embedded, embedded)
        self.token = Embedding(vocab_size, embedded)
        self.position = PositionalEncoding(max_length, embedded)
        self.ln = LayerNorm(embedded)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.position(self.token(x))
        for m in self.layers:
            x = m(x, enc_out, src_mask, trg_mask)
        return self.fc(self.ln(x))


# =========================================================> PGT => models


class Conv1D(nn.Module):
    def __init__(self, c1, c2):
        super(Conv1D, self).__init__()
        self.c2 = c2
        w = torch.empty(c1, c2)
        nn.init.normal_(w, std=0.2)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(c2))

    def forward(self, x):
        new_shape = x.size()[:-1] + (self.c2,)

        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(new_shape)

        return x


class MultiCNNAttention(nn.Module):
    def __init__(self, config, layer_idx=None, use_mask: bool = None):
        super(MultiCNNAttention, self).__init__()
        self.layer_idx = layer_idx
        self.embedding = config.num_embedding
        self.num_heads = config.num_heads
        self.num_div = self.embedding // self.num_heads
        self.scale_attn_by_layer_idx = config.scale_attn_by_layer_idx
        self.use_mask = config.use_mask if use_mask is None else use_mask
        if self.num_heads // self.embedding != 0:
            raise ValueError(
                f'hidden_size must be dividable to num_heads {self.num_heads} // {self.embedding} = {self.num_heads // self.embedding}'
            )
        self.c_attn = Conv1D(self.embedding, self.embedding * 3)
        self.c_proj = Conv1D(self.embedding, self.embedding)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.register_buffer('bias', torch.tril(
            torch.ones(config.chunk, config.chunk, dtype=torch.uint8, device=config.device).view(1, 1,
                                                                                                 config.chunk,
                                                                                                 config.chunk)))

        self.register_buffer('masked_bias', torch.tensor(float(-1e4)))

    def _split_heads(self, tensor: torch.Tensor):
        new_shape = tensor.size()[:-1] + (self.num_heads, self.num_div)
        # print(f'Shape : {new_shape}')
        tensor = tensor.view(new_shape).permute(0, 2, 1, 3)
        return tensor

    def _merge_heads(self, tensor: torch.Tensor):
        tensor = tensor.permute(0, 2, 1, 3)
        new_shape = tensor.size()[:-2] + (self.num_heads * self.num_div,)
        return tensor.reshape(new_shape)

    def _attn(self, query, key, value, attention_mask, head_mask):
        attn_weight = torch.matmul(query, key.transpose(-2, -1))

        attn_weight = attn_weight / torch.full([], value.size(-1) ** 0.5, dtype=attn_weight.dtype,
                                               device=attn_weight.device)
        if self.scale_attn_by_layer_idx:
            attn_weight /= self.layer_idx
        if self.use_mask:
            key_len, query_len = key.size(-2), query.size(-2)
            masked = self.bias[:, :, key_len - query_len:query_len, :key_len].to(attn_weight.device)
            attn_weight = attn_weight.masked_fill(masked == 0, self.masked_bias)
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_weight = attn_weight + attention_mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        attn_weight = attn_weight.type(value.dtype)
        if head_mask is not None:
            attn_weight = attn_weight * head_mask

        attn_weight = torch.matmul(attn_weight, value)
        return attn_weight

    def forward(self, hidden_state: Optional[torch.Tensor], attention_mask=None, head_mask=None):
        query, key, value = self.c_attn(hidden_state).split(self.embedding, dim=len(hidden_state.shape) - 1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        attn_output = self._attn(query=query, key=key, value=value, attention_mask=attention_mask, head_mask=head_mask)
        attn_output = self.residual_dropout(self.c_proj(self._merge_heads(attn_output)))
        return attn_output


class PGTMLP(nn.Module):
    def __init__(self, config):
        super(PGTMLP, self).__init__()
        self.c_op = Conv1D(config.num_embedding, config.num_embedding * config.intermediate_size)
        self.c_proj = Conv1D(config.num_embedding * config.intermediate_size, config.num_embedding)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.act = get_activation(config.activation)

    def forward(self, hidden_state):
        hidden_state = self.c_op(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.c_proj(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class PGTBlock(nn.Module):
    def __init__(self, config, layer_idx_1=None, layer_idx_2=None):
        super(PGTBlock, self).__init__()
        # self.ln1 = nn.LayerNorm(config.num_embedding)
        # self.ln2 = nn.LayerNorm(config.num_embedding)
        # self.ln3 = nn.LayerNorm(config.num_embedding)
        # self.h_1 = MultiCNNAttention(config=config, layer_idx=layer_idx_1)
        # self.h_2 = MultiCNNAttention(config=config, layer_idx=layer_idx_2, use_mask=False)
        # self.mlp = PGTMLP(config)
        self.ln1 = nn.LayerNorm(config.num_embedding)
        self.ln2 = nn.LayerNorm(config.num_embedding)
        self.h_1 = MultiCNNAttention(config=config, layer_idx=layer_idx_1)
        self.mlp = PGTMLP(config)

    def forward(self, hidden_state, attention_mask=None, heads_mask=None):
        # residual = hidden_state
        # hidden_state = self.ln1(hidden_state)
        # hidden_state = self.h_1(hidden_state, attention_mask, heads_mask) + residual
        #
        # residual = hidden_state
        # hidden_state = self.ln2(hidden_state)
        # hidden_state = self.h_2(hidden_state, attention_mask, heads_mask) + residual
        #
        # residual = hidden_state
        # hidden_state = self.ln3(hidden_state)
        # hidden_state = self.mlp(hidden_state) + residual
        # return hidden_state
        residual = hidden_state
        hidden_state = self.ln1(hidden_state)
        hidden_state = self.h_1(hidden_state, attention_mask, heads_mask) + residual

        residual = hidden_state
        hidden_state = self.ln2(hidden_state)
        hidden_state = self.mlp(hidden_state) + residual
        return hidden_state


class CC_PGT_Block(nn.Module):
    def __init__(self, config, layer_idx: int = None):
        super(CC_PGT_Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.num_embedding)
        self.ln2 = nn.LayerNorm(config.num_embedding)
        self.h = MultiCNNAttention(config=config, layer_idx=layer_idx)
        self.mlp = PGTMLP(config=config)

    def forward(self, hidden_state, attention_mask=None, heads_mask=None):
        return self.mlp(self.ln2(hidden_state)) + self.h(
            self.ln1(hidden_state), attention_mask=attention_mask)
