import json
import math
import time
from typing import Union

import cv2 as cv
import requests
import toml
import yaml
from moviepy.editor import *
import dataclasses

import torch

# __call__ = ['as_minutes', 'time_since', 'read_video', 'write_video_frame', 'download', 'read_yaml', 'read_json',
#             'read_txt', 'str_to_list', 'wrd_print', 'mp4_to_mp3', 'read_toml', 'file_reader']


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def read_video(path):
    fr = cv.VideoCapture(path)
    return fr


def write_video_frame(video):
    if not os.path.exists('../data/reader'):
        os.mkdir("../data/reader")
    cap = read_video(video)
    f = 0
    while True:
        ret, frame = cap.read()
        # print(ret,frame)
        if ret:
            f += 1
            cv.imshow('windows', frame)
            cv.imwrite(f'reader/{f}.jpg', frame)
            cv.waitKey(1)
            if cv.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


def download(path: [str, os.PathLike], from_file: bool = False):
    if from_file:
        s = open(path, 'r').readlines()
        for u in s:
            u = u.replace('\n', '')
            download(u, from_file=False)
    else:
        get = requests.get(url=path, stream=True)
        name = path.split('/')[-1].replace('%20', '')

        file_size = int(get.headers['content-length'])
        mb_size = int(file_size / 1000000)
        downloaded = 0
        if not os.path.exists(name):
            with open(name, 'wb') as fp:
                start = time.time()
                for chunk in get.iter_content(chunk_size=4096):
                    downloaded += fp.write(chunk)
                    now = time.time()
                    ts = now - start
                    if ts > 1:
                        pct_done = round(downloaded / file_size * 100)
                        speed = round(downloaded / (now - start) / 1024)
                        pct = int((pct_done / 100) * 50)
                        downloads = (pct_done / mb_size) if pct_done != 0 else 0
                        print(
                            f'\r Download [\033[1;36m{"█" * pct}{" " * (50 - pct)}\033[1;37m] '
                            f'[{pct_done} % done] | avg speed {speed} kbps || {int(abs(start - now))}'
                            f' sec :: file size {downloaded / 1000000} / {mb_size} MB',
                            end='')

        else:
            def rt():
                sas: str = input(f'\033[1;31m File already exists [{name}] do you want to replace this file ? [y/n]   ')
                if sas.lower() == 'y':
                    os.remove(name)
                    download(path, from_file)
                elif sas.lower() == 'n':
                    pass
                else:
                    print(f'Wrong input type  : {sas}')
                    rt()

            rt()


def read_yaml(path: Union[str, os.PathLike] = None):
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data


def read_json(path: Union[str, os.PathLike] = None):
    with open(path, 'r') as stream:
        try:
            data = json.load(stream)
        except json.JSONDecodeError as exc:
            print(exc)
    return data


def read_txt(path: Union[str, os.PathLike] = None):
    data = []
    with open(path, 'r') as r:
        data.append([v for v in r.readlines()])
    data = [d.replace('\n', '') for d in data[0]]
    return data


def str_to_list(string: str = ''):
    string += ' '
    cp = [i for i, v in enumerate(string) if v == ' ']
    word = [string[0:idx] if i == 0 else string[cp[i - 1]:idx] for i, idx in enumerate(cp)]
    word = [w.replace(' ', '') for w in word]
    return word


def wrd_print(words: list = None, action: str = None):
    for idx, i in enumerate(words):
        name = f'{i=}'.split('=')[0]
        string = f"{name}" + ("." if action is not None else "") + (action if action is not None else "")
        print(
            f'\033[1;36m{idx} : {eval(string)}')


def read_toml(path: Union[str, os.PathLike] = None):
    with open(path, 'r') as r:
        data = toml.load(r)
    return data


def mp4_to_mp3(mp4, mp3):
    fc = AudioFileClip(mp4)
    fc.write_audiofile(mp3)
    fc.close()


def file_reader(path: Union[str, os.PathLike]) -> list:
    """

    :param path: path to crawl
    :return: a list with 2 index, index 1 for directories and index 2 for file paths
    """
    files = [os.path.join(path, s) for s in os.listdir(path) if os.path.exists(os.path.join(path, s)) if
             not os.path.isdir(os.path.join(path, s))]
    dirs = [os.path.join(path, s) for s in os.listdir(path) if os.path.exists(os.path.join(path, s)) if
            os.path.isdir(os.path.join(path, s))]
    if len(dirs) != 0:

        for da in dirs:

            if not len(da) == 0 and not isinstance(da, list):

                d, f = file_reader(da)
                files.append(f)
                dirs.append(d)
            else:
                return [[x for x in dirs if x], [x for x in files if x]]
    else:
        return [[x for x in dirs if x], [x for x in files if x]]


class GB:
    def __init__(self, train_data, eval_data, batch_size, chunk_size):
        self.train_data = train_data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def __call__(self, *args, **kwargs):
        return self.forward(*args, *kwargs)

    def forward(self, mode: str, *args, **kwargs):
        data = self.train_data if mode == 'train' else self.eval_data
        ix = torch.randint(len(data) - self.chunk_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.chunk_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.chunk_size + 1] for i in ix])
        return x, y


def save_model(name: str = 'model_save.pt', **kwargs):
    v = {**kwargs}

    torch.save(v, name)


def tokenize_words(word: list, first_word_token: int = 0, swap: int = 1001, last_word_token: int = 1002,
                   pad_index: int = 1003):
    """
    :param swap:
    :param pad_index:
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: 0 for start token | 1002 for end token
    """
    word = [(swap if w == 0 else w) for w in word]
    word = [first_word_token] + word
    word.append(last_word_token)
    word.append(pad_index)
    return word


def detokenize_words(word: list, first_word_token: int = 0, last_word_token: int = 1002, pad_index: int = 1003):
    """
    :param pad_index:
    :param last_word_token:
    :param first_word_token:
    :param word: index
    :return: un tokenized words
    """

    w = [(first_word_token if w == last_word_token - 1 else w) for w in
         [w for w in word if w not in [last_word_token, first_word_token]]]
    del w[-1]
    # print(f'W : {w}')
    return w


@dataclasses.dataclass
class CF:
    ...


def create_config(
        model_type: str = 'PGT-s',
        num_embedding: int = 512,
        num_heads: int = 8,
        chunk: int = 256,
        vocab_size: int = 5000,
        num_layers: int = 2,
        scale_attn_by_layer_idx: bool = False,
        use_mask: bool = True,
        attn_dropout: float = 0.2,
        residual_dropout: float = 0.2,
        activation: str = "gelu_new",
        embd_pdrop: float = 0.1,
        epochs: int = 500,
        lr: float = 4e-4,
        pad_token_id: int = 0,
        create_attention_mask: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        weight_decay: float = 2e-1,
        **kwargs

):
    intermediate_size: int = num_embedding * 4
    hidden_size: int = num_embedding
    max_len = chunk
    max_position_embeddings = max_len
    ttl = ['max_position_embeddings', 'hidden_size',
           'intermediate_size', 'device', 'lr', 'chunk',
           'embd_pdrop', 'activation', 'epochs', 'pad_token_id',
           'create_attention_mask',
           'residual_dropout', 'attn_dropout', 'weight_decay',
           'use_mask', 'scale_attn_by_layer_idx',
           'num_layers', 'vocab_size',
           'max_len', 'num_heads', 'num_embedding']
    cash = CF()
    for t in ttl:
        cash.__setattr__(t, eval(t))
    v = {**kwargs}
    if len(v) != 0:
        for k, v in v.items():
            cash.__setattr__(k, v)

    return cash


def make2d(tensor):
    return tensor.view(-1, tensor.size(-1))


def get_pgt_config_by_name(name: str = 'PGT-s', vocab_size: int = 5000,
                           device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> create_config:
    """
    :param device: device for model
    :param vocab_size: vocab_size
    :param name: name of the type of model you want to get config
    [chooses] = ['PGT-ss']['PGT-s']['PGT-m']['PGT-x']['PGT-l']['PGT-A']
    :return: Config
    """
    if name == 'PGT-Cs':
        return create_config(
            name,
            num_embedding=256,
            num_heads=8,
            epochs=1000,
            num_layers=6,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            lr=4e-4,
            use_mask=True
        )
    if name == 'PGT-As':
        return create_config(
            name,
            num_embedding=624,
            num_heads=12,
            epochs=1000,
            num_layers=10,
            device=device,
            vocab_size=vocab_size,
            chunk=128,
            lr=4e-4,
            use_mask=True
        )
    elif name == 'PGT-s':
        return create_config(
            name,
            num_embedding=256,
            num_heads=8,
            num_layers=4,
            device=device,
            vocab_size=vocab_size,
            chunk=64,
            use_mask=True
        )
    elif name == 'PGT-m':
        return create_config(
            name,
            num_embedding=512,
            num_heads=8,
            num_layers=8,
            device=device,
            vocab_size=vocab_size,
            chunk=128,
            use_mask=True
        )
    elif name == 'PGT-x':
        return create_config(
            name,
            num_embedding=512,
            num_heads=16,
            num_layers=14,
            device=device,
            vocab_size=vocab_size,
            chunk=256,
            use_mask=True
        )
    elif name == 'PGT-l':
        return create_config(
            name,
            num_embedding=728,
            num_heads=14,
            num_layers=20,
            device=device,
            vocab_size=vocab_size,
            chunk=512,
            use_mask=True
        )
    elif name == 'PGT-A':
        prp = torch.cuda.get_device_properties("cuda")
        print(f'\033[1;32mWarning You Loading the Largest Model on {prp.name} : {prp.total_memory / 1e9} GB')
        return create_config(
            name,
            num_embedding=1024,
            num_heads=32,
            num_layers=42,
            device=device,
            vocab_size=vocab_size,
            chunk=728,
            use_mask=True
        )
    else:
        raise NameError(
            f"Valid Names for Model are ['PGT-Cs']['PGT-As']['PGT-s']['PGT-m']['PGT-x']['PGT-l']['PGT-A'] | [ERROR : Unknown {name} type]")
