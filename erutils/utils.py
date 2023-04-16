import json
import math
import time
import typing
from typing import Union, Optional
import pefile
import glob
import os
import shutil
from pathlib import Path
import cv2 as cv
import psutil
import requests
import toml
import yaml
# from moviepy.editor import *
import dataclasses

import torch


# __call__ = ['as_minutes', 'time_since', 'read_video', 'write_video_frame', 'download', 'read_yaml', 'read_json',
#             'read_txt', 'str_to_list', 'wrd_print', 'mp4_to_mp3', 'read_toml', 'file_reader']


class HyperParameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


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


def download(path: Union[str, os.PathLike], from_file: bool = False):
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


# def mp4_to_mp3(mp4, mp3):
#     fc = AudioFileClip(mp4)
#     fc.write_audiofile(mp3)
#     fc.close()


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


def device_info() -> None:
    prp = torch.cuda.get_device_properties("cuda")
    memory = psutil.virtual_memory()
    free, total_gpu = torch.cuda.mem_get_info('cuda:0')
    used_gpu = total_gpu - free
    print('\033[1;36m',
          f'DEVICES : [ {torch.cuda.get_device_name()} ] | [ Free : {free / 1e9} GB ] | [ Used : {used_gpu / 1e9} GB ] | '
          f'[ Total : {total_gpu / 1e9} GB ]\n'
          f'RAM : [ Free : {memory.free / 1e9} GB ] | [ Total : {memory.total / 1e9} GB ]')


def get_memory(index: int) -> typing.Tuple[float, float, float]:
    """
    :param index: cuda index
    :return: free,used_gpu,total_gpu memory
    """
    free, total_gpu = torch.cuda.mem_get_info(f'cuda:{index}')
    used_gpu = total_gpu - free
    free, total_gpu, used_gpu = free / 1e9, total_gpu / 1e9, used_gpu / 1e9
    return free, used_gpu, total_gpu


def monitor_function(function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        end = time.perf_counter()
        print(f'\033[1;92m {function.__name__} took {end - start:.6f} seconds to complete')
        return result

    return wrapper


def create_output_path(path: Union[os.PathLike, str], name: Optional[str]):
    pp = Path(os.path.join(path, name))
    pp.mkdir(parents=True, exist_ok=True)
    return pp


def fixer_dll(input_path: Union[str, os.PathLike] = "*.dll", backup: bool = False, recursive: bool = False):
    failures = []
    for file in glob.glob(input_path, recursive=recursive):
        print(f"\n---\nChecking {file}...")
        pe = pefile.PE(file, fast_load=True)
        nvbSect = [section for section in pe.sections if section.Name.decode().startswith(".nv_fatb")]
        if len(nvbSect) == 1:
            sect = nvbSect[0]
            size = sect.Misc_VirtualSize
            aslr = pe.OPTIONAL_HEADER.IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE
            writable = 0 != (sect.Characteristics & pefile.SECTION_CHARACTERISTICS['IMAGE_SCN_MEM_WRITE'])
            print(f"Found NV FatBin! Size: {size / 1024 / 1024:0.2f}MB  ASLR: {aslr}  Writable: {writable}")
            if (writable or aslr) and size > 0:
                print("- Modifying DLL")
                if backup:
                    bakFile = f"{file}_bak"
                    print(f"- Backing up [{file}] -> [{bakFile}]")
                    if os.path.exists(bakFile):
                        print(
                            f"- Warning: Backup file already exists ({bakFile}), not modifying file! Delete the 'bak' to allow modification")
                        failures.append(file)
                        continue
                    try:
                        shutil.copy2(file, bakFile)
                    except Exception as e:
                        print(f"- Failed to create backup! [{str(e)}], not modifying file!")
                        failures.append(file)
                        continue
                # Disable ASLR for DLL, and disable writing for section
                pe.OPTIONAL_HEADER.DllCharacteristics &= ~pefile.DLL_CHARACTERISTICS[
                    'IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE']
                sect.Characteristics = sect.Characteristics & ~pefile.SECTION_CHARACTERISTICS['IMAGE_SCN_MEM_WRITE']
                try:
                    newFile = f"{file}_mod"
                    print(f"- Writing modified DLL to [{newFile}]")
                    pe.write(newFile)
                    pe.close()
                    print(f"- Moving modified DLL to [{file}]")
                    os.remove(file)
                    shutil.move(newFile, file)
                except Exception as e:
                    print(f"- Failed to write modified DLL! [{str(e)}]")
                    failures.append(file)
                    continue

    print("\n\nDone!")
    if len(failures) > 0:
        print("***WARNING**** These files needed modification but failed: ")
        for failure in failures:
            print(f" - {failure}")
