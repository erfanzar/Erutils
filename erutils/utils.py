import json
import math
import time
from typing import Union

import cv2 as cv
import requests
import toml
import yaml
from moviepy.editor import *


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
