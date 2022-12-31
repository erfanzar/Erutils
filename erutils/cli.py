import sys
import cv2 as cv
import keyboard
import numpy as np
import matplotlib.pylab as plt


def show_array(array: (np.ndarray, list, tuple)):
    while True:
        if isinstance(array, list):
            array = np.array(array)
        if array.shape[0] == 3:
            array = array.reshape((array.shape[1], array.shape[2], array.shape[0]))
        array = array.astype(np.uint8)
        cv.imshow('show function', array)
        cv.waitKey(1)
        if keyboard.is_pressed('q'):
            break


list_color = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [100, 200, 220],
]


def draw_rec(x1, y1, x2, y2, img_size: int = 640, thickness=3):
    fig, ax = plt.subplots()
    frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    frame[:, :] = [255, 255, 255]
    print(f'x1, y1, x2, y2 : {x1, y1, x2, y2}')
    for i, (_x1, _x2, _y1, _y2) in enumerate(zip(x1, x2, y1, y2)):
        for h in range(img_size):
            for w in range(img_size):
                if _x1 < w < _x2 and _y1 < h < _y2:
                    if _x1 < w < _x1 + thickness or _y1 < h < _y1 + thickness or _x2 - thickness < w < _x2 or _y2 - thickness < h < _y2:
                        frame[h, w] = list_color[i]

    plt.imshow(frame)


class Cp:
    Type = 1
    BLACK = f'\033[{Type};30m'
    RED = f'\033[{Type};31m'
    GREEN = f'\033[{Type};32m'
    YELLOW = f'\033[{Type};33m'
    BLUE = f'\033[{Type};34m'
    MAGENTA = f'\033[{Type};35m'
    CYAN = f'\033[{Type};36m'
    WHITE = f'\033[{Type};1m'
    RESET = f"\033[{Type};39m"


def fprint(*args, color: str = Cp.CYAN, **kwargs):
    print(*(f"{color}{arg}" for arg in args), **kwargs)


def attar_print(end: str = '\n', color: str = Cp.CYAN, **kwargs):
    assert len(kwargs) > 0, 'Keys And Vals Should Have same size'

    fprint((f'{k} : {v} \n' for k, v in kwargs.items()), color=color, end=end)


def print_model(model, args, form, rank, index):
    print('{}  {:<5}{:>20}{:>5}{:>10}    -    {:<25} \n'.format(f'\033[1;39m', f"{index}", f"{form}", f"{rank}",
                                                                f"{model}",
                                                                f"{args}"))


class Logger:
    def __init__(self):
        super(Logger, self).__init__()
        self.data = ''

    def __call__(self, *args, **kwargs):
        sys.stdout.write(f"\r {self.data}")

    def set_desc(self, *args):
        self.data = ''.join(*(d for d in args))

    def end(self):
        sys.stdout.write('\n')

    def flush(self):
        sys.stdout.flush()
