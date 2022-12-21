import sys
import cv2 as cv
import keyboard
import numpy as np

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


def attar_print(keys: [str, list[str]], values: [str, list[str]], color: str = Cp.CYAN, **kwargs):
    assert len(keys) == len(values), 'Keys And Vals Should Have same size'
    for i, (k, v) in enumerate(zip(keys, values)):
        fprint(f'{k} : {v}', color=color, **kwargs)





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
