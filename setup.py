from setuptools import setup

if __name__ == "__main__":
    setup(name='erutils',
          version='0.2',

          description='''
          Erutils Package is a self made package for self uses but if you find a way to use it to make you job easier 
          feel free to use as you wish but for a simple introduction erutils have 4 fls and those are
           [cli , dll , lightning(AI) , utils]
           *cli* is used for command line interface
           *dll* for dll file problems cause i ran into a lot of em
           *lightning* some classes and functions that i think that i always use like iou kmeans and many more ... 
           *utils* some functions like downloader timeSince and some more ...
           
          ''',
          author='erfan zare chavoshi',
          url='https://github.com/erfanzar/',
          author_email='erfanzare82@yahoo.com',
          license='MIT',
          packages=['erutils'],
          requires=['numpy', 'torch', 'torchvision', 'numba', 'nltk', 'pandas', 'json5', 'PyYAML', 'torchtext',
                    'torchaudio'],
          zip_safe=False)
