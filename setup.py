from setuptools import setup

if __name__ == "__main__":
    setup(name='erutils',
          version='0.1',
          description='Erfan ',
          author='erfan zare chavoshi',
          url='https://github.com/erfanzar/',
          author_email='erfanzare82@yahoo.com',
          license='MIT',
          packages=['erutils'],
          requires=['numpy', 'torch', 'torchvision', 'numba', 'nltk', 'pandas', 'json5', 'PyYAML', 'torchtext',
                    'torchaudio'],
          zip_safe=False)
