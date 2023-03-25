from setuptools import setup

if __name__ == "__main__":
    setup(name='erutils',
          author='erfan zare chavoshi',
          url='https://github.com/erfanzar/',
          author_email='erfanzare82@yahoo.com',
          # license='MIT',
          packages=
          [
              'erutils'
          ],
          requires=
          [
              'numpy',
              'torchvision',
              'numba',
              'nltk',
              'pandas',
              'json5',
              'PyYAML',
              'torchtext',
              'torchaudio'
          ],
          zip_safe=False)
