

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Erutils",
    version='1.3.1',
    author="Erfan Zare Chavoshi",
    author_email="erfanzare82@yahoo.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erfanzar/",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'torchvision>=1.13.0',
        'numba',
        'nltk',
        'pandas',
        'json5',
        'PyYAML',
        'torchtext>=0.9.0',
        'torchaudio>=0.9.0',
        'onnxruntime',
        'thop',
        'matplotlib'
    ],
    python_requires=">=3.6, <3.11",
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
