from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='past',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Nadav Har-Tuv, Or Tal and Yossi Adi',
    description='PAST: A PyTorch-based Audio Compression Model',
    url='https://github.com/slp-rl/PAST',
    python_requires='>=3.10',
)
