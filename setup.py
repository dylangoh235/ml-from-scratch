from setuptools import setup, find_packages

__version__ = '0.0.1'

setup(
    name='dl-from-scratch', 
    version='1.0', 
    packages=find_packages(),
    setup_requires=['numpy>=1.10'],
)