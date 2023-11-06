from setuptools import setup, find_packages

setup(
    name='pytorch_scripts',
    version='0.1',
    packages=find_packages(),
    install_requires=[
      'numpy >= 1.25.2',
      'humanize >= 4.8.0',
      'torch >= 2.0.1',
      'tqdm >= 4.66.1',
      'matplotlib >= 3.7.2'
    ],
    # Optional metadata about your package:
    description='Training scripts for pytorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url='https://github.com/vaharoni/pytorch_scripts',
)
