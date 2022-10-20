from setuptools import setup

setup(
  name='wandb_plot',
  version='0.1.0',
  description='Utility to help create matplotlib plots directly from wandb',
  url='https://github.com/danielpalen/wandb_plot',
  author='Daniel Palenicek',
  license='BSD 2-clause',
  packages=['wandb_plot'],
  install_requires=['wandb', 'matplotlib', 'pandas'],
  classifiers=[
    'Programming Language :: Python :: 3',
  ],
)