from setuptools import find_packages, setup
import numpy as np

# Setup installation
setup(name='Inference_testing',
      version='1.0',
      #setup_requires=['setuptools-git-version'],
      description='a code to run inference test',
      author='Joseph Hennawi',
      author_email='',
      url='https://github.com/SBTengHu/Inference_testing',
      license='',
      packages=find_packages(include=['*.py']),
      include_dirs=np.get_include(),
      install_requires=[
                        'corner',
                        'h5py',
                        'sklearn',
                        'scipy',
                        'numpy',
                        'matplotlib'],
      )
