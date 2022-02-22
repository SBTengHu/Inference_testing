from setuptools import find_packages, setup
import numpy as np

# Setup installation
setup(name='Inference_testing',
      version_format='{tag}.dev{commitcount}+{gitsha}',
      #setup_requires=['setuptools-git-version'],
      description='a code to run inference test',
      author='Joe Hennawi',
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
