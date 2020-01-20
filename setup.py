#!/usr/bin/env python
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(name='slim_python3',
      version='0.1',
      description='learn optimized scoring systems from data',
      long_description = '''
      slim-python is a free software package to train SLIM scoring systems
                         ''',
      author='Berk Ustun, Hallee Wong & Aaron Elliot',
      author_email='aaronelliot27@gmail.com', #'ustunb@mit.edu',
      url = 'https://github.com/AaEll/slim-python', #url='https://www.berkustun.com/',
      packages=find_packages('slim_python3'),
      include_package_data=True,
      zip_safe=False,
      python_requires = '>=3.5',
      install_requires=['ortools',
                        'numpy',
                        'scipy',
                        'pandas',
                        'ortools',
                        'PrettyTable'
                        ''],  


      )
