from setuptools import setup
from os import path


current_dir = path.abspath(path.dirname(__file__))

with open(path.join(current_dir, 'README.md'), 'r') as f:
    long_description = f.read()

with open(path.join(current_dir, 'requirements.txt'), 'r') as f:
    install_requires = f.read().split('\n')

setup(
    name='safeopt',
    version='0.1',
    author='Felix Berkenkamp',
    author_email='befelix (at) inf.ethz.ch',
    packages=['safeopt'],
    url='https://github.com/befelix/SafeOpt',
    license='LICENSE.txt',
    description='Safe Bayesian optimization',
    long_description=long_description,
    install_requires=install_requires,
    keywords='Bayesian optimization, Safety',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Researchers',
        'License :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5'],
)
