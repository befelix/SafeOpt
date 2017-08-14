from distutils.core import setup

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
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
)
