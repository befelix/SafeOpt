from distutils.core import setup

setup(
    name='safeopt',
    version='0.1',
    author='Felix Berkenkamp',
    author_email='befelix (at) inf.ethz.ch',
    packages=['safeopt'],
    url='https://github.com/befelix/SafeOpt',
    license='LICENSE.txt',
    description='Safe Bayesian optimization',
    long_description=open('README.md').read(),
    install_requires=[
        'GPy >= 0.6.0',
        'numpy >= 1.7',
        'scipy >= 0.12',
	'matplotlib >= 1.3',
    ],
)
