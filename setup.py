from setuptools import setup, find_packages

setup(
    name="csrgraph",
    version="0.0.1",
    license='MIT',
    description='Tiny Library for Large Graphs',
    author='Matt Ranger',
    url='https://github.com/VHRanger/CSRGraph/',
    packages=find_packages(),
    # download_url='https://github.com/VHRanger/CSRGraph/archive/0.0.1.tar.gz',
    keywords=['graph', 'network'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.md', '*.txt', '*.rst']
    },
    install_requires=[
    'networkx',
    'numba',
    'numpy',
    'pandas',
    'scipy',
    ],
)
