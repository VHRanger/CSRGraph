from setuptools import setup, find_packages

setup(
    name="csrgraph",
    version="0.0.1",
    license='MIT',
    description='Fast python graphs',
    author='Matt Ranger',
    url='https://github.com/VHRanger/graph2vec/',
    packages=find_packages(),
    # download_url='https://github.com/VHRanger/graph2vec/archive/0.0.1.tar.gz',
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
