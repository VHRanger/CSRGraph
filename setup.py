from setuptools import setup, find_packages

setup(
    name="csrgraph",
    version="0.1.28",
    license='MIT',
    description='Fast python graphs',
    author='Matt Ranger',
    url='https://github.com/VHRanger/CSRGraph',
    packages=find_packages(),
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
        'scikit-learn',
        'tqdm',
    ],
)
