from setuptools import setup, find_packages

setup(
    name='lightning-network-gym',
    version='1.0.0',
    description='A Gym Environment for Lightning Network Channel and Capacity Selection Research',
    author='[Original Authors]',
    author_email='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: Free for non-commercial use',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    keywords='lightning-network, reinforcement-learning, gym, bitcoin, channel-selection, resource-allocation',
    packages=find_packages(where='.'),
    python_requires='>=3.8, <4',
    install_requires=[
        'cloudpickle>=3.0.0',
        'gym>=0.26.2',
        'matplotlib>=3.8.2',
        'networkx>=3.2.1',
        'numpy>=1.26.3',
        'pandas>=2.2.0',
        'plotly>=5.18.0',
        'python-louvain>=0.16',
        'scikit-learn>=1.4.0',
        'scipy>=1.12.0',
        'seaborn>=0.13.2',
        'tqdm>=4.66.1'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/YOUR_USERNAME/lightning-network-gym/issues',
        'Source': 'https://github.com/YOUR_USERNAME/lightning-network-gym',
        'Original Research': 'https://github.com/AAhmadS/Lightning-Network-Centralization'
    }
)
