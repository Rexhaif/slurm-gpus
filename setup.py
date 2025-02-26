from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="slurm-gpus",
    version="0.1.0",
    description="A tool to list availability of GPUs on your SLURM cluster",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Original Author",
    author_email="author@example.com",
    url="https://github.com/original-author/slurm-gpus",
    packages=find_packages(),
    install_requires=[
        "rich>=10.0.0",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "slurm-gpus=slurm_gpus.core:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: System :: Clustering",
        "Topic :: System :: Monitoring",
    ],
    keywords="slurm, gpu, cluster, hpc",
)