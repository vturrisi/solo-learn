# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]

REQUIREMENTS_FILE = os.path.join(os.path.dirname(__file__), "requirements.txt")
with open(REQUIREMENTS_FILE) as fo:
    REQUIREMENTS = [str(req) for req in parse_requirements(fo.readlines())]

EXTRA_REQUIREMENTS = {
    "dali": ["nvidia-dali-cuda110"],
    "umap": ["matplotlib", "seaborn", "pandas", "umap-learn"],
    "h5": ["h5py"],
}


def parse_requirements(path):
    with open(path) as f:
        requirements = [p.strip().split()[-1] for p in f.readlines()]
    return requirements


setup(
    name="solo-learn",
    packages=find_packages(exclude=["bash_files", "docs", "downstream", "tests", "zoo"]),
    version="1.0.6",
    license="MIT",
    author="solo-learn development team",
    author_email="vturrisi@gmail.com, enrico.fini@gmail.com",
    url="https://github.com/vturrisi/solo-learn",
    keywords=KW,
    install_requires=REQUIREMENTS,
    extras_require=EXTRA_REQUIREMENTS,
    dependency_links=["https://developer.download.nvidia.com/compute/redist"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
