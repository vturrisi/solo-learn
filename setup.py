from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [p.strip().split()[-1] for p in f.readlines()]
4
KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]

setup(
    name="solo",
    packages=find_packages(exclude=["bash_files"]),
    version="0.0.1",
    license="MIT",
    author="Victor G. Turrisi da Costa, Enrico Fini",
    author_email="vturrisi@gmail.com, enrico.fini@gmail.com",
    url="https://github.com/vturrisi/solo-learn",
    keywords=KW,
    install_requires=requirements,
    dependency_links=["https://developer.download.nvidia.com/compute/redist"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
