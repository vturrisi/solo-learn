from setuptools import find_packages, setup

KW = ["artificial intelligence", "deep learning", "unsupervised learning", "contrastive learning"]

EXTRA_DEPENDENCIES = ["dali", "umap"]


def parse_requirements(path):
    with open(path) as f:
        requirements = [p.strip().split()[-1] for p in f.readlines()]
    return requirements


extra_requirements = {}
for lib in parse_requirements("extra_requirements.txt"):
    name = [name for name in EXTRA_DEPENDENCIES if name in lib][0]
    extra_requirements[name] = [lib]


setup(
    name="solo",
    packages=find_packages(exclude=["bash_files"]),
    version="0.0.6",
    license="MIT",
    author="Victor G. Turrisi da Costa, Enrico Fini",
    author_email="vturrisi@gmail.com, enrico.fini@gmail.com",
    url="https://github.com/vturrisi/solo-learn",
    keywords=KW,
    install_requires=parse_requirements("requirements.txt"),
    extras_require=extra_requirements,
    dependency_links=["https://developer.download.nvidia.com/compute/redist"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    zip_safe=False,
)
