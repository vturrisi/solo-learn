# Contributing to solo-learn
We want to make contributing to this project as easy and transparent as possible. Also, we encourage the community to help and improve the library.

## Our Development Process
We fix issues as we discover them, either by running the code or by checking the issues tab on github. Major changes (e.g., new methods, other downstream tasks, additional features etc), are added periodically and based on how we judge its priority.

## Issues
We use GitHub issues to track public bugs and questions. We will make templates for each type of issue in [issue templates](https://github.com/vturrisi/solo-learn/issues/new/choose), but for now, check if your issue has a fitting template or please be as verbose and clear as possible. Also, provide ways to reproduce the issue (e.g., bash script).


## Pull Requests
We actively welcome pull requests.

Before you start implementing a new method, make sure that the method has an accompaning paper (arxiv versions are fine) with competitive experimental results.

When implementing the new method:
1. Fork the repo and create your branch from `main`.
2. Extend the available `BaseMethod` and `BaseMomentumMethod` classes, or even the methods that are already implemented. Your new method should reuse our base structures or provide enough justification for changing/incrementing the base structures if needed.
3. Provide a clear and simple implementation following our code style.
4. Provide unit tests.
5. Modify the documentation to add the new method and accompanying features. We use Sphinx for that.
6. Provide bash files for running the method.
7. Reproduce competitive results on at least CIFAR-10 and CIFAR-100. Imagenet-100/Imagenet results are also desirable, but we can run those eventually.
8. Ensure that your tests by running `python -m pytest --cov=solo tests`.

If you are fixing a bug, please open an issue beforehand. If should also modify the library as minimally as possible.

When implementing extensions for the library, detail why this is important by providing use cases.

## Coding Style

Please follow our coding style and use [black](https://github.com/psf/black) with line length of 100. We will update this section with a more hands-on guide.

## License
By contributing to solo-learn, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.