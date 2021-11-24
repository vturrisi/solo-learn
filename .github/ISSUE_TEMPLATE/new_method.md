<!-- Adapted from https://github.com/facebookresearch/vissl/blob/main/.github/ISSUE_TEMPLATE/new-ssl-approach.md -->

---
name: "\U0001F31F New SSL method addition"
about: Submit a proposal of implementation/request to implement a new SSL method in solo-learn.

---

# 🌟 New SSL method addition

## Approach description

Important information only describing the approach, link to the arXiv paper.

## Status of the current implement

Information regarding stuff that you have already implemented for the method.
As a general checklist, you should have:
* [ ] the method implemented as new file in `solo/methods` and added to `solo/methods/__init__.py`
* [ ] the loss implemented as new file in `solo/losses` and added to `solo/losses/__init__.py`
* [ ] bash files to run experiments
* [ ] tests for the method and the loss(es) in  `tests/methods` and `tests/losses`.
* [ ] add documentation to `docs/source/solo/methods` and `docs/source/solo/losses`, and modify `docs/source/index.rst` to reference the new files (use alphabetical order).
* [ ] check if all tests pass by running `python -m pytest --cov=solo tests`.
* [ ] reproduce competitive results on at least CIFAR10/CIFAR100 (possibly Imagenet-100 and Imagenet).

