# Copyright 2022 solo-learn development team.

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

from .convnext import convnext_tiny as default_convnext_tiny
from .convnext import convnext_small as default_convnext_small
from .convnext import convnext_base as default_convnext_base
from .convnext import convnext_large as default_convnext_large


def convnext_tiny(method, *args, **kwargs):
    return default_convnext_tiny(*args, **kwargs)


def convnext_small(method, *args, **kwargs):
    return default_convnext_small(*args, **kwargs)


def convnext_base(method, *args, **kwargs):
    return default_convnext_base(*args, **kwargs)


def convnext_large(method, *args, **kwargs):
    return default_convnext_large(*args, **kwargs)


__all__ = ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]
