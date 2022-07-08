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

from .swin import swin_tiny as default_swin_tiny
from .swin import swin_small as default_swin_small
from .swin import swin_base as default_swin_base
from .swin import swin_large as default_swin_large


def swin_tiny(method, *args, **kwargs):
    return default_swin_tiny(*args, **kwargs)


def swin_small(method, *args, **kwargs):
    return default_swin_small(*args, **kwargs)


def swin_base(method, *args, **kwargs):
    return default_swin_base(*args, **kwargs)


def swin_large(method, *args, **kwargs):
    return default_swin_large(*args, **kwargs)


__all__ = ["swin_tiny", "swin_small", "swin_base", "swin_large"]
