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

from .vit import vit_tiny as default_vit_tiny
from .vit import vit_small as default_vit_small
from .vit import vit_base as default_vit_base
from .vit import vit_large as default_vit_large

from .vit_mocov3 import vit_tiny as mocov3_vit_tiny
from .vit_mocov3 import vit_small as mocov3_vit_small
from .vit_mocov3 import vit_base as mocov3_vit_base
from .vit_mocov3 import vit_large as mocov3_vit_large


def vit_tiny(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_tiny}
    return custom_backbone_constructor.get(str(method).lower(), default_vit_tiny)(*args, **kwargs)


def vit_small(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_small}
    return custom_backbone_constructor.get(str(method).lower(), default_vit_small)(*args, **kwargs)


def vit_base(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_base}
    return custom_backbone_constructor.get(str(method).lower(), default_vit_base)(*args, **kwargs)


def vit_large(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_large}
    return custom_backbone_constructor.get(str(method).lower(), default_vit_large)(*args, **kwargs)


__all__ = ["vit_tiny", "vit_small", "vit_base", "vit_large"]
