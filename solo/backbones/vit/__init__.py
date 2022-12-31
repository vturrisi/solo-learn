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

import logging

from .vit import vit_tiny as default_vit_tiny
from .vit import vit_small as default_vit_small
from .vit import vit_base as default_vit_base
from .vit import vit_large as default_vit_large

from .vit_mocov3 import vit_tiny as mocov3_vit_tiny
from .vit_mocov3 import vit_small as mocov3_vit_small
from .vit_mocov3 import vit_base as mocov3_vit_base
from .vit_mocov3 import vit_large as mocov3_vit_large

from .vit_mae import vit_tiny as mae_vit_tiny
from .vit_mae import vit_small as mae_vit_small
from .vit_mae import vit_base as mae_vit_base
from .vit_mae import vit_large as mae_vit_large


def get_constructor(method, options, default):
    if str(method).lower() in options:
        logging.warn(f"Using custom backbone for {method}")
        return options[method]

    logging.warn(f"No custom backbone found for {method}, defaulting to default")
    return default


def vit_tiny(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_tiny, "mae": mae_vit_tiny}
    return get_constructor(method, custom_backbone_constructor, default_vit_tiny)(*args, **kwargs)


def vit_small(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_small, "mae": mae_vit_small}
    return get_constructor(method, custom_backbone_constructor, default_vit_small)(*args, **kwargs)


def vit_base(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_base, "mae": mae_vit_base}
    return get_constructor(method, custom_backbone_constructor, default_vit_base)(*args, **kwargs)


def vit_large(method, *args, **kwargs):
    custom_backbone_constructor = {"mocov3": mocov3_vit_large, "mae": mae_vit_large}
    return get_constructor(method, custom_backbone_constructor, default_vit_large)(*args, **kwargs)


__all__ = ["vit_tiny", "vit_small", "vit_base", "vit_large"]
