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

from .poolformer import poolformer_s12 as default_poolformer_s12
from .poolformer import poolformer_s24 as default_poolformer_s24
from .poolformer import poolformer_s36 as default_poolformer_s36
from .poolformer import poolformer_m36 as default_poolformer_m36
from .poolformer import poolformer_m48 as default_poolformer_m48


def poolformer_s12(method, *args, **kwargs):
    return default_poolformer_s12(*args, **kwargs)


def poolformer_s24(method, *args, **kwargs):
    return default_poolformer_s24(*args, **kwargs)


def poolformer_s36(method, *args, **kwargs):
    return default_poolformer_s36(*args, **kwargs)


def poolformer_m36(method, *args, **kwargs):
    return default_poolformer_m36(*args, **kwargs)


def poolformer_m48(method, *args, **kwargs):
    return default_poolformer_m48(*args, **kwargs)


__all__ = ["poolformer_s12", "poolformer_s24", "poolformer_s36", "poolformer_m36", "poolformer_m48"]
