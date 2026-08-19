"""Importing this package registers every layer.

Order matters only for the order categories appear in the Add-Node menu.
"""

from . import basic        # noqa: F401  linear, embedding, dropout, norms
from . import activations  # noqa: F401
from . import conv         # noqa: F401  convolutions and pooling
from . import recurrent    # noqa: F401  LSTM / GRU / RNN
from . import attention    # noqa: F401  self and cross attention, positions
from . import shaping      # noqa: F401  flatten, reshape, concat, slice, reduce
from . import ops          # noqa: F401  add, multiply, matmul, einsum, cast
from . import blocks       # noqa: F401  MLP / conv / residual / transformer blocks
