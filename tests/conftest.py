"""Test configuration — mock MLX modules when running on non-Apple-Silicon platforms."""

import sys
from unittest.mock import MagicMock

# If mlx is not installed (e.g. Linux CI), inject mock modules so that
# ``import server`` succeeds.  The unit tests mock ModelManager anyway,
# so the actual MLX code paths are never exercised.
try:
    import mlx  # noqa: F401
except ImportError:
    for mod in ("mlx", "mlx.core", "mlx.nn", "mlx_lm"):
        sys.modules.setdefault(mod, MagicMock())
