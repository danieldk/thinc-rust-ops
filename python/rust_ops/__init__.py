from typing import TypeVar, cast
import numpy
from thinc.api import NumpyOps
from thinc.types import Xp, DeviceTypes, FloatsXd

from .rust_ops import RustOps as _RustOps


FloatsType = TypeVar("FloatsType", bound=FloatsXd)


class RustOps(_RustOps, NumpyOps):
    name = "rust"
    xp: Xp = numpy

    def __init__(self, device_type: DeviceTypes = "cpu", device_id: int = -1):
        self.device_type = device_type
        self.device_id = device_id


    # Remove once https://github.com/explosion/thinc/pull/709 is merged and we
    # have a Thinc version with this change. Patching sigmoid here to make sure
    # that we can pass tests.
    def sigmoid(self, X: FloatsType, *, inplace: bool = False) -> FloatsType:
        if inplace:
            # To prevent overflows and help with regularization/numerical stability
            X = self.xp.clip(X, -20.0, 20.0, out=X)
            self.xp.exp(-X, out=X)
            X += 1.0  # type: ignore[assignment]
            X **= -1.0  # type: ignore[assignment]
            return cast(FloatsType, X)
        else:
            X = self.xp.clip(X, -20.0, 20.0)
            return cast(FloatsType, 1.0 / (1.0 + self.xp.exp(-X)))