from typing import TYPE_CHECKING, TypeVar, cast
import numpy
from thinc.api import NumpyOps, Ops
from thinc.types import Xp, DeviceTypes, FloatsXd

from .rust_ops import RustOps as _RustOps


if TYPE_CHECKING:
    _Ops = Ops
else:
    try:
        from thinc_apple_ops import AppleOps

        _Ops = AppleOps
    except ImportError:
        _Ops = NumpyOps


FloatsType = TypeVar("FloatsType", bound=FloatsXd)


class RustOps(_RustOps, _Ops):
    name = "rust"
    xp: Xp = numpy

    def __init__(self, device_type: DeviceTypes = "cpu", device_id: int = -1):
        self.device_type = device_type
        self.device_id = device_id
