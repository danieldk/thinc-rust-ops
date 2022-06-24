import numpy
from thinc.api import NumpyOps
from thinc.types import Xp, DeviceTypes

from .rust_ops import RustOps as _RustOps

class RustOps(_RustOps, NumpyOps):
    name = "rust"
    xp: Xp = numpy

    def __init__(self, device_type: DeviceTypes = "cpu", device_id: int = -1):
        self.device_type = device_type
        self.device_id = device_id