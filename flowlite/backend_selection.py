"""Logic for backend selection"""
import os


BACKEND = os.environ.get("FLOWLITE_BACKEND", "nd")


if BACKEND == "nd":
    print("Using flowlite backend")
    from . import backend_ndarray as array_api
    from .backend_ndarray import (
        all_devices,
        cuda,
        cpu,
        cpu_numpy,
        default_device,
        BackendDevice as Device,
    )

    NDArray = array_api.NDArray
elif BACKEND == "np":
    print("Using numpy backend")
    import numpy as array_api
    from .backend_numpy import all_devices, cpu, default_device, Device

    NDArray = array_api.ndarray
else:
    raise RuntimeError(f"Unknown needle array backend {BACKEND}")
