import os
from dataclasses import dataclass

import numpy as np
from rpgpy.utils import rpg_seconds2datetime64
from scipy.interpolate import interp1d
from torch import Tensor

IntTuples = tuple[tuple[int, int], ...]
Ints = tuple[int, ...]


@dataclass
class VoodooOptions:
    trained_model: str = f"{os.path.dirname(__file__)}/trained_models/Vnet2_0-dy0.00-fnXX-cuda0.pt"
    kernel_sizes: IntTuples = ((3, 3), (3, 3), (1, 3), (1, 3), (1, 3))
    pad_sizes: IntTuples = ((1, 1), (1, 1), (0, 1), (0, 1), (0, 1))
    stride_sizes: IntTuples = ((1, 2), (1, 2), (1, 2), (2, 2), (1, 2))
    num_filters: Ints = (16, 32, 64, 128, 256)
    dense_layers: Ints = (128, 128, 64)
    output_shape: int = 2
    n_channels: int = 6
    z_limits: tuple[int, int] = (-50, 20)
    device: str = "cpu"


def time_grid(date: str, resolution: int = 30) -> np.ndarray:
    date_components = date.split("-")
    n_time = int(24 * 60 * 60 / resolution)
    time = np.linspace(0, 24, n_time)
    return decimal_hour2unix(date_components, time)


def rpg_time2unix(time: np.ndarray) -> np.ndarray:
    radar_times = rpg_seconds2datetime64(time)
    return radar_times.astype("datetime64[s]").astype("int")


def decimal_hour2unix(date: list, time: np.ndarray | list) -> np.ndarray:
    unix_timestamp = np.datetime64("-".join(date)).astype("datetime64[s]").astype("int")
    return (time * 60 * 60 + unix_timestamp).astype(int)


def lin2z(array: np.ndarray | list | float) -> np.ndarray:
    return 10 * np.ma.log10(array)


def arg_nearest(array: np.ndarray, value: float | int) -> np.int64:
    i = np.searchsorted(array, value)
    return i if i <= array.shape[0] - 1 else i - 1


def interpolate_to_256(rpg_data: np.ndarray, rpg_header: dict) -> np.ndarray:
    n_bins = 256
    n_time, n_range, _ = rpg_data.shape
    spec_new = np.zeros((n_time, n_range, n_bins))
    chirp_limits = np.append(rpg_header["RngOffs"], n_range)

    for ind, (ia, ib) in enumerate(zip(chirp_limits[:-1], chirp_limits[1:])):
        spec = rpg_data[:, ia:ib, :]
        if rpg_header["SpecN"][ind] == n_bins:
            spec_new[:, ia:ib, :] = spec
        else:
            old = rpg_header["velocity_vectors"][ind]
            f = interp1d(old, spec, axis=2, bounds_error=False, fill_value=-999.0, kind="linear")
            spec_new[:, ia:ib, :] = f(np.linspace(old[np.argmin(old)], old[np.argmax(old)], n_bins))

    return spec_new


def reshape(data: Tensor, mask: np.ndarray) -> np.ndarray:
    new_shape = mask.shape + (data.shape[1],)
    input_reshaped = np.zeros(new_shape)
    counter = 0
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            if mask[i, j]:
                continue
            input_reshaped[i, j, :] = data[counter, :]
            counter += 1
    return input_reshaped
