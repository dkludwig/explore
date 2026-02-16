import dataclasses
from typing import Callable

import numpy as np
from numpy.random import rand
import math

from .dataset import Scan, Dataset, Axes

@dataclasses.dataclass
class SimIndAxis:
    name: str
    pts: int = 200
    start: float = 0
    stop: float = 1

    def get_array(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, self.pts)


def sin(*args: np.ndarray):
    return np.sin(sum(args)) + rand(*args[0].shape) * 0.1


def cos(*args: np.ndarray):
    return np.cos(math.prod(args) + rand() * 2 * np.pi) + rand(*args[0].shape) * 0.2


def cool(*args):
    match len(args):
        case 0:
            raise ValueError
        case 1:
            return np.sin(args[0] + rand() * 2 * np.pi)
        case 2:
            return np.cos(rand() * args[0] * args[1])
        case 3:
            return np.sin(args[0] + rand() * 2 * np.pi) * np.cos(rand() * args[1] * args[2])
        case _:
            return np.sin(args[0] + rand() * 2 * np.pi) * np.cos(rand() * args[1] * args[2]) * math.prod(args[3:])


def sim_scan(
    num_exp: int = 20,
    num_reps: int = 20,
    data_func: Callable[..., np.floating] = cool,
    ind_axes: list[SimIndAxis] | None = None,
) -> Scan:
    """Simulate a scan and save it to disk.

    data_func is called symmetrically on the independent axis values,
    i.e. data_func(*meshgrid_values) for each grid point.
    """
    if ind_axes is None:
        ind_axes = [
            SimIndAxis(name="frequency", pts=100, start=0, stop=10),
            SimIndAxis(name="amplitude", pts=50, start=-1, stop=1),
        ]

    # Build axes arrays and meshgrid
    axes_arrays = [ax.get_array() for ax in ind_axes]
    grids = np.meshgrid(*axes_arrays, indexing="ij")

    # Generate experiment labels: A, B, C, ...
    exp_labels = [chr(ord("A") + i) for i in range(num_exp)]

    datasets: list[Dataset] = []
    for exp in exp_labels:
        for rep in range(num_reps):
            data = data_func(*grids)
            axes = [Axes(name=ax.name, values=arr) for ax, arr in zip(ind_axes, axes_arrays)]
            datasets.append(Dataset(experiment=exp, repeat=rep, data=data, axes=axes))

    scan = Scan(datasets=datasets)
    scan.save()
    return scan


def sim_scan_3d(num_exp=10, num_reps=10, data_func=cool) -> Scan:
    """Simulate a 3D scan: 200x200x3 (x, y, frequency)."""
    return sim_scan(
        num_exp=num_exp,
        num_reps=num_reps,
        data_func=data_func,
        ind_axes=[
            SimIndAxis(name="time", pts=150, start=0, stop=10),
            SimIndAxis(name="frequency", pts=3, start=1, stop=5),
            SimIndAxis(name="position", pts=150, start=1, stop=5),
        ],
    )


def sim_scan_4d(num_exp=5, num_reps=5, data_func=cool) -> Scan:
    """Simulate a 4D scan: 200x200x3x2 (x, y, frequency, power)."""
    return sim_scan(
        num_exp=num_exp,
        num_reps=num_reps,
        data_func=data_func,
        ind_axes=[
            SimIndAxis(name="time", pts=150, start=0, stop=10),
            SimIndAxis(name="frequency", pts=3, start=1, stop=5),
            SimIndAxis(name="position", pts=150, start=1, stop=5),
            SimIndAxis(name="power", pts=2, start=-1, stop=1),
        ],
    )
