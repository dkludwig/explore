"""Transform library: Dataset → Dataset operations along a single axis."""

import numpy as np

from .dataset import Axes, Dataset, Scan


def mean_transform(dataset: Dataset, over_axis: str) -> Dataset:
    """Take the mean along `over_axis`, reducing dimensionality by 1."""
    axis_idx = next(i for i, a in enumerate(dataset.axes) if a.name == over_axis)
    new_data = np.mean(dataset.data, axis=axis_idx)
    new_axes = [a for a in dataset.axes if a.name != over_axis]
    return Dataset(
        experiment=dataset.experiment,
        repeat=dataset.repeat,
        data=new_data,
        axes=new_axes,
    )


def fft_transform(dataset: Dataset, over_axis: str) -> Dataset:
    """Real FFT along `over_axis`. Axis values become frequencies, name becomes '1/{name}'."""
    axis_idx = next(i for i, a in enumerate(dataset.axes) if a.name == over_axis)
    ax = dataset.axes[axis_idx]

    new_data = np.abs(np.fft.rfft(dataset.data, axis=axis_idx))

    spacing = ax.values[1] - ax.values[0]
    freqs = np.fft.rfftfreq(len(ax.values), d=spacing)
    new_ax = Axes(name=f"1/{ax.name}", values=freqs)

    new_axes = list(dataset.axes)
    new_axes[axis_idx] = new_ax
    return Dataset(
        experiment=dataset.experiment,
        repeat=dataset.repeat,
        data=new_data,
        axes=new_axes,
    )


REGISTRY: dict[str, callable] = {
    "mean": mean_transform,
    "fft": fft_transform,
}


def apply_transform(dataset: Dataset, operation: str, over_axis: str) -> Dataset:
    """Apply a named transform to a dataset."""
    return REGISTRY[operation](dataset, over_axis)


def apply_to_scan(scan: Scan, operation: str, over_axis: str) -> Scan:
    """Apply a transform to every dataset in a scan, returning a new Scan."""
    new_datasets = [apply_transform(ds, operation, over_axis) for ds in scan.datasets]
    return Scan(datasets=new_datasets)
