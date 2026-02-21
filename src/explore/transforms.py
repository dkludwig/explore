"""Transform library: Dataset → Dataset operations with schema-driven parameters."""

import dataclasses

import numpy as np

from .dataset import Axes, Dataset, Scan


@dataclasses.dataclass
class TransformSpec:
    """A transform function plus its parameter schema for dynamic UI generation.

    Each entry in ``params`` is a dict with keys:
      - "name": parameter name passed as kwarg to ``fn``
      - "type": one of "axis", "axis_value", "float"
    """

    fn: callable
    params: list[dict]


# ---------------------------------------------------------------------------
# Transform implementations
# ---------------------------------------------------------------------------

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


def slice_transform(dataset: Dataset, over_axis: str, value: float) -> Dataset:
    """Select a single slice at the given value along over_axis, reducing dimensionality by 1."""
    axis_idx = next(i for i, a in enumerate(dataset.axes) if a.name == over_axis)
    ax = dataset.axes[axis_idx]
    idx = int(np.argmin(np.abs(ax.values - value)))
    new_data = np.take(dataset.data, idx, axis=axis_idx)
    new_axes = [a for a in dataset.axes if a.name != over_axis]
    return Dataset(
        experiment=dataset.experiment,
        repeat=dataset.repeat,
        data=new_data,
        axes=new_axes,
    )


def diff_transform(dataset: Dataset, over_axis: str, value_a: float, value_b: float) -> Dataset:
    """Subtract slice at value_a from slice at value_b along over_axis."""
    axis_idx = next(i for i, a in enumerate(dataset.axes) if a.name == over_axis)
    ax = dataset.axes[axis_idx]
    idx_a = int(np.argmin(np.abs(ax.values - value_a)))
    idx_b = int(np.argmin(np.abs(ax.values - value_b)))
    slice_a = np.take(dataset.data, idx_a, axis=axis_idx)
    slice_b = np.take(dataset.data, idx_b, axis=axis_idx)
    new_data = slice_b - slice_a
    new_axes = [a for a in dataset.axes if a.name != over_axis]
    return Dataset(
        experiment=dataset.experiment,
        repeat=dataset.repeat,
        data=new_data,
        axes=new_axes,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, TransformSpec] = {
    "mean": TransformSpec(
        fn=mean_transform,
        params=[{"name": "over_axis", "type": "axis"}],
    ),
    "fft": TransformSpec(
        fn=fft_transform,
        params=[{"name": "over_axis", "type": "axis"}],
    ),
    "slice": TransformSpec(
        fn=slice_transform,
        params=[
            {"name": "over_axis", "type": "axis"},
            {"name": "value", "type": "axis_value"},
        ],
    ),
    "diff": TransformSpec(
        fn=diff_transform,
        params=[
            {"name": "over_axis", "type": "axis"},
            {"name": "value_a", "type": "axis_value"},
            {"name": "value_b", "type": "axis_value"},
        ],
    ),
}


# ---------------------------------------------------------------------------
# Application helpers
# ---------------------------------------------------------------------------

def apply_transform(dataset: Dataset, operation: str, **kwargs) -> Dataset:
    """Apply a named transform to a dataset."""
    return REGISTRY[operation].fn(dataset, **kwargs)


def apply_to_scan(scan: Scan, operation: str, **kwargs) -> Scan:
    """Apply a transform to every dataset in a scan, returning a new Scan."""
    new_datasets = [apply_transform(ds, operation, **kwargs) for ds in scan.datasets]
    return Scan(datasets=new_datasets)
