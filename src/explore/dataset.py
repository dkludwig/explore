import dataclasses
import datetime
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image

from .constants import DATA_DIR, ASPECT_RATIO


def make_serial() -> str:
    """Generate a timestamp-based serial number."""
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(" ", "-").replace(":", "-")
    return timestamp[:-7]


def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """Convert cell centers to cell edges (N centers → N+1 edges)."""
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return edges


def _render_2d(ax: plt.Axes, data: np.ndarray, x_ax: "Axes", y_ax: "Axes") -> None:
    x_edges = _centers_to_edges(x_ax.values)
    y_edges = _centers_to_edges(y_ax.values)
    im = ax.pcolorfast(x_edges, y_edges, data)
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.locator = mticker.MaxNLocator(5)
    cbar.update_ticks()
    ax.set_xlabel(x_ax.name, fontsize=16)
    ax.set_ylabel(y_ax.name, fontsize=16)


def _render_1d(ax: plt.Axes, dataset: "Dataset") -> None:
    ax.plot(dataset.axes[0].values, dataset.data)
    ax.set_xlabel(dataset.axes[0].name, fontsize=16)
    ax.set_ylabel("data", fontsize=16)


def _encode_png(rgba: np.ndarray) -> bytes:
    """Encode RGBA array to PNG bytes (low compression for speed)."""
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


@dataclasses.dataclass
class Axes:
    name: str
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                f"Axes values must be 1D, got shape {self.values.shape}"
            )


@dataclasses.dataclass
class Dataset:
    experiment: str
    repeat: int
    data: np.ndarray
    axes: list[Axes]
    _plot_selections: dict[str, str] | None = dataclasses.field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        expected_shape = tuple(len(ax.values) for ax in self.axes)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"Data shape {self.data.shape} does not match "
                f"axes product {expected_shape}"
            )

    @property
    def plot_selections(self) -> dict[str, str]:
        """{"x": axis_name, "y": axis_name} — defaults to two longest axes."""
        if self._plot_selections is not None:
            return self._plot_selections
        sorted_axes = sorted(
            self.axes, key=lambda a: len(a.values), reverse=True
        )
        sel = {"x": sorted_axes[0].name}
        if len(sorted_axes) > 1:
            sel["y"] = sorted_axes[1].name
        return sel

    @plot_selections.setter
    def plot_selections(self, sel: dict[str, str]) -> None:
        self._plot_selections = sel

    @property
    def plot_order(self) -> list[str]:
        """[x_name, y_name, slider0_name, ...] derived from plot_selections."""
        sel = self.plot_selections
        chosen = [sel["x"]]
        if "y" in sel:
            chosen.append(sel["y"])
        remaining = [a.name for a in self.axes if a.name not in chosen]
        return chosen + remaining

    def slice_2d(self) -> tuple[np.ndarray, "Axes", "Axes"]:
        """Slice nD data to 2D using plot_selections.

        Returns (data_2d, x_axis, y_axis) where data_2d has shape (len(y), len(x)).
        """
        sel = self.plot_selections
        x_name, y_name = sel["x"], sel["y"]

        # Build index: full slice for plot axes, middle for sliders
        idx = []
        for ax in self.axes:
            if ax.name in (x_name, y_name):
                idx.append(slice(None))
            else:
                idx.append(len(ax.values) // 2)

        sliced = self.data[tuple(idx)]

        x_ax = next(a for a in self.axes if a.name == x_name)
        y_ax = next(a for a in self.axes if a.name == y_name)

        # Ensure (y, x) order for pcolorfast
        x_dim = next(i for i, a in enumerate(self.axes) if a.name == x_name)
        y_dim = next(i for i, a in enumerate(self.axes) if a.name == y_name)
        if x_dim < y_dim:
            sliced = sliced.T

        return sliced, x_ax, y_ax

    def __repr__(self) -> str:
        axes_repr = ", ".join(
            f"{ax.name}={ax.values.shape}" for ax in self.axes
        )
        return (
            f"Dataset(exp='{self.experiment}', rep={self.repeat}, \n"
            f"        data.shape={self.data.shape}, \n"
            f"        axes=[{axes_repr}])"
        )


@dataclasses.dataclass
class Scan:
    """A collection of datasets from multiple experiments and repeats."""

    datasets: list[Dataset]
    serial: str = dataclasses.field(default_factory=make_serial)

    @property
    def dir(self) -> Path:
        return DATA_DIR / self.serial

    def __repr__(self) -> str:
        experiments = sorted(set(ds.experiment for ds in self.datasets))
        repeats_per_exp = {
            exp: len([ds for ds in self.datasets if ds.experiment == exp])
            for exp in experiments
        }
        exp_summary = ", ".join(
            f"{exp}(n={repeats_per_exp[exp]})" for exp in experiments
        )
        return (
            f"Scan(serial='{self.serial}', \n"
            f"     experiments=[{exp_summary}], \n"
            f"     total_datasets={len(self.datasets)})"
        )

    def save(self) -> None:
        """Save all datasets to DATA_DIR/{serial}/data.npz."""
        self.dir.mkdir(parents=True, exist_ok=True)
        filepath = self.dir / "data.npz"

        save_dict = {}

        # Save dataset metadata
        exp_letters = [ds.experiment for ds in self.datasets]
        repeat_nums = [ds.repeat for ds in self.datasets]
        save_dict["exp_letters"] = np.array(exp_letters, dtype=object)
        save_dict["repeat_nums"] = np.array(repeat_nums, dtype=np.int32)

        # Save each dataset with a prefix
        for dataset in self.datasets:
            prefix = f"{dataset.experiment}_{dataset.repeat}"
            save_dict[f"{prefix}_data"] = dataset.data
            save_dict[f"{prefix}_num_axes"] = np.array(len(dataset.axes))

            # Save each axis
            for i, axis in enumerate(dataset.axes):
                save_dict[f"{prefix}_axis_{i}_name"] = np.array(
                    axis.name, dtype=object
                )
                save_dict[f"{prefix}_axis_{i}_values"] = axis.values

        np.savez_compressed(filepath, **save_dict)

    @classmethod
    def load(cls, serial: str) -> "Scan":
        """Load scan from DATA_DIR/{serial}/data.npz."""
        filepath = DATA_DIR / serial / "data.npz"

        if not filepath.exists():
            raise FileNotFoundError(f"Scan not found: {filepath}")

        npz = np.load(filepath, allow_pickle=True)

        exp_letters = npz["exp_letters"]
        repeat_nums = npz["repeat_nums"]

        datasets = []
        for exp, rep in zip(exp_letters, repeat_nums):
            exp = str(exp)
            rep = int(rep)
            prefix = f"{exp}_{rep}"

            data = npz[f"{prefix}_data"]
            num_axes = int(npz[f"{prefix}_num_axes"])

            axes = []
            for i in range(num_axes):
                name = str(npz[f"{prefix}_axis_{i}_name"])
                values = npz[f"{prefix}_axis_{i}_values"]
                axes.append(Axes(name=name, values=values))

            datasets.append(
                Dataset(experiment=exp, repeat=rep, data=data, axes=axes)
            )

        return cls(datasets=datasets, serial=serial)

    def render(
        self,
        figsize: tuple[float, float] = (4 * ASPECT_RATIO, 4),
        dpi: int = 100,
        encode_workers: int = 4,
    ) -> dict[tuple[str, int], bytes]:
        """Render all datasets to PNG thumbnails in {dir}/thumbnails/.

        Creates a fresh figure per dataset (avoids gridspec/colorbar corruption)
        and encodes PNGs in parallel.
        Returns a dict mapping (experiment, repeat) → PNG bytes.
        """
        thumb_dir = self.dir / "thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()

        # Phase 1: render all frames to RGBA buffers
        keys: list[tuple[str, int]] = []
        frames: list[np.ndarray] = []

        for dataset in self.datasets:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.subplots_adjust(left=0.15, right=0.85, top=0.88, bottom=0.15)

            if dataset.data.ndim >= 2:
                data_2d, x_ax, y_ax = dataset.slice_2d()
                _render_2d(ax, data_2d, x_ax, y_ax)
            elif dataset.data.ndim == 1:
                _render_1d(ax, dataset)
            else:
                raise ValueError("Must have ndim >= 1")

            ax.set_title(
                f"{dataset.experiment}{dataset.repeat}", fontsize=18, pad=4
            )
            ax.tick_params(labelsize=14)

            fig.canvas.draw()
            frames.append(np.array(fig.canvas.buffer_rgba()))
            keys.append((dataset.experiment, dataset.repeat))
            plt.close(fig)

        # Phase 2: encode PNGs in parallel
        with ThreadPoolExecutor(max_workers=encode_workers) as pool:
            png_list = list(pool.map(_encode_png, frames))

        thumbnails = dict(zip(keys, png_list))

        # Phase 3: write to disk
        for (exp, rep), png_bytes in thumbnails.items():
            (thumb_dir / f"{exp}_{rep}.png").write_bytes(png_bytes)

        elapsed = time.perf_counter() - t0
        print(
            f"Rendered {len(thumbnails)} thumbnails in {elapsed:.2f}s "
            f"({elapsed / len(thumbnails) * 1000:.1f}ms each)"
        )

        return thumbnails
