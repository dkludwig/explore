"""Minimal standalone demo: hv.Dataset.to(hv.Image, groupby) with auto-generated slider."""

import numpy as np
import holoviews as hv
import panel as pn

from explore.sim import cool

hv.extension("bokeh")
pn.extension()

def main():
    x = np.linspace(0, 10, 200)
    y = np.linspace(1, 5, 150)
    time = np.linspace(-10, 10, 20)
    freq = np.array([1.0, 3.0, 5.0])

    X, Y, T, F = np.meshgrid(x, y, time, freq, indexing="ij")
    data = cool(X, F, Y, T)

    # hv.Dataset expects value array in reversed kdims order:
    #   kdims = ['x', 'y', 'time', 'frequency']  →  value shape = (len(freq), len(time), len(y), len(x)) = (3, 20, 200, 200)
    value_array = data.transpose(3, 2, 1, 0)

    ds = hv.Dataset(
        (x, y, time, freq, value_array),
        kdims=["hey", "there", "time", "frequency"],
        vdims=["value"],
    )

    # 4D gets plotted as a heatmap with an extra slider to the right
    # to switch through slices
    plot = ds.to(hv.Image, kdims=["hey", "there"], groupby=["time", "frequency"], dynamic=True)
    plot = plot.opts(hv.opts.Image(colorbar=True, width=600, height=500, cmap="viridis"))

    # this line moves the slider to the bottom
    plot = pn.pane.HoloViews(plot, widget_location="bottom")

    # Use .layout to keep widgets when wrapping in a Column
    # (see https://github.com/holoviz/panel/issues/5628)
    col = pn.Column("## Slider Demo", plot.layout)

    pn.serve(col, port=5008, show=True)


if __name__ == "__main__":
    main()



