"""Panel server for batch thumbnail viewing and interactive exploration.

Launch with:
    python -m explore.server

Each browser tab is independent — open http://localhost:5006/explore?serial=XXXX
to view a specific scan.

Features:
- Thumbnail grid in main area
- Click thumbnail → interactive Bokeh plot in sidebar
- Arrow keys to navigate between datasets
- Click on plot to pick points → export to JSON
"""

import json
import subprocess
import sys
import threading

import holoviews as hv
import numpy as np
import panel as pn
import param

from .dataset import Scan
from .transforms import REGISTRY as TRANSFORM_REGISTRY, apply_transform, apply_to_scan
from .launcher import notify
from .constants import ASPECT_RATIO, DATA_DIR


hv.extension("bokeh")
pn.extension(sizing_mode="stretch_width")

# Registry of active ScanBrowser instances by serial, for the picks REST endpoint.
_browsers: dict[str, "ScanBrowser"] = {}

# --- Long-lived render worker (one subprocess for the lifetime of the server) ---

_RENDER_WORKER_SCRIPT = (
    "import sys\n"
    "_proto = sys.stdout\n"
    "sys.stdout = sys.stderr\n"
    "from explore.dataset import Scan\n"
    "for line in sys.stdin:\n"
    "    serial = line.strip()\n"
    "    if not serial: continue\n"
    "    try:\n"
    "        Scan.load(serial).render()\n"
    '        _proto.write("done " + serial + chr(10))\n'
    "        _proto.flush()\n"
    "    except Exception as e:\n"
    '        _proto.write("error " + serial + " " + str(e) + chr(10))\n'
    "        _proto.flush()\n"
)

_render_worker: subprocess.Popen | None = None
_render_callbacks: dict[str, callable] = {}
_render_lock = threading.Lock()


def _start_render_worker():
    """Launch the long-lived render subprocess and its stdout reader thread."""
    global _render_worker
    _render_worker = subprocess.Popen(
        [sys.executable, "-c", _RENDER_WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def reader():
        for line in _render_worker.stdout:
            parts = line.strip().split(" ", 1)
            if len(parts) >= 2 and parts[0] == "done":
                serial = parts[1]
                with _render_lock:
                    cb = _render_callbacks.pop(serial, None)
                if cb:
                    pn.state.execute(cb)

    threading.Thread(target=reader, daemon=True).start()


def _submit_render(serial: str, callback: callable):
    """Send a serial to the render worker; callback fires on the main thread when done."""
    if _render_worker is None or _render_worker.poll() is not None:
        _start_render_worker()
    with _render_lock:
        _render_callbacks[serial] = callback
    _render_worker.stdin.write(serial + "\n")
    _render_worker.stdin.flush()


class ScanBrowser(param.Parameterized):
    """Per-session state for one browser tab viewing one scan."""

    serial = param.String(default="")
    selected_index = param.Integer(default=0)
    picked_points = param.List(default=[])
    _picks_changed = param.Event()
    _transform_changed = param.Event()

    def __init__(self, serial: str, **params):
        super().__init__(serial=serial, **params)
        self._scan = None
        self._active_transform = None  # (operation, over_axis) or None
        self._dataset_keys = []
        self._tap_stream = hv.streams.Tap(transient=True)
        self._tap_stream.param.watch(self._on_tap, ["x", "y"])
        self._points_pipe = hv.streams.Pipe(data=[])
        self._sidebar_initialized = False
        self._grid_pane = None  # set by app() after grid HTML pane is created

        if serial:
            self._scan = Scan.load(serial)
            self._dataset_keys = sorted(
                [(ds.experiment, ds.repeat) for ds in self._scan.datasets],
                key=lambda k: (k[0], k[1]),
            )
            _browsers[serial] = self

    def _current_dataset(self):
        if not self._dataset_keys:
            return None
        exp, rep = self._dataset_keys[self.selected_index]
        return next(
            ds
            for ds in self._scan.datasets
            if ds.experiment == exp and ds.repeat == rep
        )

    def _current_key(self):
        if not self._dataset_keys:
            return ""
        exp, rep = self._dataset_keys[self.selected_index]
        return f"{exp}_{rep}"

    # --- Point picking ---

    def _on_tap(self, *event_stuff):
        x, y = self._tap_stream.x, self._tap_stream.y
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            return
        dataset = self._current_dataset()
        if dataset is None:
            return

        order = dataset.axis_plot_order if dataset.data.ndim >= 2 else [dataset.axes[0].name]
        x_name = order[0]
        y_name = order[1] if len(order) > 1 else None

        point = {
            "experiment": dataset.experiment,
            "repeat": dataset.repeat,
            x_name: float(x),
        }

        if dataset.data.ndim == 1:
            # For 1D: y is the data value; also look up nearest actual value
            x_ax = dataset.axes[0]
            xi = np.argmin(np.abs(x_ax.values - x))
            point["value"] = float(dataset.data[xi])
        elif dataset.data.ndim >= 2:
            point[y_name] = float(y)
            # Look up data value from the current 2D slice
            x_ax = next(a for a in dataset.axes if a.name == x_name)
            y_ax = next(a for a in dataset.axes if a.name == y_name)
            xi = np.argmin(np.abs(x_ax.values - x))
            yi = np.argmin(np.abs(y_ax.values - y))
            data_2d, _, _ = dataset.slice_2d()
            point["value"] = float(data_2d[yi, xi])

            # Record current slider positions for extra dimensions
            for slider_name in order[2:]:
                ax = next(a for a in dataset.axes if a.name == slider_name)
                point[slider_name] = float(ax.values[len(ax.values) // 2])

        self.picked_points = self.picked_points + [point]
        # Update points overlay via pipe (no full re-render)
        pts = self._points_for_current()
        self._points_pipe.send(self._pick_coords(pts))
        self.param.trigger("_picks_changed")

    def _points_for_current(self):
        """Get picked points for the currently selected dataset."""
        if not self._dataset_keys:
            return []
        exp, rep = self._dataset_keys[self.selected_index]
        return [
            p
            for p in self.picked_points
            if p["experiment"] == exp and p["repeat"] == rep
        ]

    def _pick_coords(self, pts):
        """Extract (plot_x, plot_y) tuples from pick dicts using axis names."""
        if not pts:
            return []
        dataset = self._current_dataset()
        if dataset is None:
            return []
        if dataset.data.ndim == 1:
            x_name = dataset.axes[0].name
            return [(p[x_name], p["value"]) for p in pts]
        order = dataset.axis_plot_order
        x_name, y_name = order[0], order[1]
        return [(p[x_name], p[y_name]) for p in pts]

    # --- Sidebar: interactive plot ---

    @param.depends("selected_index", "_transform_changed")
    def sidebar_plot(self):
        if not self._sidebar_initialized:
            return pn.pane.Markdown("*Loading...*", width=680, height=200)
        dataset = self._current_dataset()
        if dataset is None:
            return pn.pane.Markdown("*No dataset selected*")

        # Apply active transform if set
        if self._active_transform is not None:
            op, kwargs = self._active_transform
            dataset = apply_transform(dataset, op, **kwargs)

        exp, rep = dataset.experiment, dataset.repeat

        if dataset.data.ndim == 1:
            plot = hv.Curve(
                (dataset.axes[0].values, dataset.data),
                kdims=[dataset.axes[0].name],
            ).opts(
                width=680,
                height=450,
                tools=["hover"],
                title=f"{exp}_{rep}",
            )
        elif dataset.data.ndim >= 2:
            order = dataset.axis_plot_order
            x_name, y_name = order[0], order[1]
            slider_names = order[2:]

            img_opts = dict(
                colorbar=True,
                width=680,
                height=550,
                tools=["hover"],
                cmap="viridis",
                title=f"{exp}_{rep}",
            )

            # Build hv.Dataset with all dimensions, reordered to match axis_plot_order
            axis_names = [a.name for a in dataset.axes]
            perm = [axis_names.index(name) for name in reversed(order)]
            reordered = np.transpose(dataset.data, perm)

            coords = [
                next(a for a in dataset.axes if a.name == name).values
                for name in order
            ]
            hv_ds = hv.Dataset(
                (*coords, reordered),
                kdims=order,
                vdims=["value"],
            )

            if slider_names:
                plot = hv_ds.to(
                    hv.Image, kdims=[x_name, y_name], groupby=slider_names,
                    dynamic=True,
                ).opts(hv.opts.Image(**img_opts))
            else:
                plot = hv_ds.to(
                    hv.Image, kdims=[x_name, y_name],
                ).opts(**img_opts)
        else:
            return pn.pane.Markdown(f"Cannot display {dataset.data.ndim}D data")

        # Connect tap stream to base plot element
        self._tap_stream.source = plot

        # Points overlay via Pipe — updates without re-rendering base plot
        def make_points(data):
            if not data:
                return hv.Points([]).opts(
                    color="red", size=10, marker="x", line_width=3
                )
            xs, ys = zip(*data)
            return hv.Points((xs, ys)).opts(
                color="red", size=10, marker="x", line_width=3
            )

        points_dmap = hv.DynamicMap(make_points, streams=[self._points_pipe])

        # Push current picks for this dataset
        pts = self._points_for_current()
        self._points_pipe.send(self._pick_coords(pts))

        hv_pane = pn.pane.HoloViews(plot * points_dmap, widget_location="bottom")
        return hv_pane.layout

    # --- Sidebar: controls ---

    @param.depends("selected_index", "_picks_changed")
    def sidebar_controls(self):
        key = self._current_key()
        idx = self.selected_index
        total = len(self._dataset_keys)

        nav_label = pn.pane.Markdown(
            f"**{key}** &nbsp; ({idx + 1} / {total}) &nbsp; ← →",
            styles={"font-size": "14px"},
        )

        pts = self._points_for_current()

        # Build picks table with actual axis names
        if pts:
            # Get column names from the first pick (skip experiment/repeat)
            cols = [k for k in pts[0] if k not in ("experiment", "repeat")]
            header = "| " + " | ".join(cols) + " |"
            sep = "|" + "|".join("---" for _ in cols) + "|"
            rows = []
            for p in pts:
                cells = [f"{p[c]:.4f}" if isinstance(p.get(c), float) else str(p.get(c, "")) for c in cols]
                rows.append("| " + " | ".join(cells) + " |")
            table_md = f"{header}\n{sep}\n" + "\n".join(rows)
        else:
            table_md = "*Click on the plot to pick points.*"

        points_table = pn.pane.Markdown(table_md)

        export_btn = pn.widgets.Button(
            name="Export picks", button_type="success", width=150
        )
        export_btn.on_click(lambda e: self._export_picks())

        clear_btn = pn.widgets.Button(
            name="Clear picks", button_type="warning", width=150
        )
        clear_btn.on_click(lambda e: self._clear_current_picks())

        undo_btn = pn.widgets.Button(
            name="Undo pick", button_type="default", width=150
        )
        undo_btn.on_click(lambda e: self._undo_pick())

        total_picks = len(self.picked_points)
        current_picks = len(pts)

        pick_summary = pn.pane.Markdown(
            f"*{current_picks} picks on this plot, {total_picks} total*"
        )

        return pn.Column(
            nav_label,
            pn.layout.Divider(),
            points_table,
            pn.Row(export_btn, undo_btn, clear_btn),
            pick_summary,
        )

    def _export_picks(self):
        if not self._scan:
            return
        out_path = self._scan.dir / "picks.json"
        with open(out_path, "w") as f:
            json.dump(self.picked_points, f, indent=2)
        print(f"Exported {len(self.picked_points)} picks to {out_path}")

    def _undo_pick(self):
        """Remove the most recent pick on the current dataset."""
        if not self._dataset_keys:
            return
        exp, rep = self._dataset_keys[self.selected_index]
        # Find and remove the last pick for this dataset
        for i in range(len(self.picked_points) - 1, -1, -1):
            p = self.picked_points[i]
            if p["experiment"] == exp and p["repeat"] == rep:
                self.picked_points = self.picked_points[:i] + self.picked_points[i + 1:]
                pts = self._points_for_current()
                self._points_pipe.send(self._pick_coords(pts))
                self.param.trigger("_picks_changed")
                return

    def _clear_current_picks(self):
        if not self._dataset_keys:
            return
        exp, rep = self._dataset_keys[self.selected_index]
        self.picked_points = [
            p
            for p in self.picked_points
            if not (p["experiment"] == exp and p["repeat"] == rep)
        ]
        self._points_pipe.send([])
        self.param.trigger("_picks_changed")

    # --- Sidebar: transforms ---

    @param.depends("selected_index")
    def sidebar_transforms(self):
        dataset = self._current_dataset()
        if dataset is None:
            return pn.pane.Markdown("")

        axis_names = [a.name for a in dataset.axes]

        op_select = pn.widgets.Select(
            name="Operation",
            options=["none"] + list(TRANSFORM_REGISTRY),
            value="none" if self._active_transform is None else self._active_transform[0],
            width=200,
        )

        # Container for dynamically generated param widgets
        param_row = pn.Row()
        # Map from param name → widget, for collecting values
        param_widgets: dict[str, pn.widgets.Widget] = {}

        def _build_param_widgets(op_name):
            param_row.clear()
            param_widgets.clear()
            if op_name == "none":
                return
            spec = TRANSFORM_REGISTRY[op_name]
            prev_kwargs = self._active_transform[1] if self._active_transform and self._active_transform[0] == op_name else {}
            axis_widget = None
            for p in spec.params:
                if p["type"] == "axis":
                    w = pn.widgets.Select(
                        name=p["name"], options=axis_names,
                        value=prev_kwargs.get(p["name"], axis_names[0]),
                        width=150,
                    )
                    axis_widget = w
                elif p["type"] == "axis_value":
                    # Populate from the axis currently selected in the preceding axis widget
                    ax_name = axis_widget.value if axis_widget else axis_names[0]
                    ax_vals = next(a for a in dataset.axes if a.name == ax_name).values
                    options = [round(float(v), 6) for v in ax_vals]
                    default = prev_kwargs.get(p["name"], options[0])
                    if default not in options:
                        default = options[0]
                    w = pn.widgets.Select(
                        name=p["name"], options=options, value=default, width=150,
                    )
                    # Re-populate when the axis widget changes
                    if axis_widget is not None:
                        _aw, _vw = axis_widget, w  # capture for closure
                        def _update_axis_values(event, vw=_vw):
                            new_ax = next(a for a in dataset.axes if a.name == event.new)
                            vw.options = [round(float(v), 6) for v in new_ax.values]
                            vw.value = vw.options[0]
                        _aw.param.watch(_update_axis_values, "value")
                elif p["type"] == "float":
                    w = pn.widgets.FloatInput(
                        name=p["name"], value=prev_kwargs.get(p["name"], 0.0), width=150,
                    )
                else:
                    continue
                param_widgets[p["name"]] = w
                param_row.append(w)

        _build_param_widgets(op_select.value)
        op_select.param.watch(lambda e: _build_param_widgets(e.new), "value")

        def _collect_kwargs():
            return {name: w.value for name, w in param_widgets.items()}

        preview_btn = pn.widgets.Button(name="Preview", button_type="primary", width=120)
        apply_btn = pn.widgets.Button(name="Apply", button_type="success", width=120)

        def on_preview(event):
            if op_select.value == "none":
                self._active_transform = None
            else:
                self._active_transform = (op_select.value, _collect_kwargs())
            self.param.trigger("_transform_changed")

        def on_apply(event):
            op = op_select.value
            if op == "none":
                return
            kwargs = _collect_kwargs()
            new_scan = apply_to_scan(self._scan, op, **kwargs)
            new_scan.save()
            # Render via shared worker, open tab immediately (grid populates when done)
            _submit_render(new_scan.serial, lambda: None)
            notify(new_scan.serial)

        preview_btn.on_click(on_preview)
        apply_btn.on_click(on_apply)

        return pn.Column(
            pn.layout.Divider(),
            pn.pane.Markdown("**Transform**"),
            pn.Row(op_select),
            param_row,
            pn.Row(preview_btn, apply_btn),
        )

    # --- Main: thumbnail grid ---

    _THUMB_H = 150

    def _build_grid_html(self) -> str:
        """Generate raw HTML for the thumbnail grid. One string, zero Bokeh models."""
        thumb_dir = self._scan.dir / "thumbnails"
        img_w = int(self._THUMB_H * ASPECT_RATIO)

        cells = []
        for i, (exp, rep) in enumerate(self._dataset_keys):
            key = f"{exp}_{rep}"
            png_path = thumb_dir / f"{key}.png"
            if png_path.exists():
                img_tag = (
                    f'<img src="/thumbnails/{self.serial}/thumbnails/{key}.png"'
                    f' width="{img_w}" height="{self._THUMB_H}"'
                    f' style="display:block;" />'
                )
            else:
                img_tag = (
                    f'<div style="width:{img_w}px;height:{self._THUMB_H}px;'
                    f'background:#333;"></div>'
                )

            cells.append(
                f'<div class="thumb" data-idx="{i}" onclick="window._selectThumb({i})"'
                f' style="cursor:pointer;display:inline-block;'
                f'margin:2px;text-align:center;border:1px solid #555;">'
                f'<div style="background:#222;color:white;padding:4px 2px;'
                f'font-size:14px;font-weight:bold;width:{img_w}px;">{key}</div>'
                f'{img_tag}'
                f'</div>'
            )

        return (
            f'<div style="display:flex;flex-wrap:wrap;">'
            f'{"".join(cells)}'
            f'</div>'
        )

    def _start_background_render(self):
        """Render thumbnails in a subprocess, then update grid HTML at once."""
        thumb_dir = self._scan.dir / "thumbnails"

        needs_render = any(
            not (thumb_dir / f"{exp}_{rep}.png").exists()
            for exp, rep in self._dataset_keys
        )
        if not needs_render:
            return

        proc = subprocess.Popen(
            [sys.executable, "-c",
             f"from explore.dataset import Scan; Scan.load('{self.serial}').render()"],
        )

        def wait_and_rebuild():
            proc.wait()
            pn.state.execute(lambda: setattr(self._grid_pane, 'object', self._build_grid_html()))

        t = threading.Thread(target=wait_and_rebuild, daemon=True)
        t.start()


class GridNav(pn.reactive.ReactiveHTML):
    """Invisible element: bridges grid onclick and Tab/Shift-Tab → Python selected_index."""

    selected_index = param.Integer(default=0)
    max_index = param.Integer(default=0)

    _template = """<div id="nav" style="width:0;height:0;overflow:hidden;"></div>"""

    _scripts = {
        "render": """
        window._selectThumb = (idx) => { data.selected_index = idx; };
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab' && !e.shiftKey) {
                e.preventDefault();
                data.selected_index = Math.min(data.selected_index + 1, data.max_index);
            } else if (e.key === 'Tab' && e.shiftKey) {
                e.preventDefault();
                data.selected_index = Math.max(data.selected_index - 1, 0);
            }
        });
        """,
    }


def app():
    """Panel app factory — called once per browser session."""
    serial = ""
    if pn.state.session_args.get("serial"):
        serial = pn.state.session_args["serial"][0].decode("utf-8")

    browser = ScanBrowser(serial=serial)

    # Grid: plain HTML pane (handles any size, no Bokeh model overhead)
    grid_pane = pn.pane.HTML(
        browser._build_grid_html() if browser._scan else "",
        sizing_mode="stretch_width",
    )
    browser._grid_pane = grid_pane

    # Nav: tiny invisible ReactiveHTML bridging grid onclick → Python
    nav = GridNav(max_index=max(len(browser._dataset_keys) - 1, 0))
    nav.param.watch(
        lambda e: setattr(browser, "selected_index", e.new), "selected_index"
    )

    # Kick off background rendering now that grid pane is wired up
    if browser._scan:
        browser._start_background_render()

    # Defer sidebar: render placeholder initially, full HoloViews after page loads
    def _init_sidebar():
        browser._sidebar_initialized = True
        browser.param.trigger("_transform_changed")  # forces sidebar_plot re-eval

    pn.state.onload(_init_sidebar)

    template = pn.template.FastListTemplate(
        title=f"Scan: {serial}" if serial else "Explore",
        sidebar_width=750,
        sidebar=[browser.sidebar_plot, browser.sidebar_controls, browser.sidebar_transforms],
        main=[nav, grid_pane],
        theme="dark",
    )

    return template


def _make_extra_patterns():
    """Create Tornado URL patterns for REST endpoints."""
    import tornado.web

    class PicksHandler(tornado.web.RequestHandler):
        def get(self):
            serial = self.get_argument("serial", "")
            browser = _browsers.get(serial)
            if browser is None:
                self.set_status(404)
                self.write({"error": f"No active browser for serial={serial}"})
                return
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(browser.picked_points))

    return [
        ("/picks", PicksHandler),
        (r"/thumbnails/(.*)", tornado.web.StaticFileHandler, {"path": str(DATA_DIR)}),
    ]


def serve(port: int = 5006, show: bool = False) -> None:
    """Start the Panel server (blocking)."""
    # Start the render worker early so it pays the import cost while the server boots
    _start_render_worker()
    pn.serve(
        {"explore": app},
        port=port,
        allow_websocket_origin=[f"localhost:{port}"],
        show=show,
        extra_patterns=_make_extra_patterns(),
    )


if __name__ == "__main__":
    serve()
