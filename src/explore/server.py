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
from .constants import ASPECT_RATIO


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

        self._thumb_panes: list[pn.pane.PNG | None] = []
        self._thumb_btns: list[pn.widgets.Button | None] = []
        self._grid = None

        if serial:
            self._scan = Scan.load(serial)
            # Build sorted key list: [(exp, rep), ...]
            self._dataset_keys = sorted(
                [(ds.experiment, ds.repeat) for ds in self._scan.datasets],
                key=lambda k: (k[0], k[1]),
            )
            self._grid = self._build_grid()
            # Kick off background thumbnail rendering
            self._start_background_render()
            self.param.watch(self._update_grid_highlight, "selected_index")
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
        dataset = self._current_dataset()
        if dataset is None:
            return pn.pane.Markdown("*No dataset selected*")

        # Apply active transform if set
        if self._active_transform is not None:
            op, over_ax = self._active_transform
            dataset = apply_transform(dataset, op, over_ax)

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
        axis_select = pn.widgets.Select(
            name="Over axis",
            options=axis_names,
            value=self._active_transform[1] if self._active_transform else axis_names[0],
            width=200,
        )

        preview_btn = pn.widgets.Button(name="Preview", button_type="primary", width=120)
        apply_btn = pn.widgets.Button(name="Apply", button_type="success", width=120)

        def on_preview(event):
            if op_select.value == "none":
                self._active_transform = None
            else:
                self._active_transform = (op_select.value, axis_select.value)
            self.param.trigger("_transform_changed")

        def on_apply(event):
            op = op_select.value
            if op == "none":
                return
            axis = axis_select.value
            new_scan = apply_to_scan(self._scan, op, axis)
            new_scan.save()
            # Render via shared worker, open tab immediately (grid populates when done)
            _submit_render(new_scan.serial, lambda: None)
            notify(new_scan.serial)

        preview_btn.on_click(on_preview)
        apply_btn.on_click(on_apply)

        return pn.Column(
            pn.layout.Divider(),
            pn.pane.Markdown("**Transform**"),
            pn.Row(op_select, axis_select),
            pn.Row(preview_btn, apply_btn),
        )

    # --- Main: thumbnail grid ---

    _BTN_SELECTED = ":host .bk-btn { background: #2196F3; color: white; font-size: 16px; font-weight: bold }"
    _BTN_DEFAULT = ":host .bk-btn { background: #222; color: white; font-size: 16px }"

    _THUMB_H = 150

    def _build_grid(self):
        """Build the thumbnail grid with buttons above their PNGs."""
        thumb_dir = self._scan.dir / "thumbnails"
        img_w = int(self._THUMB_H * ASPECT_RATIO)
        col_w = img_w + 10
        col_h = self._THUMB_H + 45

        thumbs = []
        for i, (exp, rep) in enumerate(self._dataset_keys):
            key = f"{exp}_{rep}"
            png_path = thumb_dir / f"{key}.png"

            is_selected = i == self.selected_index
            if png_path.exists():
                img = pn.pane.PNG(
                    str(png_path),
                    width=img_w,
                    height=self._THUMB_H,
                    styles={"border": "3px solid #2196F3" if is_selected else "1px solid #ddd"},
                )
            else:
                img = pn.pane.PNG(
                    None,
                    width=img_w,
                    height=self._THUMB_H,
                    styles={
                        "border": "3px solid #2196F3" if is_selected else "1px solid #ddd",
                        "background": "#333",
                    },
                )
            btn = pn.widgets.Button(
                name=key,
                width=img_w,
                height=35,
                button_type="default",
                stylesheets=[self._BTN_SELECTED if is_selected else self._BTN_DEFAULT],
            )
            btn.on_click(
                lambda e, idx=i: setattr(self, "selected_index", idx)
            )
            self._thumb_panes.append(img)
            self._thumb_btns.append(btn)
            thumbs.append(pn.Column(btn, img, width=col_w, height=col_h, margin=2))

        return pn.Column(
            pn.pane.Markdown(f"## Scan: {self.serial} &nbsp; ({len(self._dataset_keys)} datasets)"),
            pn.FlexBox(*thumbs),
            sizing_mode="stretch_width",
        )

    def _start_background_render(self):
        """Submit rendering to the shared worker subprocess."""
        thumb_dir = self._scan.dir / "thumbnails"

        # Check if any thumbnails are missing
        needs_render = any(
            not (thumb_dir / f"{exp}_{rep}.png").exists()
            for exp, rep in self._dataset_keys
        )
        if not needs_render:
            return

        def on_done():
            self._thumb_panes.clear()
            self._thumb_btns.clear()
            new_grid = self._build_grid()
            self._grid[1] = new_grid[1]

        _submit_render(self.serial, on_done)

    def _update_grid_highlight(self, event):
        """Update only the two affected thumbnails' styles. O(1) not O(n)."""
        for idx, is_sel in [(event.old, False), (event.new, True)]:
            if 0 <= idx < len(self._thumb_panes) and self._thumb_panes[idx]:
                self._thumb_panes[idx].styles = {
                    "border": "3px solid #2196F3" if is_sel else "1px solid #ddd"
                }
                self._thumb_btns[idx].stylesheets = [
                    self._BTN_SELECTED if is_sel else self._BTN_DEFAULT
                ]

    def grid_view(self):
        """Return the pre-built grid. No @param.depends — never rebuilds."""
        if not self._grid:
            return pn.pane.Markdown(
                "# Waiting for scan...\n\n"
                "The experiment process will open a tab automatically."
            )
        return self._grid


class KeyNav(pn.reactive.ReactiveHTML):
    """Invisible element that captures global arrow key events."""

    index = param.Integer(default=0)
    max_index = param.Integer(default=0)

    _template = """<div id="keynav" style="width:0;height:0;overflow:hidden;"></div>"""

    _scripts = {
        "render": """
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab' && !e.shiftKey) {
                e.preventDefault();
                data.index = Math.min(data.index + 1, data.max_index);
            } else if (e.key === 'Tab' && e.shiftKey) {
                e.preventDefault();
                data.index = Math.max(data.index - 1, 0);
            }
        });
        """
    }


def app():
    """Panel app factory — called once per browser session."""
    serial = ""
    if pn.state.session_args.get("serial"):
        serial = pn.state.session_args["serial"][0].decode("utf-8")

    browser = ScanBrowser(serial=serial)

    # Wire keyboard nav to browser
    keynav = KeyNav(max_index=max(len(browser._dataset_keys) - 1, 0))
    keynav.param.watch(
        lambda e: setattr(browser, "selected_index", e.new), "index"
    )
    browser.param.watch(
        lambda e: setattr(keynav, "index", e.new), "selected_index"
    )

    template = pn.template.FastListTemplate(
        title=f"Scan: {serial}" if serial else "Explore",
        sidebar_width=750,
        sidebar=[browser.sidebar_plot, browser.sidebar_controls, browser.sidebar_transforms],
        main=[keynav, browser.grid_view],
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

    return [("/picks", PicksHandler)]


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
