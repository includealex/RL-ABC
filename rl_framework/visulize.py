"""Notebook-friendly visualizations for an RL-ABC Elegant environment.

The filename keeps the historical ``visulize`` spelling so existing notebooks
can continue to import ``ElegantVisualizer``.  All plots are derived from the
SDDS files produced by Elegant; no beamline positions are hard-coded here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape
import warnings
from pathlib import Path
from typing import Any

import holoviews as hv
from IPython.display import HTML

from rl_framework.Utils import run_episode


def _ordered_values(
    variable_names: Sequence[str],
    settings: Mapping[str, float] | Sequence[float] | None,
) -> list[float]:
    """Return settings in the chronological order expected by the environment."""
    if settings is None:
        return [0.0] * len(variable_names)
    if isinstance(settings, Mapping):
        return [float(settings.get(name, 0.0)) for name in variable_names]

    values = [float(value) for value in settings]
    if len(values) != len(variable_names):
        raise ValueError(
            "Expected one value for each controllable parameter: "
            f"{len(variable_names)} values, received {len(values)}."
        )
    return values


class ElegantVisualizer:
    """Create interactive plots from an initialized RL-ABC environment.

    Parameters
    ----------
    env:
        An :class:`ACCElegantEnvironment`.
    settings:
        Optional magnet settings. A dictionary is reordered using Elegant's
        chronological variable list; omitted dictionary entries remain zero.
    vis_dir:
        Directory used for reusable HTML files.
    seed:
        Optional visualization suffix. The environment seed is used by default.
    run_simulation:
        Run one complete sequential RL-ABC episode before plotting. Set this to
        ``False`` only when the caller has just run or evaluated the environment.
    """

    def __init__(
        self,
        env: Any,
        settings: Mapping[str, float] | Sequence[float] | None = None,
        vis_dir: str | Path = "vis",
        seed: int | None = None,
        *,
        run_simulation: bool = True,
    ) -> None:
        self.env = env
        self.seed = int(
            seed if seed is not None else getattr(env, "seed", 0)
        )
        self.vis_dir = Path(f"{vis_dir}_{self.seed}")
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        hv.extension("bokeh")

        variable_names = env.wrapper.chroneological_variables
        self.values = _ordered_values(variable_names, settings)
        self.val_dict = dict(zip(variable_names, self.values))

        if run_simulation:
            run_episode(
                self.env,
                source="Values",
                actions=self.values,
            )

    @staticmethod
    def _read_html(path: Path) -> HTML:
        """Embed a standalone plot document in an isolated notebook iframe."""
        document = path.read_text(encoding="utf-8")
        # IPython warns for any inline iframe even though srcdoc is precisely
        # what keeps this self-contained. Suppress only that known warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Consider using IPython.display.IFrame instead",
                category=UserWarning,
            )
            return HTML(
                '<iframe '
                f'srcdoc="{escape(document, quote=True)}" '
                'style="width:100%; height:650px; border:0;" '
                'sandbox="allow-scripts allow-same-origin" '
                'loading="lazy"></iframe>'
            )

    def plot_magnets(
        self,
        output_html: str | Path | None = None,
    ):
        """Plot the longitudinal Elegant magnet profile."""
        magnets = self.env.wrapper.plot_magnets()
        path = Path(output_html) if output_html else (
            self.vis_dir / "magnet_plot.html"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        hv.save(magnets, path, fmt="html")
        return magnets, self._read_html(path)

    def plot_twiss(
        self,
        output_html: str | Path | None = None,
    ):
        """Plot transverse beam envelopes, apertures, and magnet positions."""
        twiss = self.env.wrapper.plot_twiss()
        path = Path(output_html) if output_html else (
            self.vis_dir / "twiss_plot.html"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        hv.save(twiss, path, fmt="html")
        return twiss, self._read_html(path)

    def plot_mag_3d(
        self,
        output_html: str | Path | None = None,
        *,
        show: bool = True,
    ):
        """Plot the Elegant floor-coordinate geometry as a Plotly figure."""
        magnet_geometry = self.env.wrapper.get_mag_3d()
        figure = self.env.wrapper.plot_mag_3d(
            magnet_geometry,
            show=show,
        )
        if output_html is not None:
            path = Path(output_html)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.write_html(path, include_plotlyjs="cdn")
        return figure

    def plot_betatron(self):
        """Return the Elegant betatron-function Plotly figure."""
        return self.env.wrapper.plot_betatron()

    def plot_dispersion(self):
        """Return the Elegant dispersion Plotly figure."""
        return self.env.wrapper.plot_dispersion()
