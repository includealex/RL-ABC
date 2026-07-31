"""Reusable Elegant-based visualizations for notebooks and scripts.

The functions in this module run the configured Elegant simulation before
reading ``xyz.sdds`` and ``beamline.mag``. Beamline geometry therefore comes
from the selected lattice instead of a hand-written drawing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from config_manager import load_config
from rl_framework.visulize import ElegantVisualizer, _ordered_values
from train import RLConfig


def _write_html(figure: Any, path: str | Path) -> Path:
    """Write a Plotly figure as a self-contained notebook-friendly HTML file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs="cdn")
    return output


def _require_elegant_outputs(env: Any, filenames: Sequence[str]) -> None:
    """Raise a useful error when Elegant did not create expected SDDS files."""
    results_path = Path(env.wrapper.results_path)
    required_files = tuple(results_path / name for name in filenames)
    missing_files = [path for path in required_files if not path.is_file()]
    if not missing_files:
        return

    elegant_error = getattr(env.wrapper, "last_elegant_error", None)
    details = f" Last Elegant error: {elegant_error}" if elegant_error else ""
    raise FileNotFoundError(
        "Elegant did not create the files required for visualization: "
        f"{', '.join(map(str, missing_files))}.{details}"
    )


def plot_configured_beamline(
    config_path: str | Path,
    *,
    output_html: str | Path | None = None,
    settings: Mapping[str, float] | Sequence[float] | None = None,
    show: bool = False,
):
    """Run Elegant and plot the configured beamline floor geometry.

    Environment construction already performs an Elegant reset simulation with
    all controllable parameters set to zero. A second full episode is run only
    when explicit ``settings`` are supplied. A mapping may provide only
    selected parameters; unspecified parameters remain zero.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure created by :meth:`ElegantWrapper.plot_mag_3d`.
    """
    config = RLConfig(load_config(str(config_path)))
    env, _, file_handler = config.setup_environment()

    try:
        visualizer = ElegantVisualizer(
            env,
            settings,
            run_simulation=settings is not None,
        )
        _require_elegant_outputs(env, ("xyz.sdds", "beamline.mag"))
        return visualizer.plot_mag_3d(
            output_html=output_html,
            show=show,
        )
    finally:
        file_handler.close()


def plot_configured_optics(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    settings: Mapping[str, float] | Sequence[float] | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Run one Elegant configuration and save its layout and Twiss envelope."""
    config = RLConfig(load_config(str(config_path)))
    env, _, file_handler = config.setup_environment()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    try:
        visualizer = ElegantVisualizer(
            env,
            settings,
            vis_dir=output / "visualizer",
            run_simulation=settings is not None,
        )
        _require_elegant_outputs(
            env,
            (
                "xyz.sdds",
                "beamline.mag",
                "twiss.twi",
                f"{env.wrapper.input_beam_file}.sig",
                f"{env.wrapper.input_beam_file}.cen",
            ),
        )
        layout = visualizer.plot_mag_3d(
            output_html=output / "layout.html",
            show=show,
        )
        twiss, _ = visualizer.plot_twiss(
            output_html=output / "twiss.html",
        )
        return {"layout": layout, "twiss": twiss}
    finally:
        file_handler.close()


def plot_trained_policy(
    config_path: str | Path,
    *,
    output_dir: str | Path,
    load_model_tag: str = "last",
    show: bool = False,
) -> dict[str, Any]:
    """Evaluate a saved DDPG policy and create standard Elegant plots.

    The returned dictionary contains ``betatron``, ``dispersion``, ``layout``,
    and ``twiss`` figures. Each figure is also written to ``output_dir``.
    """
    config_dict = load_config(str(config_path))
    config_dict.setdefault("training", {})["load_model"] = True
    config_dict["training"]["load_model_tag"] = load_model_tag
    config = RLConfig(config_dict)

    env, _, file_handler = config.setup_environment()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    try:
        agent = config.setup_agent(env)
        config.load_agent_weights(agent)
        agent.evaluate(episodes=1)

        visualizer = ElegantVisualizer(
            env,
            vis_dir=output / "visualizer",
            run_simulation=False,
        )
        figures = {
            "betatron": visualizer.plot_betatron(),
            "dispersion": visualizer.plot_dispersion(),
        }
        figures["layout"] = visualizer.plot_mag_3d(
            output_html=output / "layout.html",
            show=show,
        )
        figures["twiss"], _ = visualizer.plot_twiss(
            output_html=output / "twiss.html",
        )

        for name in ("betatron", "dispersion"):
            figure = figures[name]
            _write_html(figure, output / f"{name}.html")
        return figures
    finally:
        file_handler.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Elegant-derived RL-ABC visualizations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    layout = subparsers.add_parser(
        "layout",
        help="Run the configured beamline and plot its floor geometry.",
    )
    layout.add_argument("--config", required=True)
    layout.add_argument("--output", required=True)
    layout.add_argument("--show", action="store_true")

    overview = subparsers.add_parser(
        "overview",
        help="Run the configured beamline and save its layout and Twiss plots.",
    )
    overview.add_argument("--config", required=True)
    overview.add_argument("--output-dir", required=True)
    overview.add_argument("--show", action="store_true")

    policy = subparsers.add_parser(
        "policy",
        help="Evaluate a saved policy and create standard optics plots.",
    )
    policy.add_argument("--config", required=True)
    policy.add_argument("--output-dir", required=True)
    policy.add_argument(
        "--load-model-tag",
        choices=("last", "best"),
        default="last",
    )
    policy.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "layout":
        output = Path(args.output)
        plot_configured_beamline(
            args.config,
            output_html=output,
            show=args.show,
        )
        print(output.resolve())
        return

    if args.command == "overview":
        output_dir = Path(args.output_dir)
        plot_configured_optics(
            args.config,
            output_dir=output_dir,
            show=args.show,
        )
        print(output_dir.resolve())
        return

    output_dir = Path(args.output_dir)
    plot_trained_policy(
        args.config,
        output_dir=output_dir,
        load_model_tag=args.load_model_tag,
        show=args.show,
    )
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
