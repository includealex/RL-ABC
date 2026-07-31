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
from rl_framework.Utils import run_episode
from train import RLConfig


def _write_html(figure: Any, path: str | Path) -> Path:
    """Write a Plotly figure as a self-contained notebook-friendly HTML file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs="cdn")
    return output


def _ordered_values(
    variable_names: Sequence[str],
    settings: Mapping[str, float] | Sequence[float] | None,
) -> list[float]:
    """Return settings in the chronological order required by the environment."""
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
        if settings is not None:
            variable_names = env.wrapper.chroneological_variables
            values = _ordered_values(variable_names, settings)
            run_episode(env, source="Values", actions=values)

        results_path = Path(env.wrapper.results_path)
        required_files = (
            results_path / "xyz.sdds",
            results_path / "beamline.mag",
        )
        missing_files = [path for path in required_files if not path.is_file()]
        if missing_files:
            elegant_error = getattr(
                env.wrapper,
                "last_elegant_error",
                None,
            )
            details = (
                f" Last Elegant error: {elegant_error}"
                if elegant_error
                else ""
            )
            raise FileNotFoundError(
                "Elegant did not create the files required for the beamline "
                f"layout: {', '.join(map(str, missing_files))}.{details}"
            )

        magnet_geometry = env.wrapper.get_mag_3d()
        figure = env.wrapper.plot_mag_3d(magnet_geometry, show=show)
        if output_html is not None:
            _write_html(figure, output_html)
        return figure
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

    The returned dictionary contains ``betatron``, ``dispersion``, and
    ``layout`` Plotly figures. Each figure is also written to ``output_dir``.
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

        wrapper = env.wrapper
        figures = {
            "betatron": wrapper.plot_betatron(),
            "dispersion": wrapper.plot_dispersion(),
        }
        magnet_geometry = wrapper.get_mag_3d()
        figures["layout"] = wrapper.plot_mag_3d(
            magnet_geometry,
            show=show,
        )

        for name, figure in figures.items():
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
