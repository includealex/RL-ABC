import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import holoviews as hv
import pandas as pd
import plotly.graph_objects as go

from rl_framework.Elegant import ElegantWrapper
from rl_framework.Utils import run_episode
from rl_framework.visulize import ElegantVisualizer
from rl_framework.visualization import (
    _ordered_values,
    _write_html,
    plot_configured_beamline,
)


class VisualizationHelpersTest(unittest.TestCase):
    def test_ordered_values_defaults_and_mapping(self):
        names = ["QF1K1", "QF1HKICK", "QF1VKICK"]
        self.assertEqual(_ordered_values(names, None), [0.0, 0.0, 0.0])
        self.assertEqual(
            _ordered_values(names, {"QF1K1": 2.5}),
            [2.5, 0.0, 0.0],
        )

    def test_ordered_values_rejects_wrong_sequence_length(self):
        with self.assertRaises(ValueError):
            _ordered_values(["QF1K1", "QD1K1"], [1.0])

    def test_html_writer_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "figure.html"
            written = _write_html(go.Figure(), output)
            self.assertEqual(written, output)
            self.assertTrue(output.is_file())

    def test_plot_mag_3d_returns_figure_without_showing(self):
        geometry = pd.DataFrame(
            {
                "Z": [0.0, 1.0],
                "X": [0.0, 0.1],
                "ElementName": ["QF1", "QD1"],
            }
        )
        wrapper = ElegantWrapper.__new__(ElegantWrapper)
        figure = wrapper.plot_mag_3d(geometry, show=False)
        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(list(figure.data[0].hovertext), ["QF1", "QD1"])

    def test_run_episode_uses_current_count_from_mask_pair(self):
        environment = Mock()
        environment.reset.return_value = [0.0]
        environment._check_number_of_variables_to_be_set_at_this_iteration.return_value = (
            3,
            4,
        )
        environment.step.return_value = (
            [1.0],
            1.0,
            True,
            False,
            {
                "output_file": "watch.sdds",
                "number_of_particles": 10,
            },
        )
        environment.get_number_of_particles.return_value = 10

        run_episode(
            environment,
            source="Values",
            actions=[1.0, 2.0, 3.0],
        )

        action = environment.step.call_args.args[0]
        self.assertEqual(action.tolist(), [1.0, 2.0, 3.0, 0.0])

    def test_visualizer_writes_twiss_html(self):
        environment = Mock()
        environment.seed = 4
        environment.wrapper.chroneological_variables = []
        environment.wrapper.plot_twiss.return_value = hv.Curve(
            [(0.0, 1.0), (1.0, 2.0)]
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "twiss.html"
            visualizer = ElegantVisualizer(
                environment,
                run_simulation=False,
            )
            twiss, html = visualizer.plot_twiss(output_html=output)

            self.assertIsInstance(twiss, hv.Curve)
            self.assertTrue(output.is_file())
            self.assertIn("<html", html.data.lower())

    @patch("rl_framework.visulize.run_episode")
    @patch("rl_framework.visualization.RLConfig")
    @patch("rl_framework.visualization.load_config")
    def test_configured_layout_uses_elegant_geometry(
        self,
        load_config_mock,
        config_class_mock,
        run_episode_mock,
    ):
        load_config_mock.return_value = {"training": {"seed": 0}}

        wrapper = Mock()
        wrapper.chroneological_variables = ["QF1K1", "QD1K1"]
        wrapper.results_path = "results_0"
        geometry = pd.DataFrame(
            {
                "Z": [0.0, 1.0],
                "X": [0.0, 0.0],
                "ElementName": ["QF1", "QD1"],
            }
        )
        wrapper.get_mag_3d.return_value = geometry
        wrapper.plot_mag_3d.return_value = go.Figure()

        environment = Mock()
        environment.seed = 0
        environment.wrapper = wrapper
        file_handler = Mock()
        config_class_mock.return_value.setup_environment.return_value = (
            environment,
            Mock(),
            file_handler,
        )

        with tempfile.TemporaryDirectory() as directory:
            wrapper.results_path = directory
            Path(directory, "xyz.sdds").touch()
            Path(directory, "beamline.mag").touch()
            figure = plot_configured_beamline("demo.json")

        self.assertIsInstance(figure, go.Figure)
        run_episode_mock.assert_not_called()
        wrapper.get_mag_3d.assert_called_once_with()
        wrapper.plot_mag_3d.assert_called_once_with(geometry, show=False)
        file_handler.close.assert_called_once_with()

    @patch("rl_framework.visulize.run_episode")
    @patch("rl_framework.visualization.RLConfig")
    @patch("rl_framework.visualization.load_config")
    def test_explicit_layout_settings_run_an_episode(
        self,
        load_config_mock,
        config_class_mock,
        run_episode_mock,
    ):
        load_config_mock.return_value = {"training": {"seed": 0}}
        wrapper = Mock()
        wrapper.chroneological_variables = ["QF1K1", "QD1K1"]
        wrapper.get_mag_3d.return_value = pd.DataFrame(
            {"Z": [], "X": [], "ElementName": []}
        )
        wrapper.plot_mag_3d.return_value = go.Figure()
        environment = Mock()
        environment.seed = 0
        environment.wrapper = wrapper
        file_handler = Mock()
        config_class_mock.return_value.setup_environment.return_value = (
            environment,
            Mock(),
            file_handler,
        )

        with tempfile.TemporaryDirectory() as directory:
            wrapper.results_path = directory
            Path(directory, "xyz.sdds").touch()
            Path(directory, "beamline.mag").touch()
            plot_configured_beamline(
                "demo.json",
                settings={"QF1K1": 1.5},
            )

        run_episode_mock.assert_called_once_with(
            environment,
            source="Values",
            actions=[1.5, 0.0],
        )

    @patch("rl_framework.visualization.RLConfig")
    @patch("rl_framework.visualization.load_config")
    def test_missing_elegant_layout_files_report_expected_paths(
        self,
        load_config_mock,
        config_class_mock,
    ):
        load_config_mock.return_value = {"training": {"seed": 0}}
        environment = Mock()
        environment.seed = 0
        environment.wrapper.chroneological_variables = []
        environment.wrapper.last_elegant_error = "example Elegant failure"
        file_handler = Mock()
        config_class_mock.return_value.setup_environment.return_value = (
            environment,
            Mock(),
            file_handler,
        )

        with tempfile.TemporaryDirectory() as directory:
            environment.wrapper.results_path = directory
            with self.assertRaisesRegex(
                FileNotFoundError,
                "xyz.sdds.*beamline.mag.*example Elegant failure",
            ):
                plot_configured_beamline("demo.json")

        file_handler.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
