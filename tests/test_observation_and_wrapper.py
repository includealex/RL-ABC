"""Regression tests for the shared 59-D observation and strengthened wrapper."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pickle

import numpy as np
import pandas as pd

from rl_framework.Elegant import ElegantWrapper as CoreElegantWrapper
from rl_framework.Elegant_edited import ElegantWrapper as EnhancedElegantWrapper
from rl_framework.Environment import ACCElegantEnvironment
from rl_framework.Agents.DDPG import ReplayBuffer
from rl_framework.Utils import (
    configure_initial_particles,
    create_nn_representation_new,
    lattice_context_for_watch_point,
    nn_observation_size,
    reset_specific_keys,
)
from rl_framework.Utils_magnet_actions import (
    build_magnet_action_registry,
    normalize_reset_keys,
    reset_policy_mask,
)


class ObservationSchemaTests(unittest.TestCase):
    def test_default_schema_is_59_and_context_is_last(self):
        frame = pd.DataFrame(columns=["x", "y", "xp", "yp"])
        observation = create_nn_representation_new(
            frame, 0.1, 0.2, 0.3, 0.4, n_bins=5, angle=-0.15, tilt=0.025
        )
        self.assertEqual(nn_observation_size(5), 59)
        self.assertEqual(observation.shape, (59,))
        np.testing.assert_allclose(observation[-2:], [-0.15, 0.025])

    def test_histogram_size_is_dynamic(self):
        self.assertEqual(nn_observation_size(7), 83)

    def test_all_supported_bend_aliases_and_reversal(self):
        graph = [
            {"name": "WB", "type": "WATCH", "attributes": {}},
            {
                "name": "B1",
                "type": "CSBEND",
                "attributes": {"ANGLE": "0.2", "TILT": "-0.1"},
                "reverse": True,
            },
        ]
        self.assertEqual(lattice_context_for_watch_point("WB", graph), (-0.2, -0.1))

    def test_quadrupole_has_zero_angle_and_real_tilt(self):
        graph = [
            {"name": "WQ", "type": "WATCH", "attributes": {}},
            {"name": "Q1", "type": "KQUAD", "attributes": {"TILT": "0.3"}},
        ]
        self.assertEqual(lattice_context_for_watch_point("WQ", graph), (0.0, 0.3))

    def test_dipole_without_tilt_defaults_to_zero(self):
        graph = [
            {"name": "WB", "type": "WATCH", "attributes": {}},
            {"name": "B1", "type": "SBEND", "attributes": {"ANGLE": "0.4"}},
        ]
        self.assertEqual(lattice_context_for_watch_point("WB", graph), (0.4, 0.0))

    def test_invalid_context_is_not_silently_hidden(self):
        graph = [
            {"name": "WB", "type": "WATCH", "attributes": {}},
            {"name": "B1", "type": "SBEND", "attributes": {"ANGLE": "bad"}},
        ]
        with self.assertRaisesRegex(ValueError, "Invalid ANGLE"):
            lattice_context_for_watch_point("WB", graph)

    def test_environment_schema_guard_rejects_legacy_width(self):
        environment = ACCElegantEnvironment.__new__(ACCElegantEnvironment)
        environment.n_bins = 5
        with self.assertRaisesRegex(RuntimeError, "expected 59"):
            environment._validated_observation(np.zeros(58), "test")
        accepted = environment._validated_observation(np.zeros(59), "test")
        self.assertEqual(accepted.dtype, np.float32)


class MagnetRegistryTests(unittest.TestCase):
    def test_registry_is_occurrence_specific(self):
        graph = [
            {
                "name": "Q1_1",
                "type": "QUAD",
                "attributes": {"K1": "0", "HKICK": "0"},
                "controllable": True,
                "controllable_parameters": ["K1", "HKICK"],
            },
            {
                "name": "B1",
                "type": "CSBEND",
                "attributes": {"FSE": "0", "ANGLE": "0.1"},
                "controllable": True,
                "controllable_parameters": ["FSE"],
            },
        ]
        variables = ["Q1_1K1", "Q1_1HKICK", "B1FSE"]
        registry = build_magnet_action_registry(
            graph, variables, chronological_order=["Q1_1", "B1"]
        )
        np.testing.assert_array_equal(
            registry.availability_policy_mask(0), [1, 1, 0, 0]
        )
        np.testing.assert_array_equal(
            registry.availability_policy_mask(1), [0, 0, 0, 1]
        )

    def test_reset_keys_can_select_any_policy_channels(self):
        np.testing.assert_array_equal(
            reset_policy_mask(True, reset_keys=["K1", "FSE"]),
            [0, 1, 1, 0],
        )
        np.testing.assert_array_equal(
            reset_policy_mask(True, reset_keys=["HKICK", "VKICK"]),
            [1, 0, 0, 1],
        )
        np.testing.assert_array_equal(
            reset_policy_mask(True, reset_keys=[]),
            [1, 1, 1, 1],
        )
        np.testing.assert_array_equal(
            reset_policy_mask(
                True, reset_keys=["K1", "HKICK", "VKICK", "FSE"]
            ),
            [0, 0, 0, 0],
        )

    def test_reset_keys_are_case_insensitive_validated_and_canonical(self):
        self.assertEqual(normalize_reset_keys(["fse", "k1", "K1"]), ("K1", "FSE"))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            normalize_reset_keys(["ANGLE"])
        with self.assertRaisesRegex(TypeError, "must be a list"):
            normalize_reset_keys("K1")

    def test_wrapper_side_zeroing_uses_the_same_configured_keys(self):
        values = {"Q1K1": 2.0, "Q1HKICK": 0.1, "Q1VKICK": 0.2, "B1FSE": 0.3}
        reset_specific_keys(values, ["K1", "VKICK"])
        self.assertEqual(
            values,
            {"Q1K1": 0, "Q1HKICK": 0.1, "Q1VKICK": 0, "B1FSE": 0.3},
        )


class ParticleSourceTests(unittest.TestCase):
    def test_generated_beam_count_is_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            deck = Path(directory) / "track.ele"
            deck.write_text("&bunched_beam\n n_particles_per_bunch = 10\n&end\n")
            count, mode, beam_path = configure_initial_particles(
                str(deck), True, configured_count=321
            )
            self.assertEqual((count, mode, beam_path), (321, "generated bunched_beam", None))
            self.assertIn("n_particles_per_bunch = 321", deck.read_text())

    def test_source_mode_mismatch_fails_before_tracking(self):
        with tempfile.TemporaryDirectory() as directory:
            deck = Path(directory) / "track.ele"
            deck.write_text('&sdds_beam\n input = "beam.sdds"\n&end\n')
            with self.assertRaisesRegex(ValueError, "cannot be used with"):
                configure_initial_particles(str(deck), True, configured_count=10)


class CompatibilityGuardTests(unittest.TestCase):
    def test_replay_buffer_rejects_legacy_observation_width(self):
        transition = (
            np.zeros(58), np.zeros(4), 0.0, np.zeros(58), False,
            np.ones(4), np.ones(4),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "buffer.pkl"
            with path.open("wb") as stream:
                pickle.dump([transition], stream)
            with self.assertRaisesRegex(ValueError, "expected 59"):
                ReplayBuffer().load(path, expected_obs_dim=59)


class EnhancedWrapperTests(unittest.TestCase):
    def test_enhanced_wrapper_keeps_core_and_qdac_optics(self):
        self.assertTrue(issubclass(EnhancedElegantWrapper, CoreElegantWrapper))
        self.assertTrue(hasattr(EnhancedElegantWrapper, "compare_watch_to_target"))
        self.assertAlmostEqual(
            EnhancedElegantWrapper.momentum_mev_to_elegant_p(0.51099895), 1.0
        )


if __name__ == "__main__":
    unittest.main()
