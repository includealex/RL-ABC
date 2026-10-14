"""Particle-distribution optics extension for :mod:`rl_framework.Elegant`.

This module deliberately leaves ``Elegant.py`` unchanged.  Its ``ElegantWrapper``
subclasses the existing wrapper, so all existing tracking, plotting, and lattice
methods remain available.  The added methods analyze coordinate-mode WATCH files
with the SDDS ``sddsanalyzebeam`` program and compare the resulting beam ellipse
with configurable target optics.

Typical use::

    from rl_framework.Elegant_edited import ElegantWrapper

    wrapper = ElegantWrapper(...)
    result = wrapper.compare_watch_to_target(
        target={"betax": 4.90185, "alphax": -0.101508,
                "betay": 1.20299, "alphay": -0.52017},
        watch_point="final_WP",
        corrected=True,
        scales={"betax": 0.1, "alphax": 0.02,
                "betay": 0.1, "alphay": 0.02},
    )

``result["score"]`` is a bounded scalar suitable for a reward component and
``result["descriptor_vector"]`` is the elementwise normalized error vector that
QDAC can use as a descriptor.  A scale is a matching tolerance in the same units
as its quantity, not merely a numerical normalization constant.
"""

from __future__ import annotations

from io import StringIO
import math
import os
import shutil
import subprocess
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .Elegant import ElegantWrapper as _OriginalElegantWrapper
from .Utils import (
    add_final_watch_point,
    add_watch_points,
    change_initial_content,
    parse_lattice_file,
    remove_watch_points,
    validate_with_elegant,
)


Number = Union[int, float, np.number]


class ElegantWrapper(_OriginalElegantWrapper):
    """The original wrapper plus particle-derived optical matching methods."""

    _VALIDATED_LATTICES = set()

    def __init__(self, *args, validate_generated_lattice=True, **kwargs):
        elegant_path = kwargs.get("elegant_path", "")
        rpn_defns = kwargs.get("sddsPath", "") or None
        super().__init__(*args, **kwargs)
        if validate_generated_lattice:
            executable = os.path.join(str(elegant_path), "elegant")
            source_key = (
                os.path.abspath(self.input_file_path),
                os.path.getmtime(self.input_file_path),
                self.beamline_name,
                executable,
            )
            if source_key not in self._VALIDATED_LATTICES:
                validate_with_elegant(
                    self.output_file,
                    beamline_name=self.beamline_name,
                    elegant_executable=executable,
                    rpn_defns=rpn_defns,
                    p_central_mev=400.0,
                )
                self._VALIDATED_LATTICES.add(source_key)

    # These defaults are deliberately physical tolerances.  For production
    # optimization, pass tolerances agreed with the accelerator physicist.
    DEFAULT_OPTICS_SCALES = {
        "betax": 1.0,
        "betay": 1.0,
        "alphax": 1.0,
        "alphay": 1.0,
        "etax": 0.1,
        "etay": 0.1,
        "etaxp": 0.1,
        "etayp": 0.1,
        "ex": 1e-6,
        "ey": 1e-6,
        "enx": 1e-6,
        "eny": 1e-6,
        "Sx": 1e-3,
        "Sxp": 1e-3,
        "Sy": 1e-3,
        "Syp": 1e-3,
        "Sdelta": 1e-2,
        "pAverage": 1.0,
        "Cx": 1e-3,
        "Cxp": 1e-3,
        "Cy": 1e-3,
        "Cyp": 1e-3,
    }

    _TARGET_ALIASES = {
        "beta_x": "betax",
        "beta_y": "betay",
        "alpha_x": "alphax",
        "alpha_y": "alphay",
        "eta_x": "etax",
        "eta_y": "etay",
        "eta_xp": "etaxp",
        "eta_yp": "etayp",
        "etap_x": "etaxp",
        "etap_y": "etayp",
        "sigma_x": "Sx",
        "sigma_xp": "Sxp",
        "sigma_y": "Sy",
        "sigma_yp": "Syp",
        "sigma_delta": "Sdelta",
    }

    _CORRECTED_COLUMNS = {
        "ex": "ecx",
        "enx": "ecnx",
        "betax": "betacx",
        "alphax": "alphacx",
        "ey": "ecy",
        "eny": "ecny",
        "betay": "betacy",
        "alphay": "alphacy",
    }

    _ANALYSIS_COLUMNS = (
        "ElementName",
        "ex", "enx", "betax", "alphax", "etax", "etaxp",
        "ey", "eny", "betay", "alphay", "etay", "etayp",
        "Sx", "Sxp", "Sy", "Syp", "St", "Sdelta", "pAverage",
        "el", "ecx", "ecnx", "betacx", "alphacx",
        "ecy", "ecny", "betacy", "alphacy",
        "Cx", "Cxp", "Cy", "Cyp", "Ct", "Cdelta",
        "s11", "s12", "s22", "s13", "s23", "s33",
        "s14", "s24", "s34", "s44", "s15", "s25",
        "s35", "s45", "s55", "s16", "s26", "s36",
        "s46", "s56", "s66",
    )

    def _preprocess_lattice_file(self):
        """Preprocess a seed-local copy without mutating the source lattice."""
        work_lattice = f"machine_work_{self.seed}.lte"
        modified_lattice = f"modifide_machine_no_finalWP_{self.seed}.lte"
        template_lattice = f"temp_modified_machine_{self.seed}.lte"
        backup_lattice = f"unchanged_input_{self.seed}.lte"

        shutil.copy2(self.input_file_path, work_lattice)
        remove_watch_points(work_lattice, backup_path=backup_lattice)
        add_watch_points(
            work_lattice,
            modified_lattice,
            results_path=self.results_path,
            beamline_name=self.beamline_name,
        )
        add_final_watch_point(
            modified_lattice,
            self.output_file,
            self.beamline_name,
            results_path=self.results_path,
        )
        parsed = parse_lattice_file(self.output_file)
        watch_points = [
            name for name, value in parsed.items()
            if value["type"] == "watch point"
        ]
        formatted_lattice, variables = change_initial_content(
            self.output_file,
            template_lattice,
            beamline_name=self.beamline_name,
        )
        return watch_points, formatted_lattice, variables

    def _get_chroneological_order_elements(self):
        """Record only elements that actually expose an agent parameter."""
        from .Utils import process_lte_file_to_graph

        graph = process_lte_file_to_graph(self.output_file, self.beamline_name)
        for node in graph:
            if str(node.get("type", "")).upper() == "WATCH":
                self.chronolgical_order_watch_points.append(node["name"])
            if node.get("controllable") or node.get("controllable_parameters"):
                self.chroneological_order_controllable_vars.append(node["name"])
        return graph

    def _sdds_tool(self, name: str) -> str:
        """Return the executable path using the wrapper's configured SDDS path."""
        return os.path.join(str(self.elegantPath), name)

    def _result_file(self, filename: str) -> str:
        """Resolve a simple filename relative to ``results_path``."""
        if os.path.isabs(filename) or os.path.dirname(filename):
            return filename
        return os.path.join(self.results_path, filename)

    @staticmethod
    def momentum_mev_to_elegant_p(momentum_mev_c: Number) -> float:
        """Convert positron/electron momentum in MeV/c to Elegant's beta*gamma p."""
        return float(momentum_mev_c) / 0.51099895

    @staticmethod
    def elegant_p_to_momentum_mev(momentum_beta_gamma: Number) -> float:
        """Convert Elegant's beta*gamma momentum coordinate to MeV/c."""
        return float(momentum_beta_gamma) * 0.51099895

    def _query_sdds_columns(self, filename: str) -> Sequence[str]:
        command = [self._sdds_tool("sddsquery"), filename, "-columnlist"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sddsquery failed for {filename}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def read_beam_analysis(self, analysis_file: str) -> pd.DataFrame:
        """Read the useful columns of an ``sddsanalyzebeam`` output file."""
        analysis_file = self._result_file(analysis_file)
        available = set(self._query_sdds_columns(analysis_file))
        columns = [name for name in self._ANALYSIS_COLUMNS if name in available]
        if not columns:
            raise RuntimeError(f"No beam-analysis columns found in {analysis_file}")

        command = [
            self._sdds_tool("sdds2stream"),
            analysis_file,
            "-col=" + ",".join(columns),
            "-pipe=out",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"sdds2stream failed for {analysis_file}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        frame = pd.read_csv(
            StringIO(result.stdout), names=columns, sep=r"\s+", engine="python"
        )
        if frame.empty:
            raise RuntimeError(f"Beam analysis contains no rows: {analysis_file}")
        for column in columns:
            if column != "ElementName":
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def analyze_watch_particles(
        self,
        watch_point: str = "final_WP",
        *,
        watch_file: Optional[str] = None,
        output_file: Optional[str] = None,
        momentum_range: Optional[Tuple[Number, Number]] = None,
        corrected_only: bool = False,
        canonical: bool = False,
        overwrite: bool = True,
    ) -> pd.DataFrame:
        """Run ``sddsanalyzebeam`` on a coordinate-mode WATCH file.

        ``momentum_range`` uses Elegant's dimensionless beta*gamma ``p`` values,
        exactly as in the supplied dat2sdds example.  Use
        :meth:`momentum_mev_to_elegant_p` when the desired cuts are in MeV/c.
        The unfiltered and dispersion-corrected columns are both retained unless
        ``corrected_only`` is requested.
        """
        watch_file = self._result_file(watch_file or f"{watch_point}.sdds")
        output_file = self._result_file(
            output_file or f"{watch_point}_beam_analyzed.sdds"
        )
        if not os.path.isfile(watch_file):
            raise FileNotFoundError(f"WATCH coordinate file not found: {watch_file}")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        if os.path.exists(output_file):
            if not overwrite:
                return self.read_beam_analysis(output_file)
            os.remove(output_file)

        analyze_command = [self._sdds_tool("sddsanalyzebeam")]
        if corrected_only:
            analyze_command.append("-correctedOnly")
        if canonical:
            analyze_command.append("-canonical")

        if momentum_range is None:
            analyze_command.extend([watch_file, output_file])
            analyzed = subprocess.run(analyze_command, capture_output=True, text=True)
            if analyzed.returncode != 0:
                raise RuntimeError(
                    "sddsanalyzebeam failed: "
                    + (
                        analyzed.stderr.strip()
                        or analyzed.stdout.strip()
                        or f"process exited with code {analyzed.returncode}"
                    )
                )
        else:
            lower, upper = map(float, momentum_range)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError("momentum_range must contain finite (lower, upper) values")
            filter_command = [
                self._sdds_tool("sddsprocess"),
                watch_file,
                "-pipe=out",
                f"-filter=column,p,{lower:.17g},{upper:.17g}",
            ]
            analyze_command.extend(["-pipe=in", output_file])
            filtering = subprocess.Popen(
                filter_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            analyzed = subprocess.run(
                analyze_command,
                stdin=filtering.stdout,
                capture_output=True,
                text=True,
            )
            if filtering.stdout is not None:
                filtering.stdout.close()
            filter_stderr = filtering.stderr.read().decode(errors="replace")
            filter_code = filtering.wait()
            if filter_code != 0:
                raise RuntimeError(f"sddsprocess momentum filter failed: {filter_stderr.strip()}")
            if analyzed.returncode != 0:
                raise RuntimeError(
                    "sddsanalyzebeam failed: "
                    + (
                        analyzed.stderr.strip()
                        or analyzed.stdout.strip()
                        or f"process exited with code {analyzed.returncode}"
                    )
                )

        return self.read_beam_analysis(output_file)

    def get_particle_beam_optics(self, *args, row: int = -1, **kwargs) -> dict:
        """Return one analyzed WATCH page as a plain dictionary."""
        frame = self.analyze_watch_particles(*args, **kwargs)
        if not -len(frame) <= row < len(frame):
            raise IndexError(f"row {row} is outside an analysis with {len(frame)} rows")
        return frame.iloc[row].to_dict()

    @classmethod
    def _normalize_optics_name(cls, name: str) -> str:
        stripped = str(name).strip()
        return cls._TARGET_ALIASES.get(stripped.lower(), stripped)

    def compare_beam_optics(
        self,
        measured: Mapping[str, Number],
        target: Mapping[str, Number],
        *,
        corrected: bool = True,
        scales: Optional[Mapping[str, Number]] = None,
        weights: Optional[Mapping[str, Number]] = None,
    ) -> dict:
        """Compare particle-derived optics with targets.

        For target names such as ``betax``, ``corrected=True`` selects ``betacx``
        when available.  Dispersion targets always use ``etax``, ``etaxp``,
        ``etay``, and ``etayp``.  The returned score is ``1/sqrt(1+loss)`` and
        lies in ``(0, 1]``.  The
        descriptor vector preserves target insertion order.
        """
        if not target:
            raise ValueError("target must contain at least one optical quantity")
        normalized_scales = {
            self._normalize_optics_name(key): float(value)
            for key, value in (scales or {}).items()
        }
        normalized_weights = {
            self._normalize_optics_name(key): float(value)
            for key, value in (weights or {}).items()
        }

        rows = []
        seen = set()
        for requested_name, target_value_raw in target.items():
            name = self._normalize_optics_name(requested_name)
            if name in seen:
                raise ValueError(f"Duplicate target after alias normalization: {name}")
            seen.add(name)
            source = self._CORRECTED_COLUMNS.get(name, name) if corrected else name
            if source not in measured and source != name and name in measured:
                source = name
            if source not in measured:
                raise KeyError(
                    f"Measured beam analysis has no '{source}' column for target '{name}'"
                )
            measured_value = float(measured[source])
            target_value = float(target_value_raw)
            scale_name = name
            default_scale = self.DEFAULT_OPTICS_SCALES.get(scale_name)
            scale = normalized_scales.get(scale_name, default_scale)
            if scale is None:
                raise KeyError(f"No scale supplied for target quantity '{name}'")
            weight = normalized_weights.get(name, 1.0)
            if not all(map(math.isfinite, (measured_value, target_value, scale, weight))):
                raise ValueError(f"Non-finite matching input for '{name}'")
            if scale <= 0:
                raise ValueError(f"Scale for '{name}' must be positive")
            if weight < 0:
                raise ValueError(f"Weight for '{name}' cannot be negative")
            residual = measured_value - target_value
            normalized_error = residual / scale
            rows.append(
                {
                    "quantity": name,
                    "measured_column": source,
                    "measured": measured_value,
                    "target": target_value,
                    "scale": scale,
                    "weight": weight,
                    "residual": residual,
                    "normalized_error": normalized_error,
                    "weighted_squared_error": weight * normalized_error**2,
                }
            )

        comparison = pd.DataFrame(rows).set_index("quantity", drop=False)
        weight_sum = float(comparison["weight"].sum())
        if weight_sum <= 0:
            raise ValueError("At least one target weight must be positive")
        loss = float(comparison["weighted_squared_error"].sum() / weight_sum)
        score = 1.0 / math.sqrt(1.0 + loss)
        return {
            "loss": loss,
            "score": score,
            "descriptor_names": comparison["quantity"].tolist(),
            "descriptor_vector": comparison["normalized_error"].to_numpy(dtype=float),
            "measured_vector": comparison["measured"].to_numpy(dtype=float),
            "target_vector": comparison["target"].to_numpy(dtype=float),
            "comparison": comparison,
            "corrected": corrected,
        }

    def compare_watch_to_target(
        self,
        target: Mapping[str, Number],
        *,
        watch_point: str = "final_WP",
        corrected: bool = True,
        scales: Optional[Mapping[str, Number]] = None,
        weights: Optional[Mapping[str, Number]] = None,
        **analysis_kwargs,
    ) -> dict:
        """Analyze a WATCH file and return scalar and elementwise match metrics."""
        measured = self.get_particle_beam_optics(
            watch_point=watch_point, **analysis_kwargs
        )
        result = self.compare_beam_optics(
            measured, target, corrected=corrected, scales=scales, weights=weights
        )
        result["watch_point"] = watch_point
        result["measured_optics"] = measured
        return result

    def evaluate_watch_match(
        self,
        target: Mapping[str, Number],
        *,
        initial_particles: Optional[int] = None,
        transmitted_particles: Optional[int] = None,
        transmission_weight: float = 1.0,
        optics_weight: float = 1.0,
        **kwargs,
    ) -> dict:
        """Return an RL-ready combined transmission and optical-match result.

        This method does not alter the environment's existing reward.  It exposes
        ``combined_reward`` so RLABC/QDAC callers can opt in explicitly.
        """
        result = self.compare_watch_to_target(target, **kwargs)
        transmitted = int(
            self.get_num_particles()
            if transmitted_particles is None
            else transmitted_particles
        )
        if initial_particles is None:
            transmission = None
            combined = result["score"]
        else:
            if initial_particles <= 0:
                raise ValueError("initial_particles must be positive")
            transmission = min(max(transmitted / float(initial_particles), 0.0), 1.0)
            total_weight = float(transmission_weight) + float(optics_weight)
            if transmission_weight < 0 or optics_weight < 0 or total_weight <= 0:
                raise ValueError("Reward weights must be nonnegative and not both zero")
            combined = (
                transmission_weight * transmission + optics_weight * result["score"]
            ) / total_weight
        result.update(
            {
                "transmitted_particles": transmitted,
                "initial_particles": initial_particles,
                "transmission_fraction": transmission,
                "combined_reward": float(combined),
            }
        )
        return result


# An explicit name is convenient in notebooks, while ElegantWrapper makes this
# module a one-line drop-in replacement for imports of the original wrapper.
ElegantEditedWrapper = ElegantWrapper


__all__ = ["ElegantWrapper", "ElegantEditedWrapper"]
