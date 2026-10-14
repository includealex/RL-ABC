"""Occurrence-specific Elegant magnet action discovery and masking.

This module is intentionally separate from :mod:`Utils` and
:mod:`Environment`.  It consumes the already-expanded active-lattice graph and
uses only parameters that are explicitly present on each magnet occurrence.
No Elegant defaults are invented and no lattice values are changed here.

The fixed policy layout is ``[K1, HKICK, VKICK, FSE]``.  Two independent masks
are represented:

* the availability mask says which of those parameters exist on this magnet;
* the reset mask disables the configured ``reset_keys`` when
  ``reset_specific_keys_bool`` is true.

Their intersection is the mask used by the algorithm.  The registry also maps
that four-dimensional action into the exact indices of the environment's full
beamline-variable vector, without assuming that variables are contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ACTION_CHANNELS: Tuple[str, ...] = ("K1", "HKICK", "VKICK", "FSE")
ACTION_CHANNEL_INDEX: Dict[str, int] = {
    parameter: index for index, parameter in enumerate(ACTION_CHANNELS)
}
DEFAULT_RESET_KEYS: Tuple[str, ...] = ("HKICK", "VKICK", "FSE")
# Backward-compatible public name used by older callers.
RESET_DISABLED_CHANNELS = frozenset(DEFAULT_RESET_KEYS)

# Elegant aliases are included so this utility can also consume graphs made by
# older parsers that did not normalize element names.  Parameter availability
# is still determined only from the individual definition.
QUADRUPOLE_ELEMENT_TYPES = frozenset(("QUAD", "KQUAD"))
DIPOLE_ELEMENT_TYPES = frozenset(
    (
        "SBEN",
        "SBEND",
        "RBEN",
        "RBEND",
        "CSBEND",
        "CSRCSBEND",
        "CCBEND",
        "KSBEND",
        "NIBEND",
        "LGBEND",
        "TUBEND",
    )
)
MAGNET_ELEMENT_TYPES = QUADRUPOLE_ELEMENT_TYPES | DIPOLE_ELEMENT_TYPES

# Informational Elegant-manual support for the element families used by this
# project.  These sets are deliberately not used to add absent parameters.
MANUAL_CONTROL_PARAMETERS = {
    "QUAD": ACTION_CHANNELS,
    "KQUAD": ACTION_CHANNELS,
    "SBEN": ("K1", "FSE"),
    "SBEND": ("K1", "FSE"),
    "RBEN": ("K1", "FSE"),
    "RBEND": ("K1", "FSE"),
    "CSBEND": ("K1", "FSE"),
    "CSRCSBEND": ("K1", "FSE"),
    "CCBEND": ("K1", "FSE"),
    "KSBEND": ("K1", "FSE"),
    "NIBEND": ("K1", "FSE"),
    "LGBEND": ("K1", "FSE"),
    "TUBEND": ("K1", "FSE"),
}


class MagnetActionLayoutError(ValueError):
    """Raised when the parsed lattice and environment variables disagree."""


def normalize_parameter_name(name: object) -> str:
    return str(name).strip().upper()


def normalize_element_type(name: object) -> str:
    element_type = str(name).strip().upper()
    return {
        "SBEN": "SBEND",
        "RBEN": "RBEND",
    }.get(element_type, element_type)


def magnet_family(element_type: object) -> Optional[str]:
    normalized = normalize_element_type(element_type)
    if normalized in QUADRUPOLE_ELEMENT_TYPES:
        return "quadrupole"
    if normalized in DIPOLE_ELEMENT_TYPES:
        return "dipole"
    return None


def split_control_variable(variable: object) -> Tuple[str, Optional[str]]:
    """Split ``Q1HKICK`` into ``("Q1", "HKICK")`` using a suffix match."""
    text = str(variable)
    upper = text.upper()
    # Longest suffixes first makes the intent explicit and avoids substring
    # replacement bugs when an element name itself contains K1/FSE text.
    for parameter in ("HKICK", "VKICK", "FSE", "K1"):
        if upper.endswith(parameter):
            return text[: -len(parameter)], parameter
    return text, None


def normalize_reset_keys(reset_keys=None) -> Tuple[str, ...]:
    """Validate and canonicalize configurable policy channels to reset.

    ``None`` preserves the historical behavior of resetting HKICK, VKICK, and
    FSE. An empty sequence is valid and disables no channels.
    """
    if reset_keys is None:
        return DEFAULT_RESET_KEYS
    if isinstance(reset_keys, str):
        raise TypeError(
            "environment.reset_keys must be a list, not a string; for example "
            "['HKICK', 'VKICK', 'FSE']"
        )
    try:
        requested = {normalize_parameter_name(key) for key in reset_keys}
    except TypeError as exc:
        raise TypeError("environment.reset_keys must be a sequence") from exc
    invalid = requested.difference(ACTION_CHANNELS)
    if invalid:
        raise ValueError(
            "environment.reset_keys contains unsupported values: "
            f"{sorted(invalid)}. Allowed values are {list(ACTION_CHANNELS)}"
        )
    return tuple(key for key in ACTION_CHANNELS if key in requested)


def reset_policy_mask(
    reset_specific_keys_bool: bool,
    dtype=np.float32,
    reset_keys=None,
) -> np.ndarray:
    """Return the four-channel mask after applying configured reset keys."""
    mask = np.ones(len(ACTION_CHANNELS), dtype=dtype)
    if reset_specific_keys_bool:
        for parameter in normalize_reset_keys(reset_keys):
            mask[ACTION_CHANNEL_INDEX[parameter]] = 0.0
    return mask


@dataclass(frozen=True)
class MagnetActionSpec:
    """Controllable parameters and full-vector indices for one occurrence."""

    element_name: str
    element_type: str
    family: str
    parameters: Tuple[str, ...]
    variable_names: Tuple[str, ...]
    variable_indices: Tuple[int, ...]

    def availability_policy_mask(self, dtype=np.float32) -> np.ndarray:
        mask = np.zeros(len(ACTION_CHANNELS), dtype=dtype)
        for parameter in self.parameters:
            mask[ACTION_CHANNEL_INDEX[parameter]] = 1.0
        return mask

    def policy_mask(
        self,
        reset_specific_keys_bool: bool,
        dtype=np.float32,
        reset_keys=None,
    ) -> np.ndarray:
        return self.availability_policy_mask(dtype=dtype) * reset_policy_mask(
            reset_specific_keys_bool, dtype=dtype, reset_keys=reset_keys
        )

    def environment_mask(
        self,
        variable_count: int,
        reset_specific_keys_bool: bool = False,
        dtype=np.float32,
        reset_keys=None,
    ) -> np.ndarray:
        mask = np.zeros(variable_count, dtype=dtype)
        reset_mask = reset_policy_mask(
            reset_specific_keys_bool, dtype=dtype, reset_keys=reset_keys
        )
        for parameter, variable_index in zip(self.parameters, self.variable_indices):
            if reset_mask[ACTION_CHANNEL_INDEX[parameter]]:
                mask[variable_index] = 1.0
        return mask

    def policy_values(
        self,
        action: Sequence[float],
        reset_specific_keys_bool: bool = False,
        reset_keys=None,
    ) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.size != len(ACTION_CHANNELS):
            raise MagnetActionLayoutError(
                f"Expected {len(ACTION_CHANNELS)} policy values, got {action_array.size}"
            )
        effective = action_array * self.policy_mask(
            reset_specific_keys_bool,
            dtype=np.float64,
            reset_keys=reset_keys,
        )
        return np.asarray(
            [effective[ACTION_CHANNEL_INDEX[p]] for p in self.parameters],
            dtype=np.float64,
        )


class MagnetActionRegistry:
    """Single source of truth for lattice-side and algorithm-side masks."""

    def __init__(self, specs: Sequence[MagnetActionSpec], variable_names: Sequence[str]):
        self.specs = tuple(specs)
        self.variable_names = tuple(str(name) for name in variable_names)
        self._by_name = {spec.element_name.casefold(): spec for spec in self.specs}
        if len(self._by_name) != len(self.specs):
            raise MagnetActionLayoutError(
                "Controllable occurrence names must be unique after lattice preprocessing"
            )

    @classmethod
    def from_graph(
        cls,
        graph: Sequence[Mapping[str, object]],
        variable_names: Sequence[str],
        chronological_order: Optional[Sequence[str]] = None,
    ) -> "MagnetActionRegistry":
        variables = tuple(str(name) for name in variable_names)
        variable_lookup: Dict[Tuple[str, str], Tuple[str, int]] = {}
        for index, variable in enumerate(variables):
            element_name, parameter = split_control_variable(variable)
            if parameter is None:
                raise MagnetActionLayoutError(
                    f"Unknown environment control variable {variable!r}"
                )
            key = (element_name.casefold(), parameter)
            if key in variable_lookup:
                raise MagnetActionLayoutError(f"Duplicate control variable {variable!r}")
            variable_lookup[key] = (variable, index)

        controllable_nodes: Dict[str, Mapping[str, object]] = {}
        graph_order = []
        for node in graph:
            element_name = str(node.get("name", ""))
            family = magnet_family(node.get("type", ""))
            if not element_name or family is None:
                continue
            attributes = node.get("attributes", {})
            if not isinstance(attributes, Mapping):
                attributes = {}
            attribute_names = {normalize_parameter_name(key) for key in attributes}
            parameters = tuple(
                parameter for parameter in ACTION_CHANNELS
                if parameter in attribute_names
            )
            if not parameters:
                continue
            key = element_name.casefold()
            if key in controllable_nodes:
                raise MagnetActionLayoutError(
                    f"Controllable occurrence {element_name!r} appears more than once; "
                    "occurrence qualification must run before mask construction"
                )
            controllable_nodes[key] = node
            graph_order.append(element_name)

        requested_order = tuple(chronological_order or graph_order)
        if len({name.casefold() for name in requested_order}) != len(requested_order):
            raise MagnetActionLayoutError(
                "Chronological controllable-element order contains duplicate names"
            )

        specs = []
        consumed_variables = set()
        for element_name in requested_order:
            node = controllable_nodes.get(str(element_name).casefold())
            if node is None:
                raise MagnetActionLayoutError(
                    f"No controllable magnet definition found for {element_name!r}"
                )
            attributes = node.get("attributes", {})
            attribute_names = {normalize_parameter_name(key) for key in attributes}
            parameters = tuple(
                parameter for parameter in ACTION_CHANNELS
                if parameter in attribute_names
            )
            names = []
            indices = []
            for parameter in parameters:
                variable_key = (str(element_name).casefold(), parameter)
                if variable_key not in variable_lookup:
                    raise MagnetActionLayoutError(
                        f"{element_name} explicitly defines {parameter}, but its "
                        "environment variable is missing"
                    )
                variable_name, variable_index = variable_lookup[variable_key]
                names.append(variable_name)
                indices.append(variable_index)
                consumed_variables.add(variable_key)
            specs.append(
                MagnetActionSpec(
                    element_name=str(element_name),
                    element_type=normalize_element_type(node.get("type", "")),
                    family=str(magnet_family(node.get("type", ""))),
                    parameters=parameters,
                    variable_names=tuple(names),
                    variable_indices=tuple(indices),
                )
            )

        unused = set(variable_lookup) - consumed_variables
        if unused:
            unused_names = [variable_lookup[key][0] for key in sorted(unused)]
            raise MagnetActionLayoutError(
                "Environment variables are not attached to an active controllable "
                f"magnet occurrence: {unused_names}"
            )
        return cls(specs, variables)

    def __len__(self) -> int:
        return len(self.specs)

    def spec_for_iteration(self, iteration: int, offset: int = 0) -> Optional[MagnetActionSpec]:
        index = int(iteration) + int(offset)
        if not 0 <= index < len(self.specs):
            return None
        return self.specs[index]

    def spec_for_element(self, element_name: str) -> MagnetActionSpec:
        try:
            return self._by_name[str(element_name).casefold()]
        except KeyError as exc:
            raise MagnetActionLayoutError(
                f"Unknown controllable magnet occurrence {element_name!r}"
            ) from exc

    def parameters_by_element(self) -> Dict[str, Tuple[str, ...]]:
        return {spec.element_name: spec.parameters for spec in self.specs}

    def inventory(self, reset_specific_keys_bool: bool = False, reset_keys=None):
        """Return JSON-friendly rows describing every active controllable magnet."""
        rows = []
        configured_reset_keys = normalize_reset_keys(reset_keys)
        reset_mask = reset_policy_mask(
            reset_specific_keys_bool, reset_keys=configured_reset_keys
        ).astype(int).tolist()
        for spec in self.specs:
            rows.append(
                {
                    "element_name": spec.element_name,
                    "element_type": spec.element_type,
                    "magnet_family": spec.family,
                    "explicit_parameters": list(spec.parameters),
                    "variable_names": list(spec.variable_names),
                    "variable_indices": list(spec.variable_indices),
                    "availability_policy_mask": spec.availability_policy_mask().astype(int).tolist(),
                    "reset_policy_mask": reset_mask,
                    "configured_reset_keys": list(configured_reset_keys),
                    "effective_policy_mask": spec.policy_mask(
                        reset_specific_keys_bool,
                        reset_keys=configured_reset_keys,
                    ).astype(int).tolist(),
                }
            )
        return rows

    def policy_mask(
        self,
        iteration: int,
        reset_specific_keys_bool: bool,
        offset: int = 0,
        dtype=np.float32,
        reset_keys=None,
    ) -> np.ndarray:
        spec = self.spec_for_iteration(iteration, offset)
        if spec is None:
            return np.zeros(len(ACTION_CHANNELS), dtype=dtype)
        return spec.policy_mask(
            reset_specific_keys_bool, dtype=dtype, reset_keys=reset_keys
        )

    def availability_policy_mask(
        self, iteration: int, offset: int = 0, dtype=np.float32
    ) -> np.ndarray:
        spec = self.spec_for_iteration(iteration, offset)
        if spec is None:
            return np.zeros(len(ACTION_CHANNELS), dtype=dtype)
        return spec.availability_policy_mask(dtype=dtype)

    def environment_mask(
        self,
        iteration: int,
        reset_specific_keys_bool: bool,
        offset: int = 0,
        dtype=np.float32,
        reset_keys=None,
    ) -> np.ndarray:
        spec = self.spec_for_iteration(iteration, offset)
        if spec is None:
            return np.zeros(len(self.variable_names), dtype=dtype)
        return spec.environment_mask(
            len(self.variable_names),
            reset_specific_keys_bool=reset_specific_keys_bool,
            dtype=dtype,
            reset_keys=reset_keys,
        )

    def cumulative_environment_mask(
        self,
        through_iteration: int,
        reset_specific_keys_bool: bool,
        dtype=np.float32,
        reset_keys=None,
    ) -> np.ndarray:
        mask = np.zeros(len(self.variable_names), dtype=dtype)
        last = min(int(through_iteration), len(self.specs) - 1)
        for index in range(last + 1):
            mask = np.maximum(
                mask,
                self.environment_mask(
                    index,
                    reset_specific_keys_bool=reset_specific_keys_bool,
                    dtype=dtype,
                    reset_keys=reset_keys,
                ),
            )
        return mask

    def policy_values(
        self,
        iteration: int,
        action: Sequence[float],
        reset_specific_keys_bool: bool,
        offset: int = 0,
        reset_keys=None,
    ) -> np.ndarray:
        spec = self.spec_for_iteration(iteration, offset)
        if spec is None:
            return np.asarray([], dtype=np.float64)
        return spec.policy_values(
            action, reset_specific_keys_bool, reset_keys=reset_keys
        )

    def compose_environment_values(
        self,
        previous_values: Sequence[float],
        iteration: int,
        policy_action: Sequence[float],
        reset_specific_keys_bool: bool,
        reset_keys=None,
    ) -> np.ndarray:
        """Preserve prior occurrences and update only the current occurrence."""
        previous = np.asarray(previous_values, dtype=np.float64).reshape(-1)
        if previous.size != len(self.variable_names):
            raise MagnetActionLayoutError(
                f"Expected {len(self.variable_names)} environment values, got {previous.size}"
            )
        spec = self.spec_for_iteration(iteration)
        if spec is None:
            raise MagnetActionLayoutError(f"No magnet at iteration {iteration}")

        result = np.zeros(len(self.variable_names), dtype=np.float64)
        if iteration > 0:
            prior_mask = self.cumulative_environment_mask(
                iteration - 1,
                reset_specific_keys_bool=reset_specific_keys_bool,
                dtype=np.float64,
                reset_keys=reset_keys,
            )
            result[prior_mask.astype(bool)] = previous[prior_mask.astype(bool)]

        values = spec.policy_values(
            policy_action,
            reset_specific_keys_bool,
            reset_keys=reset_keys,
        )
        for variable_index, value in zip(spec.variable_indices, values):
            result[variable_index] = value
        return result


def build_magnet_action_registry(
    graph: Sequence[Mapping[str, object]],
    variable_names: Sequence[str],
    chronological_order: Optional[Sequence[str]] = None,
) -> MagnetActionRegistry:
    return MagnetActionRegistry.from_graph(
        graph,
        variable_names,
        chronological_order=chronological_order,
    )


__all__ = [
    "ACTION_CHANNELS",
    "ACTION_CHANNEL_INDEX",
    "RESET_DISABLED_CHANNELS",
    "DEFAULT_RESET_KEYS",
    "QUADRUPOLE_ELEMENT_TYPES",
    "DIPOLE_ELEMENT_TYPES",
    "MAGNET_ELEMENT_TYPES",
    "MANUAL_CONTROL_PARAMETERS",
    "MagnetActionLayoutError",
    "MagnetActionSpec",
    "MagnetActionRegistry",
    "build_magnet_action_registry",
    "magnet_family",
    "normalize_reset_keys",
    "reset_policy_mask",
    "split_control_variable",
]
