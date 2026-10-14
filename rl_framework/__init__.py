"""RL framework for Elegant accelerator simulations.

Public classes are loaded lazily so utility-only tooling does not need to
initialize Gymnasium, Torch, HoloViews, or an accelerator runtime.
"""

from importlib import import_module

__all__ = ["ACCElegantEnvironment", "DDPGAgent", "setLogger"]


def __getattr__(name):
    if name == "ACCElegantEnvironment":
        return import_module(".Environment", __name__).ACCElegantEnvironment
    if name == "DDPGAgent":
        return import_module(".Agents.DDPG", __name__).DDPGAgent
    if name == "setLogger":
        return import_module(".Utils", __name__).setLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
