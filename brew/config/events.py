# events.py
# Joshua Boquiren (@thejoshbq)
# boquiren@musc.edu
"""
Laboratory-standard event code → label mappings and visualization settings.

This is the **only place** event dictionaries are defined. All analysis code,
plotting functions, database schemas, and the local LLM must import from here
to guarantee consistency across the entire lab.
"""

from __future__ import annotations
from typing import Dict

# =============================================================================
# Heroin head-fixed self-administration (old 2024 paradigm)
# =============================================================================
LEGACY_HER: Dict[int, str] = {
    22: "active_lever",
    222: "active_lever_timeout",
    21: "inactive_lever",
    212: "inactive_lever_timeout",
    7: "cue",
    4: "infusion",
    9: "frame_trigger",
}

# =============================================================================
# Ethanol head-fixed self-administration (new 2025 paradigm)
# =============================================================================
LEGACY_ETH: Dict[int, str] = {
    22: "active_lick",
    222: "active_lick_timeout",
    21: "inactive_lick",
    7: "cue_onset",
    50: "ethanol_delivery",
    51: "water_delivery",
    9: "frame_trigger",
}

# =============================================================================
# Future tasks (add as they come online)
# =============================================================================
REACHER: Dict[int, str] = {
    # ... will be filled when the task exists ...
}

# =============================================================================
# Standard colors for plotting (consistent across all figures)
# =============================================================================
COLORS = {
    "active_lever_press": "#d32f2f",    # red-700
    "active_lever_timeout": "#b71c1c",  # red-900
    "inactive_lever_press": "#9e9e9e",  # grey-500
    "cue_onset": "#1976d2",             # blue-700
    "infusion": "#388e3c",              # green-700
    "ethanol_delivery": "#7b1fa2",      # purple-700
    "water_delivery": "#0288d1",        # light blue-700
    "frame_trigger": "#00000000",       # transparent — never plot
}

# =============================================================================
# Convenience: task → dict lookup
# =============================================================================
TASK_TO_DICT = {
    "reacher": REACHER,
    "legacy_eth": LEGACY_ETH,
    "legacy_her": LEGACY_HER,
}

__all__ = [
    "REACHER",
    "LEGACY_ETH",
    "LEGACY_HER",
    "COLORS",
    "TASK_TO_DICT",
]