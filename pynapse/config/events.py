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
# REACHER self-administration (2026+ paradigm)
#
# Codes are assigned by device category:
#   100s — Levers       200s — Drug delivery    300s — Cues
#   400s — Stimulation  500s — Licking          600s — Pavlovian trials
#   700s — Session control (typically excluded from analysis)
#   900s — Reserved for auto-assigned unknown events
#
# Labels follow the REACHER CSV convention: "{device}_{event}" lowercased.
# =============================================================================
REACHER: Dict[int, str] = {
    # Right-hand lever
    101: "rh_lever_active_press",
    102: "rh_lever_timeout_press",
    103: "rh_lever_inactive_press",
    # Left-hand lever
    111: "lh_lever_active_press",
    112: "lh_lever_timeout_press",
    113: "lh_lever_inactive_press",
    # Drug delivery
    201: "pump_infusion",
    211: "pump_2_infusion",
    # Cue
    301: "cue_tone",
    311: "cue_2_tone",
    # Stimulation
    401: "laser_stim",
    # Licking
    501: "lick_lick",
    # Pavlovian trials
    601: "pavlov_trial_start",
    602: "pavlov_reward_delivered",
    603: "pavlov_reward_omitted",
    604: "pavlov_all_trials_complete",
    # Session control
    701: "controller_start",
    702: "controller_end",
}

# Reverse lookup: CSV composite label → integer code (used by EventLog CSV loader)
_REACHER_LABEL_TO_CODE: Dict[str, int] = {v: k for k, v in REACHER.items()}

# =============================================================================
# Standard colors for plotting (consistent across all figures)
# =============================================================================
COLORS = {
    # Legacy labels
    "active_lever_press": "#d32f2f",    # red-700
    "active_lever_timeout": "#b71c1c",  # red-900
    "inactive_lever_press": "#9e9e9e",  # grey-500
    "cue_onset": "#1976d2",             # blue-700
    "infusion": "#388e3c",              # green-700
    "ethanol_delivery": "#7b1fa2",      # purple-700
    "water_delivery": "#0288d1",        # light blue-700
    "frame_trigger": "#00000000",       # transparent — never plot
    # REACHER labels
    "rh_lever_active_press": "#d32f2f",     # red-700
    "rh_lever_timeout_press": "#b71c1c",    # red-900
    "rh_lever_inactive_press": "#9e9e9e",   # grey-500
    "lh_lever_active_press": "#ef5350",     # red-400
    "lh_lever_timeout_press": "#c62828",    # red-800
    "lh_lever_inactive_press": "#bdbdbd",   # grey-400
    "pump_infusion": "#388e3c",             # green-700
    "pump_2_infusion": "#4caf50",           # green-500
    "cue_tone": "#1976d2",                  # blue-700
    "cue_2_tone": "#42a5f5",               # blue-400
    "laser_stim": "#7b1fa2",               # purple-700
    "lick_lick": "#ff9800",                # orange-500
    "pavlov_trial_start": "#00000000",      # transparent
    "pavlov_reward_delivered": "#388e3c",   # green-700
    "pavlov_reward_omitted": "#f44336",     # red-500
    "pavlov_all_trials_complete": "#00000000",  # transparent
    "controller_start": "#00000000",        # transparent
    "controller_end": "#00000000",          # transparent
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
    "_REACHER_LABEL_TO_CODE",
    "LEGACY_ETH",
    "LEGACY_HER",
    "COLORS",
    "TASK_TO_DICT",
]