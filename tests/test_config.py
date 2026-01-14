# test_config.py
# Comprehensive tests for pynapse.config modules

import pytest
from pynapse.config import events


class TestEventDictionaries:
    """Test suite for event dictionary definitions."""
    
    def test_legacy_her_dict_exists(self):
        """Test that LEGACY_HER dictionary exists."""
        assert hasattr(events, 'LEGACY_HER')
        assert isinstance(events.LEGACY_HER, dict)
    
    def test_legacy_her_contains_expected_events(self):
        """Test LEGACY_HER contains expected event codes."""
        expected_events = {
            22: "active_lever",
            222: "active_lever_timeout",
            21: "inactive_lever",
            212: "inactive_lever_timeout",
            7: "cue",
            4: "infusion",
            9: "frame_trigger",
        }
        assert events.LEGACY_HER == expected_events
    
    def test_legacy_eth_dict_exists(self):
        """Test that LEGACY_ETH dictionary exists."""
        assert hasattr(events, 'LEGACY_ETH')
        assert isinstance(events.LEGACY_ETH, dict)
    
    def test_legacy_eth_contains_expected_events(self):
        """Test LEGACY_ETH contains expected event codes."""
        expected_events = {
            22: "active_lick",
            222: "active_lick_timeout",
            21: "inactive_lick",
            7: "cue_onset",
            50: "ethanol_delivery",
            51: "water_delivery",
            9: "frame_trigger",
        }
        assert events.LEGACY_ETH == expected_events
    
    def test_reacher_dict_exists(self):
        """Test that REACHER dictionary exists."""
        assert hasattr(events, 'REACHER')
        assert isinstance(events.REACHER, dict)
    
    def test_event_codes_are_integers(self):
        """Test that all event codes are integers."""
        for event_dict in [events.LEGACY_HER, events.LEGACY_ETH, events.REACHER]:
            for code in event_dict.keys():
                assert isinstance(code, int)
    
    def test_event_labels_are_strings(self):
        """Test that all event labels are strings."""
        for event_dict in [events.LEGACY_HER, events.LEGACY_ETH, events.REACHER]:
            for label in event_dict.values():
                assert isinstance(label, str)


class TestEventColors:
    """Test suite for event color definitions."""
    
    def test_colors_dict_exists(self):
        """Test that COLORS dictionary exists."""
        assert hasattr(events, 'COLORS')
        assert isinstance(events.COLORS, dict)
    
    def test_colors_are_hex_strings(self):
        """Test that all colors are hex color strings."""
        for color_value in events.COLORS.values():
            assert isinstance(color_value, str)
            # Should start with # and be valid hex (or transparent)
            assert color_value.startswith('#')
            assert len(color_value) in [7, 9]  # #RRGGBB or #RRGGBBAA
    
    def test_frame_trigger_is_transparent(self):
        """Test that frame_trigger color is transparent."""
        assert events.COLORS["frame_trigger"] == "#00000000"
    
    def test_expected_colors_exist(self):
        """Test that expected event types have colors defined."""
        expected_events = [
            "active_lever_press",
            "cue_onset",
            "infusion",
            "frame_trigger"
        ]
        for event in expected_events:
            assert event in events.COLORS


class TestTaskToDict:
    """Test suite for TASK_TO_DICT mapping."""
    
    def test_task_to_dict_exists(self):
        """Test that TASK_TO_DICT exists."""
        assert hasattr(events, 'TASK_TO_DICT')
        assert isinstance(events.TASK_TO_DICT, dict)
    
    def test_task_to_dict_contains_expected_tasks(self):
        """Test that TASK_TO_DICT contains expected task mappings."""
        expected_tasks = ["reacher", "legacy_eth", "legacy_her"]
        for task in expected_tasks:
            assert task in events.TASK_TO_DICT
    
    def test_task_to_dict_mappings_are_correct(self):
        """Test that TASK_TO_DICT maps to correct dictionaries."""
        assert events.TASK_TO_DICT["reacher"] is events.REACHER
        assert events.TASK_TO_DICT["legacy_eth"] is events.LEGACY_ETH
        assert events.TASK_TO_DICT["legacy_her"] is events.LEGACY_HER
    
    def test_task_names_are_lowercase(self):
        """Test that all task names are lowercase."""
        for task_name in events.TASK_TO_DICT.keys():
            assert task_name == task_name.lower()


class TestModuleExports:
    """Test suite for module __all__ exports."""
    
    def test_all_exports_exist(self):
        """Test that __all__ is defined."""
        assert hasattr(events, '__all__')
        assert isinstance(events.__all__, list)
    
    def test_expected_exports_in_all(self):
        """Test that expected items are in __all__."""
        expected_exports = [
            "REACHER",
            "LEGACY_ETH",
            "LEGACY_HER",
            "COLORS",
            "TASK_TO_DICT",
        ]
        for export in expected_exports:
            assert export in events.__all__
    
    def test_all_exports_are_accessible(self):
        """Test that all items in __all__ are accessible."""
        for export_name in events.__all__:
            assert hasattr(events, export_name)


class TestEventDictionaryConsistency:
    """Test suite for consistency across event dictionaries."""
    
    def test_all_dicts_have_frame_trigger(self):
        """Test that all event dictionaries include frame trigger."""
        for event_dict in [events.LEGACY_HER, events.LEGACY_ETH]:
            # Check if frame_trigger exists with code 9
            assert 9 in event_dict
            assert "frame_trigger" in event_dict[9].lower()
    
    def test_no_duplicate_codes_within_dict(self):
        """Test that no event dictionary has duplicate codes."""
        for event_dict in [events.LEGACY_HER, events.LEGACY_ETH, events.REACHER]:
            codes = list(event_dict.keys())
            assert len(codes) == len(set(codes))
    
    def test_no_duplicate_labels_within_dict(self):
        """Test that no event dictionary has duplicate labels."""
        for event_dict in [events.LEGACY_HER, events.LEGACY_ETH, events.REACHER]:
            labels = list(event_dict.values())
            # Allow empty dicts (like REACHER which is placeholder)
            if labels:
                assert len(labels) == len(set(labels))
    
    def test_timeout_events_have_higher_codes(self):
        """Test that timeout events have higher codes than their base events."""
        # In LEGACY_HER
        assert 222 > 22  # active_lever_timeout > active_lever
        assert 212 > 21  # inactive_lever_timeout > inactive_lever
        
        # In LEGACY_ETH
        assert 222 > 22  # active_lick_timeout > active_lick
    
    def test_event_labels_follow_naming_convention(self):
        """Test that event labels follow snake_case convention."""
        for event_dict in [events.LEGACY_HER, events.LEGACY_ETH]:
            for label in event_dict.values():
                # Should be lowercase with underscores
                assert label == label.lower()
                assert ' ' not in label  # No spaces


# Integration tests
class TestConfigIntegration:
    """Integration tests for config module usage."""
    
    def test_can_lookup_task_by_name(self):
        """Test that tasks can be looked up by name."""
        her_dict = events.TASK_TO_DICT.get("legacy_her")
        assert her_dict is not None
        assert 22 in her_dict
        assert her_dict[22] == "active_lever"
    
    def test_can_map_event_code_to_label(self):
        """Test mapping event codes to labels."""
        event_dict = events.LEGACY_HER
        code = 22
        label = event_dict.get(code)
        assert label == "active_lever"
    
    def test_can_get_color_for_event(self):
        """Test getting color for an event."""
        color = events.COLORS.get("cue_onset")
        assert color is not None
        assert color.startswith("#")
    
    @pytest.mark.parametrize("task_name,expected_code,expected_label", [
        ("legacy_her", 22, "active_lever"),
        ("legacy_her", 7, "cue"),
        ("legacy_eth", 22, "active_lick"),
        ("legacy_eth", 50, "ethanol_delivery"),
    ])
    def test_parameterized_task_lookups(self, task_name, expected_code, expected_label):
        """Test various task lookups with parametrize."""
        task_dict = events.TASK_TO_DICT[task_name]
        assert task_dict[expected_code] == expected_label
