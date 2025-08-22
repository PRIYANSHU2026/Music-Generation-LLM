"""Tests for the lib module."""

import unittest
from unittest.mock import patch, MagicMock
import json
import os

from adaptive_music_generator.lib import (
    note_name_to_midi,
    midi_to_note_name,
    parse_time_signature,
    generate_exercise_prompt,
    create_midi_file,
    generate_metronome
)
from adaptive_music_generator.exceptions import InvalidParameterError


class TestMusicTheoryHelpers(unittest.TestCase):
    """Test music theory helper functions."""
    
    def test_note_name_to_midi(self):
        """Test conversion from note names to MIDI numbers."""
        self.assertEqual(note_name_to_midi("C4"), 60)
        self.assertEqual(note_name_to_midi("A4"), 69)
        self.assertEqual(note_name_to_midi("G#3"), 56)
        self.assertEqual(note_name_to_midi("Eb5"), 75)
        
        # Test invalid note
        with self.assertRaises(InvalidParameterError):
            note_name_to_midi("H4")
    
    def test_midi_to_note_name(self):
        """Test conversion from MIDI numbers to note names."""
        self.assertEqual(midi_to_note_name(60), "C4")
        self.assertEqual(midi_to_note_name(69), "A4")
        self.assertEqual(midi_to_note_name(56), "G#3")
        self.assertEqual(midi_to_note_name(75), "D#5")
    
    def test_parse_time_signature(self):
        """Test parsing time signatures."""
        self.assertEqual(parse_time_signature("4/4"), (4, 4))
        self.assertEqual(parse_time_signature("3/4"), (3, 4))
        self.assertEqual(parse_time_signature("6/8"), (6, 8))
        
        # Test invalid time signature
        with self.assertRaises(InvalidParameterError):
            parse_time_signature("invalid")


class TestPromptGeneration(unittest.TestCase):
    """Test prompt generation for LLM."""
    
    def test_generate_exercise_prompt(self):
        """Test generating exercise prompts."""
        params = {
            "instrument": "Piano",
            "level": "Beginner",
            "key": "C Major",
            "time_signature": "4/4",
            "measures": 4,
            "focus": "Melodic"
        }
        
        prompt = generate_exercise_prompt(params)
        
        # Check that all parameters are included in the prompt
        self.assertIn("Piano", prompt)
        self.assertIn("Beginner", prompt)
        self.assertIn("C Major", prompt)
        self.assertIn("4/4", prompt)
        self.assertIn("Melodic", prompt)


class TestMIDIGeneration(unittest.TestCase):
    """Test MIDI file generation."""
    
    def test_create_midi_file(self):
        """Test creating a MIDI file from exercise data."""
        exercise_data = {
            "notes": [
                {"pitch": "C4", "duration": 1, "velocity": 80, "start_time": 0},
                {"pitch": "E4", "duration": 1, "velocity": 80, "start_time": 1},
                {"pitch": "G4", "duration": 1, "velocity": 80, "start_time": 2},
                {"pitch": "C5", "duration": 1, "velocity": 80, "start_time": 3}
            ],
            "tempo": 120,
            "key": "C Major",
            "time_signature": "4/4",
            "instrument": "Piano"
        }
        
        midi_file = create_midi_file(exercise_data)
        
        # Check that the MIDI file has the correct structure
        self.assertEqual(len(midi_file.tracks), 1)
        
        # Count note events (note_on and note_off)
        note_on_events = [msg for msg in midi_file.tracks[0] if msg.type == "note_on" and msg.velocity > 0]
        note_off_events = [msg for msg in midi_file.tracks[0] if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)]
        
        self.assertEqual(len(note_on_events), 4)  # 4 notes
        self.assertEqual(len(note_off_events), 4)  # 4 note offs
    
    def test_generate_metronome(self):
        """Test generating a metronome track."""
        midi_file = generate_metronome(120, "4/4", 2)
        
        # Check that the MIDI file has the correct structure
        self.assertEqual(len(midi_file.tracks), 1)
        
        # Count note events (note_on and note_off)
        note_on_events = [msg for msg in midi_file.tracks[0] if msg.type == "note_on" and msg.velocity > 0]
        
        # 4/4 time signature, 2 measures = 8 beats
        self.assertEqual(len(note_on_events), 8)


if __name__ == "__main__":
    unittest.main()