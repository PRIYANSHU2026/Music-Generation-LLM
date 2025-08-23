import unittest
import sys
import os
import tempfile

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from processing.midi.converter import json_to_midi, create_metronome_midi


class TestMidiConverter(unittest.TestCase):
    def test_json_to_midi(self):
        # Test basic MIDI generation
        json_data = [["C4", 2], ["D4", 2], ["E4", 4]]
        midi = json_to_midi(json_data, "Piano", 60, "4/4", 1)
        
        # Check that we got a valid MIDI file with at least one track
        self.assertIsNotNone(midi)
        self.assertGreaterEqual(len(midi.tracks), 1)
        
        # Check that the track has note events
        has_notes = False
        for msg in midi.tracks[0]:
            if msg.type == 'note_on' or msg.type == 'note_off':
                has_notes = True
                break
        self.assertTrue(has_notes)
        
        # Test with invalid note (should handle gracefully)
        json_data = [["C4", 2], ["INVALID", 2], ["E4", 4]]
        midi = json_to_midi(json_data, "Piano", 60, "4/4", 1)
        self.assertIsNotNone(midi)  # Should still produce a MIDI file

    def test_create_metronome_midi(self):
        # Test metronome MIDI generation
        midi = create_metronome_midi(60, "4/4", 2)
        
        # Check that we got a valid MIDI file
        self.assertIsNotNone(midi)
        
        # Check that it has the right number of beats
        # 2 measures of 4/4 = 8 beats (note on/off pairs)
        note_count = 0
        for msg in midi.tracks[0]:
            if msg.type == 'note_on':
                note_count += 1
        self.assertEqual(note_count, 8)
        
        # Test with 3/4 time signature
        midi = create_metronome_midi(60, "3/4", 2)
        note_count = 0
        for msg in midi.tracks[0]:
            if msg.type == 'note_on':
                note_count += 1
        self.assertEqual(note_count, 6)  # 2 measures of 3/4 = 6 beats


if __name__ == "__main__":
    unittest.main()