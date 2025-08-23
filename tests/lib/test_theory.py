import unittest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lib.music_generation.theory import note_name_to_midi, midi_to_note_name, clean_note_string


class TestTheory(unittest.TestCase):
    def test_note_name_to_midi(self):
        self.assertEqual(note_name_to_midi("C4"), 60)
        self.assertEqual(note_name_to_midi("A4"), 69)
        self.assertEqual(note_name_to_midi("G#3"), 56)
        self.assertEqual(note_name_to_midi("Bb5"), 82)
        
        # Test error cases
        with self.assertRaises(ValueError):
            note_name_to_midi("H4")  # Invalid note name
        with self.assertRaises(ValueError):
            note_name_to_midi("C")   # Missing octave

    def test_midi_to_note_name(self):
        self.assertEqual(midi_to_note_name(60), "C4")
        self.assertEqual(midi_to_note_name(69), "A4")
        self.assertEqual(midi_to_note_name(56), "G#3")
        self.assertEqual(midi_to_note_name(82), "A#5")

    def test_clean_note_string(self):
        self.assertEqual(clean_note_string("C4(trill)"), "C4")
        self.assertEqual(clean_note_string("D5(grace)"), "D5")
        self.assertEqual(clean_note_string("G3-A3"), "G3")
        self.assertEqual(clean_note_string("F#4(mordent)"), "F#4")


if __name__ == "__main__":
    unittest.main()