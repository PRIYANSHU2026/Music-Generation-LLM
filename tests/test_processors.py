"""Tests for the processors module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile

from adaptive_music_generator.processors import (
    ensure_directories,
    download_soundfont,
    midi_to_audio,
    generate_sheet_music,
    generate_vexflow_html
)
from adaptive_music_generator.exceptions import AudioConversionError, VisualizationError


class TestDirectoryManagement(unittest.TestCase):
    """Test directory management functions."""
    
    @patch('os.makedirs')
    def test_ensure_directories(self, mock_makedirs):
        """Test ensuring directories exist."""
        ensure_directories()
        
        # Check that makedirs was called for each directory
        self.assertEqual(mock_makedirs.call_count, 3)
        mock_makedirs.assert_any_call("static", exist_ok=True)
        mock_makedirs.assert_any_call("temp_audio", exist_ok=True)
        mock_makedirs.assert_any_call("soundfonts", exist_ok=True)


class TestSoundfontManagement(unittest.TestCase):
    """Test soundfont management functions."""
    
    @patch('os.path.exists')
    @patch('requests.get')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('adaptive_music_generator.processors.ensure_directories')
    def test_download_soundfont_existing(self, mock_ensure_dirs, mock_open, mock_get, mock_exists):
        """Test downloading a soundfont that already exists."""
        # Mock that the file already exists
        mock_exists.return_value = True
        
        result = download_soundfont("Piano")
        
        # Check that the function returns the correct path
        self.assertIn("soundfonts", result)
        self.assertIn(".sf2", result)
        
        # Check that requests.get was not called
        mock_get.assert_not_called()
    
    @patch('os.path.exists')
    @patch('requests.get')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('adaptive_music_generator.processors.ensure_directories')
    def test_download_soundfont_new(self, mock_ensure_dirs, mock_open, mock_get, mock_exists):
        """Test downloading a new soundfont."""
        # Mock that the file doesn't exist, then gets created
        mock_exists.side_effect = [False, True]
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b"mock soundfont data"
        mock_get.return_value = mock_response
        
        result = download_soundfont("Trumpet")
        
        # Check that the function returns the correct path
        self.assertIn("soundfonts", result)
        self.assertIn(".sf2", result)
        
        # Check that requests.get was called
        mock_get.assert_called_once()
        mock_open.assert_called_once()


class TestAudioConversion(unittest.TestCase):
    """Test audio conversion functions."""
    
    @patch('adaptive_music_generator.processors.download_soundfont')
    @patch('midi2audio.FluidSynth')
    @patch('adaptive_music_generator.processors.ensure_directories')
    def test_midi_to_audio_mp3(self, mock_ensure_dirs, mock_fluidsynth, mock_download):
        """Test converting MIDI to MP3."""
        # Mock the FluidSynth instance
        mock_fs_instance = MagicMock()
        mock_fluidsynth.return_value = mock_fs_instance
        
        # Mock the soundfont path
        mock_download.return_value = "/path/to/soundfont.sf2"
        
        # Create a temporary MIDI file
        with tempfile.NamedTemporaryFile(suffix=".mid") as temp_midi:
            # Create a temporary output path
            output_path = temp_midi.name.replace(".mid", ".mp3")
            
            # Call the function
            result = midi_to_audio(temp_midi.name, output_path, "Piano", "mp3")
            
            # Check that the function returns the correct path
            self.assertEqual(result, output_path)
            
            # Check that FluidSynth was called correctly
            mock_fluidsynth.assert_called_once_with("/path/to/soundfont.sf2")
            mock_fs_instance.midi_to_audio.assert_called_once_with(temp_midi.name, output_path)


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""
    
    def test_generate_vexflow_html(self):
        """Test generating VexFlow HTML."""
        notes = [
            {"pitch": "C4", "duration": 1, "velocity": 80, "start_time": 0},
            {"pitch": "E4", "duration": 1, "velocity": 80, "start_time": 1},
            {"pitch": "G4", "duration": 1, "velocity": 80, "start_time": 2},
            {"pitch": "C5", "duration": 1, "velocity": 80, "start_time": 3}
        ]
        
        html = generate_vexflow_html(notes, "4/4", "C Major")
        
        # Check that the HTML contains the expected elements
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("vexflow", html)
        self.assertIn("c4", html.lower())  # Note C4
        self.assertIn("e4", html.lower())  # Note E4
        self.assertIn("g4", html.lower())  # Note G4
        self.assertIn("c5", html.lower())  # Note C5
    
    def test_generate_vexflow_html_invalid_time_signature(self):
        """Test generating VexFlow HTML with invalid time signature."""
        notes = [{"pitch": "C4", "duration": 1, "velocity": 80, "start_time": 0}]
        
        with self.assertRaises(VisualizationError):
            generate_vexflow_html(notes, "invalid", "C Major")
    
    def test_generate_vexflow_html_invalid_key(self):
        """Test generating VexFlow HTML with invalid key."""
        notes = [{"pitch": "C4", "duration": 1, "velocity": 80, "start_time": 0}]
        
        with self.assertRaises(VisualizationError):
            generate_vexflow_html(notes, "4/4", "invalid")
    
    @patch('adaptive_music_generator.processors.generate_vexflow_html')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('adaptive_music_generator.processors.ensure_directories')
    def test_generate_sheet_music(self, mock_ensure_dirs, mock_open, mock_generate_html):
        """Test generating sheet music."""
        # Mock the HTML generation
        mock_generate_html.return_value = "<html>Mock sheet music</html>"
        
        exercise_data = {
            "notes": [
                {"pitch": "C4", "duration": 1, "velocity": 80, "start_time": 0},
                {"pitch": "E4", "duration": 1, "velocity": 80, "start_time": 1}
            ],
            "tempo": 120,
            "key": "C Major",
            "time_signature": "4/4",
            "instrument": "Piano"
        }
        
        output_path = "test_sheet_music.html"
        
        result = generate_sheet_music(exercise_data, output_path)
        
        # Check that the function returns the correct HTML
        self.assertEqual(result, "<html>Mock sheet music</html>")
        
        # Check that the HTML was written to the file
        mock_open.assert_called_once_with(output_path, "w")
        mock_open().write.assert_called_once_with("<html>Mock sheet music</html>")


if __name__ == "__main__":
    unittest.main()