import unittest
import sys
import os
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from processing.audio.converter import get_soundfont, midi_to_mp3, create_metronome_audio
from processing.midi.converter import create_metronome_midi


class TestAudioConverter(unittest.TestCase):
    @patch('processing.audio.converter.requests.get')
    def test_get_soundfont(self, mock_get):
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'audio/sf2'}
        mock_response.content = b'mock_soundfont_data'
        mock_get.return_value = mock_response
        
        # Test with non-existent soundfont (should download)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily override the soundfonts directory
            original_dir = os.path.join(os.getcwd(), 'soundfonts')
            temp_soundfonts = os.path.join(temp_dir, 'soundfonts')
            os.makedirs(temp_soundfonts, exist_ok=True)
            
            with patch('processing.audio.converter.SOUNDFONT_DIR', temp_soundfonts):
                sf_path = get_soundfont("Piano")
                self.assertIsNotNone(sf_path)
                self.assertTrue(os.path.exists(sf_path))
        
        # Test with invalid response
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        sf_path = get_soundfont("InvalidInstrument")
        self.assertIsNone(sf_path)

    @patch('processing.audio.converter.subprocess.run')
    @patch('processing.audio.converter.get_soundfont')
    def test_midi_to_mp3(self, mock_get_soundfont, mock_run):
        # Create a simple MIDI file for testing
        midi = create_metronome_midi(60, "4/4", 1)
        
        # Mock successful soundfont and conversion
        mock_get_soundfont.return_value = "/path/to/mock/soundfont.sf2"
        mock_run.return_value = MagicMock(returncode=0)
        
        # Test with mocked successful conversion
        with patch('processing.audio.converter.AudioSegment'):
            with patch('processing.audio.converter.shutil'):
                mp3_path, duration = midi_to_mp3(midi, "Piano", False)
                self.assertIsNotNone(mp3_path)
        
        # Test with forced fallback
        with patch('processing.audio.converter.generate_fallback_audio') as mock_fallback:
            mock_fallback.return_value = ("/path/to/fallback.mp3", 2.5)
            mp3_path, duration = midi_to_mp3(midi, "Piano", True)
            self.assertEqual(mp3_path, "/path/to/fallback.mp3")
            self.assertEqual(duration, 2.5)

    @patch('processing.audio.converter.AudioSegment')
    def test_create_metronome_audio(self, mock_audio_segment):
        # Mock the audio generation
        mock_audio_segment.silent.return_value = MagicMock()
        mock_audio_segment.from_wav.return_value = MagicMock()
        
        # Test metronome audio creation
        with patch('processing.audio.converter.create_metronome_midi'):
            with patch('processing.audio.converter.shutil'):
                mp3_path = create_metronome_audio(60, "4/4", 2)
                self.assertIsNotNone(mp3_path)


if __name__ == "__main__":
    unittest.main()