import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from processing.visualization.visualizer import create_visualization


class TestVisualizer(unittest.TestCase):
    @patch('processing.visualization.visualizer.plt')
    @patch('processing.visualization.visualizer.note_name_to_midi')
    def test_create_visualization(self, mock_note_to_midi, mock_plt):
        # Mock the matplotlib functions
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.Rectangle.return_value = MagicMock()
        
        # Mock the note conversion
        mock_note_to_midi.side_effect = lambda note: {
            "C4": 60, "D4": 62, "E4": 64
        }.get(note, 60)
        
        # Test with valid JSON data
        json_data = json.dumps([["C4", 2], ["D4", 2], ["E4", 4]])
        with patch('processing.visualization.visualizer.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "test_uuid"
            viz_path = create_visualization(json_data, "4/4")
            self.assertIsNotNone(viz_path)
            self.assertTrue("visualization_test_uuid.png" in viz_path)
        
        # Test with invalid JSON
        viz_path = create_visualization("Invalid JSON", "4/4")
        self.assertIsNone(viz_path)
        
        # Test with empty JSON array
        viz_path = create_visualization("[]", "4/4")
        self.assertIsNone(viz_path)

    @patch('processing.visualization.visualizer.plt')
    def test_visualization_error_handling(self, mock_plt):
        # Test with matplotlib import error
        mock_plt.subplots.side_effect = ImportError("No module named 'matplotlib'")
        viz_path = create_visualization(json.dumps([["C4", 2], ["D4", 2]]), "4/4")
        self.assertIsNone(viz_path)
        
        # Test with value error (invalid note)
        mock_plt.subplots.side_effect = None
        with patch('processing.visualization.visualizer.note_name_to_midi') as mock_note_to_midi:
            mock_note_to_midi.side_effect = ValueError("Invalid note")
            viz_path = create_visualization(json.dumps([["X9", 2], ["Y9", 2]]), "4/4")
            self.assertIsNone(viz_path)


if __name__ == "__main__":
    unittest.main()