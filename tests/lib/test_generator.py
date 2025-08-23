import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lib.music_generation.generator import scale_json_durations, safe_parse_json, generate_exercise


class TestGenerator(unittest.TestCase):
    def test_scale_json_durations(self):
        # Test scaling with exact match
        data = [["C4", 2], ["D4", 2], ["E4", 4]]
        scaled = scale_json_durations(data, 8)
        self.assertEqual(sum(d for _, d in scaled), 8)
        
        # Test scaling with different target
        data = [["C4", 1], ["D4", 1], ["E4", 1]]
        scaled = scale_json_durations(data, 6)
        self.assertEqual(sum(d for _, d in scaled), 6)
        
        # Test empty data
        self.assertEqual(scale_json_durations([], 10), [])
        
        # Test zero durations
        data = [["C4", 0], ["D4", 0]]
        scaled = scale_json_durations(data, 4)
        self.assertEqual(sum(d for _, d in scaled), 4)

    def test_safe_parse_json(self):
        # Test valid JSON
        valid_json = '[{"note": "C4", "duration": 2}]'
        self.assertIsNotNone(safe_parse_json(valid_json))
        
        # Test JSON with single quotes
        single_quotes = "[['C4', 2], ['D4', 4]]"
        self.assertIsNotNone(safe_parse_json(single_quotes))
        
        # Test invalid JSON
        invalid_json = "This is not JSON"
        self.assertIsNone(safe_parse_json(invalid_json))
        
        # Test JSON embedded in text
        embedded_json = "Here's your exercise: [[\"C4\", 2], [\"D4\", 4]]"
        result = safe_parse_json(embedded_json)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    @patch('lib.music_generation.generator.query_mistral')
    def test_generate_exercise(self, mock_query):
        # Mock the API response
        mock_query.return_value = json.dumps([["C4", 2], ["D4", 2], ["E4", 4]])
        
        # Test successful generation
        result = generate_exercise("Piano", "Beginner", "C Major", "4/4", 2, "")
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)  # Three notes
        
        # Test with API error
        mock_query.side_effect = Exception("API Error")
        result = generate_exercise("Piano", "Beginner", "C Major", "4/4", 2, "")
        self.assertIsNotNone(result)  # Should return fallback exercise


if __name__ == "__main__":
    unittest.main()