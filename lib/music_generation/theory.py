#!/usr/bin/env python
"""
Music Theory Helpers
==================
Utility functions for music theory operations like note conversions.
"""

import re
from typing import Dict

# -----------------------------------------------------------------------------
# Music theory helpers (note names ↔︎ MIDI numbers)
# -----------------------------------------------------------------------------
NOTE_MAP: Dict[str, int] = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11,
}

INSTRUMENT_PROGRAMS: Dict[str, int] = {
    "Piano": 0, "Trumpet": 56, "Violin": 40,
    "Clarinet": 71, "Flute": 73,
}


def note_name_to_midi(note: str) -> int:
    """
    Convert a note name (e.g., 'C4', 'F#3') to MIDI note number.
    
    Args:
        note: String representation of a note (e.g., 'C4', 'F#3')
        
    Returns:
        MIDI note number
        
    Raises:
        ValueError: If the note format is invalid
    """
    match = re.match(r"([A-Ga-g][#b]?)(\d)", note)
    if not match:
        raise ValueError(f"Invalid note: {note}")
    pitch, octave = match.groups()
    pitch = pitch.upper().replace('b', 'B')
    if pitch not in NOTE_MAP:
        raise ValueError(f"Invalid pitch: {pitch}")
    return NOTE_MAP[pitch] + (int(octave) + 1) * 12


def midi_to_note_name(midi_num: int) -> str:
    """
    Convert a MIDI note number to note name (e.g., 'C4').
    
    Args:
        midi_num: MIDI note number
        
    Returns:
        Note name string (e.g., 'C4')
    """
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"


def clean_note_string(note_str: str) -> str:
    """
    Clean note strings by removing ornamentation symbols that cause parsing errors.
    
    Args:
        note_str: Note string that may contain ornamentation
        
    Returns:
        Cleaned note string
    """
    # Remove common ornamentation patterns
    patterns_to_remove = [
        r'\(grace\)', r'\(turn\)', r'\(mordent\)', r'\(trill\)',
        r'\(appoggiatura\)', r'\(double-grace\)', r'\(fermata\)'
    ]

    for pattern in patterns_to_remove:
        note_str = re.sub(pattern, '', note_str)

    # If multiple notes are present (like in ornaments), take the first one
    if '-' in note_str:
        note_str = note_str.split('-')[0]

    # Remove any remaining parentheses or special characters
    note_str = re.sub(r'[\(\)]', '', note_str).strip()

    return note_str