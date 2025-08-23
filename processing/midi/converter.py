#!/usr/bin/env python

"""
MIDI Converter
=============
Functions for converting between JSON and MIDI formats.
"""

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import random
from typing import List, Any, Tuple

from lib.music_generation.constants import TICKS_PER_BEAT, TICKS_PER_8TH, INSTRUMENT_PROGRAMS
from lib.music_generation.theory import note_name_to_midi


def json_to_midi(json_data: List[Any], instrument: str, tempo: int, 
                 time_signature: str, measures: int) -> MidiFile:
    """
    Convert JSON note data to a MIDI file.
    
    Args:
        json_data: List of objects with 'note', 'duration', and 'cumulative_duration' properties,
                  or legacy format of [note, duration] pairs
        instrument: Instrument name
        tempo: Tempo in BPM
        time_signature: Time signature (e.g., "4/4")
        measures: Number of measures
        
    Returns:
        MidiFile object
        
    Raises:
        ValueError: If note parsing fails
    """
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)
    program = INSTRUMENT_PROGRAMS.get(instrument, 56)  # Default to trumpet if not found
    numerator, denominator = map(int, time_signature.split('/'))

    track.append(MetaMessage('time_signature', numerator=numerator,
                             denominator=denominator, time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    track.append(Message('program_change', program=program, time=0))

    for note_item in json_data:
        try:
            # Handle both object format and legacy array format
            if isinstance(note_item, dict):
                note_name = note_item['note']
                duration_units = note_item['duration']
            else:
                # Legacy format [note, duration]
                note_name, duration_units = note_item
                
            note_num = note_name_to_midi(note_name)
            ticks = int(duration_units * TICKS_PER_8TH)  # Convert 8th note units to ticks
            ticks = max(ticks, 1)  # Ensure at least 1 tick
            velocity = random.randint(60, 100)  # Random velocity for more natural sound
            track.append(Message('note_on', note=note_num, velocity=velocity, time=0))
            track.append(Message('note_off', note=note_num, velocity=velocity, time=ticks))
        except Exception as e:
            print(f"Error parsing note {note_name if 'note_name' in locals() else 'unknown'}: {e}")
            # Continue with next note instead of failing completely
    return mid


def create_metronome_midi(tempo: int, time_sig: str, measures: int) -> MidiFile:
    """
    Create a MIDI file with metronome clicks.
    
    Args:
        tempo: Tempo in BPM
        time_sig: Time signature (e.g., "4/4")
        measures: Number of measures
        
    Returns:
        MidiFile object with metronome clicks
    """
    numerator, denominator = map(int, time_sig.split('/'))
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)

    # Add time signature and tempo
    track.append(MetaMessage('time_signature', numerator=numerator,
                             denominator=denominator, time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(int(tempo)), time=0))

    # Calculate total beats
    beats_per_measure = numerator
    total_beats = beats_per_measure * measures

    # Add metronome clicks (strong beat = note 77, weak beat = note 76)
    for beat in range(total_beats):
        # Strong beat on first beat of measure, weak beat otherwise
        note_num = 77 if beat % beats_per_measure == 0 else 76
        velocity = 100 if beat % beats_per_measure == 0 else 80

        # Add note on (with time=0 for first beat)
        if beat == 0:
            track.append(Message('note_on', note=note_num, velocity=velocity, time=0))
        else:
            # Each beat is a quarter note (TICKS_PER_BEAT)
            track.append(Message('note_on', note=note_num, velocity=velocity, time=TICKS_PER_BEAT))

        # Short duration for click
        track.append(Message('note_off', note=note_num, velocity=0, time=10))
        
    return mid