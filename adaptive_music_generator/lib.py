"""Core music generation logic for the Adaptive Music Generator.

This module contains the core functionality for generating music based on
user parameters, including note processing, MIDI generation, and LLM integration.
"""

import os
import re
import json
import random
import time
import requests
from typing import Dict, List, Tuple, Optional, Any, Union
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

from .exceptions import MusicGenerationError, InvalidParameterError

# Constants
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX")

TICKS_PER_BEAT = 480  # Standard MIDI resolution
TICKS_PER_8TH = TICKS_PER_BEAT // 2  # 240 ticks per 8th note

# Music theory helpers
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
    """Convert a note name (e.g., 'C4') to MIDI note number."""
    match = re.match(r"([A-Ga-g][#b]?)(\d)", note)
    if not match:
        raise InvalidParameterError(f"Invalid note: {note}")
    pitch, octave = match.groups()
    pitch = pitch.upper().replace('b', 'B')
    return NOTE_MAP.get(pitch, 0) + (int(octave) + 1) * 12


def midi_to_note_name(midi_num: int) -> str:
    """Convert a MIDI note number to note name (e.g., 'C4')."""
    octave = (midi_num // 12) - 1
    note_idx = midi_num % 12
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{note_names[note_idx]}{octave}"


def parse_time_signature(time_sig: str) -> Tuple[int, int]:
    """Parse time signature string (e.g., '4/4') into numerator and denominator."""
    try:
        numerator, denominator = map(int, time_sig.split('/'))
        return numerator, denominator
    except (ValueError, AttributeError):
        raise InvalidParameterError(f"Invalid time signature: {time_sig}")


def generate_exercise_prompt(params: Dict[str, Any]) -> str:
    """Generate a prompt for the LLM to create a music exercise."""
    instrument = params.get("instrument", "Piano")
    level = params.get("level", "Beginner")
    key = params.get("key", "C Major")
    time_signature = params.get("time_signature", "4/4")
    measures = params.get("measures", 4)
    focus = params.get("focus", "Melodic")
    
    prompt = f"""Generate a musical exercise for {instrument} at {level} level.
    Key: {key}
    Time Signature: {time_signature}
    Measures: {measures}
    Focus: {focus}
    
    Provide the exercise as a JSON object with the following structure:
    {{
        "notes": [{{
            "pitch": "C4",  // Note name with octave
            "duration": 1,   // Duration in eighth notes
            "velocity": 80,  // MIDI velocity (0-127)
            "start_time": 0  // Start time in eighth notes
        }}, ...],
        "tempo": 120,       // BPM
        "key": "{key}",
        "time_signature": "{time_signature}",
        "instrument": "{instrument}"
    }}
    
    Ensure all notes fit within the specified time signature and measures.
    Use appropriate rhythmic patterns for {level} level.
    Focus on {focus.lower()} aspects in the exercise.
    """
    
    return prompt


def query_llm(prompt: str) -> Dict[str, Any]:
    """Query the LLM API with the given prompt and return the response."""
    # API key is now set with a default value, so this check is just for extra safety
    api_key = MISTRAL_API_KEY
    if not api_key:
        raise MusicGenerationError("MISTRAL_API_KEY environment variable not set or default key is invalid")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "system", "content": "You are a music theory expert that generates musical exercises in JSON format."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    # Implement retry logic with exponential backoff
    max_retries = 3
    retry_delay = 2  # Initial delay in seconds
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt} after {retry_delay}s delay...")
                time.sleep(retry_delay)
                # Exponential backoff with jitter
                retry_delay = min(retry_delay * 2, 30) + random.uniform(0, 1)
                
            response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
            
            # Handle rate limiting (429 errors)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", retry_delay))
                print(f"Rate limited. Waiting for {retry_after}s before retrying...")
                time.sleep(retry_after)
                continue
                
            response.raise_for_status()
            # If we get here, the request was successful
            break
                
        except requests.RequestException as e:
            last_exception = e
            print(f"Request failed: {str(e)}")
            if attempt == max_retries - 1:
                # This was our last attempt
                raise MusicGenerationError(f"Error connecting to LLM API after {max_retries} attempts: {str(e)}")
    
    # If we've exhausted all retries without breaking out of the loop
    if last_exception is not None:
        raise MusicGenerationError(f"Error connecting to LLM API after {max_retries} attempts: {str(last_exception)}")
    
    # Process the successful response
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        # If we can't parse the response, print debugging info and raise error
        print(f"\nAPI Response Status: {response.status_code}")
        print(f"API Response Headers: {response.headers}")
        print(f"API Response Text (first 500 chars): {response.text[:500]}")
        raise MusicGenerationError(f"Invalid API response format: {str(e)}")
    
    # Extract JSON from the response
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    if not content:
        raise MusicGenerationError("Empty response from LLM API")
        
    print(f"\nLLM Response (first 200 chars): {content[:200]}...")
    
    # Try to find JSON in the content
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        raise MusicGenerationError("No valid JSON found in LLM response")
    
    # Try to parse the extracted JSON    
    try:
        exercise_data = json.loads(json_match.group(0))
        return exercise_data
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to clean up the content
        json_str = json_match.group(0)
        print(f"\nFailed to parse JSON (first 200 chars): {json_str[:200]}...")
        
        # Try to fix common JSON issues
        print("Cleaning and fixing JSON...")
        
        # First, remove all comments
        # Remove full-line comments
        json_str = re.sub(r'^\s*//.*?$', '', json_str, flags=re.MULTILINE)
        # Remove inline comments (careful to handle quoted strings properly)
        json_str = re.sub(r'(?<!["\\\'])//.*?(?=\n|$)', '', json_str)
        # Remove multi-line comments
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Fix other common issues
        # Replace single quotes with double quotes
        json_str = json_str.replace("'", '"')
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
        # Fix missing commas between objects in arrays
        json_str = re.sub(r'(\})(\s*\{)', r'\1,\2', json_str)
        
        # Print the cleaned JSON for debugging
        print(f"Cleaned JSON (first 200 chars): {json_str[:200]}...")
        
        # Final cleanup - remove any remaining non-JSON syntax
        # Remove any remaining comments or annotations
        json_str = re.sub(r'\s*//.*?(?:\n|$)', '', json_str)
        
        try:
            exercise_data = json.loads(json_str)
            return exercise_data
        except json.JSONDecodeError:
            raise MusicGenerationError(f"Failed to parse LLM response as JSON: {str(e)}")


def create_midi_file(exercise_data: Dict[str, Any]) -> MidiFile:
    """Create a MIDI file from the exercise data."""
    midi_file = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    midi_file.tracks.append(track)
    
    # Add tempo
    tempo = exercise_data.get("tempo", 120)
    microseconds_per_beat = int(60_000_000 / tempo)
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    
    # Add time signature
    time_sig = exercise_data.get("time_signature", "4/4")
    numerator, denominator = parse_time_signature(time_sig)
    track.append(MetaMessage('time_signature', numerator=numerator, denominator=denominator, time=0))
    
    # Add instrument program change
    instrument = exercise_data.get("instrument", "Piano")
    program = INSTRUMENT_PROGRAMS.get(instrument, 0)
    track.append(Message('program_change', program=program, time=0))
    
    # Process notes
    notes = exercise_data.get("notes", [])
    if not notes:
        raise MusicGenerationError("No notes found in exercise data")
    
    # Sort notes by start time
    notes.sort(key=lambda x: x.get("start_time", 0))
    
    current_time = 0
    for note in notes:
        pitch = note.get("pitch")
        if not pitch:
            continue
            
        midi_note = note_name_to_midi(pitch)
        velocity = note.get("velocity", 80)
        start_time = note.get("start_time", 0) * TICKS_PER_8TH
        duration = note.get("duration", 1) * TICKS_PER_8TH
        
        # Calculate delta time
        delta_time = start_time - current_time
        
        # Add note_on event
        track.append(Message('note_on', note=midi_note, velocity=velocity, time=delta_time))
        
        # Update current time
        current_time = start_time
        
        # Add note_off event
        track.append(Message('note_off', note=midi_note, velocity=0, time=duration))
        
        # Update current time
        current_time += duration
    
    return midi_file


def generate_music(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a music exercise based on the given parameters.
    
    Args:
        params: Dictionary containing exercise parameters
            - instrument: Instrument name (Piano, Trumpet, etc.)
            - level: Difficulty level (Beginner, Intermediate, Advanced)
            - key: Musical key (e.g., "C Major")
            - time_signature: Time signature (e.g., "4/4")
            - measures: Number of measures
            - focus: Practice focus (Rhythmic, Melodic, etc.)
    
    Returns:
        Dictionary containing the generated exercise data and MIDI file
    """
    # Generate prompt for LLM
    prompt = generate_exercise_prompt(params)
    
    # Query LLM
    exercise_data = query_llm(prompt)
    
    # Create MIDI file
    midi_file = create_midi_file(exercise_data)
    
    return {
        "exercise_data": exercise_data,
        "midi_file": midi_file
    }


def generate_metronome(tempo: int, time_signature: str, measures: int) -> MidiFile:
    """Generate a metronome track with the given parameters.
    
    Args:
        tempo: Tempo in BPM
        time_signature: Time signature (e.g., "4/4")
        measures: Number of measures
    
    Returns:
        MidiFile object containing the metronome track
    """
    midi_file = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    midi_file.tracks.append(track)
    
    # Add tempo
    microseconds_per_beat = int(60_000_000 / tempo)
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    
    # Add time signature
    numerator, denominator = parse_time_signature(time_signature)
    track.append(MetaMessage('time_signature', numerator=numerator, denominator=denominator, time=0))
    
    # Add metronome clicks
    beats_per_measure = numerator
    total_beats = beats_per_measure * measures
    
    for beat in range(total_beats):
        # Strong beat on the first beat of each measure
        if beat % beats_per_measure == 0:
            velocity = 100  # Strong beat
            note = 76  # High wood block
        else:
            velocity = 70  # Weak beat
            note = 77  # Low wood block
        
        # Add note_on event
        track.append(Message('note_on', note=note, velocity=velocity, time=0 if beat == 0 else TICKS_PER_BEAT))
        
        # Add note_off event
        track.append(Message('note_off', note=note, velocity=0, time=10))
    
    return midi_file