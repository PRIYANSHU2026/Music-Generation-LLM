"""
Adaptive Music Exercise Generator Library
=========================================
Core functionality for generating custom musical exercises with LLM.
This library contains all the business logic extracted from app.py.
"""

# -----------------------------------------------------------------------------
# 1. Imports
# -----------------------------------------------------------------------------
import sys
import subprocess
from typing import Dict, Optional, Tuple, List
import random
import requests
import json
import tempfile
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
import re
from io import BytesIO
from midi2audio import FluidSynth
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile
import os
import subprocess as sp
import base64
import shutil
import ast
import uuid
from datetime import datetime
import time

# -----------------------------------------------------------------------------
# 2. Configuration & constants (UPDATED TO USE 8TH NOTES)
# -----------------------------------------------------------------------------
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX"  # ← Replace with your key!

# Reliable direct download links for soundfonts
SOUNDFONT_URLS = {
    "Trumpet": "https://www.philscomputerlab.com/uploads/3/7/2/3/37231621/weedspatches-gm.sf2",  # General MIDI soundfont
    "Piano": "https://www.philscomputerlab.com/uploads/3/7/2/3/37231621/weedspatches-gm.sf2",  # General MIDI soundfont
    "Violin": "https://www.philscomputerlab.com/uploads/3/7/2/3/37231621/weedspatches-gm.sf2",  # General MIDI soundfont
    "Clarinet": "https://www.philscomputerlab.com/uploads/3/7/2/3/37231621/weedspatches-gm.sf2",  # General MIDI soundfont
    "Flute": "https://www.philscomputerlab.com/uploads/3/7/2/3/37231621/weedspatches-gm.sf2",  # General MIDI soundfont
}

SAMPLE_RATE = 44100  # Hz
TICKS_PER_BEAT = 480  # Standard MIDI resolution
TICKS_PER_8TH = TICKS_PER_BEAT // 2  # 240 ticks per 8th note (UPDATED)

# -----------------------------------------------------------------------------
# 3. Helper functions
# -----------------------------------------------------------------------------
def install(packages: List[str]):
    """Install required packages if they are not already installed."""
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def ensure_fluidsynth():
    """Ensure FluidSynth is installed on the system."""
    if not os.path.exists('/usr/bin/fluidsynth') and not os.path.exists('/usr/local/bin/fluidsynth') and not os.path.exists('/opt/homebrew/bin/fluidsynth'):
        try:
            if sys.platform == 'linux':
                os.system('apt-get update && apt-get install -y fluidsynth')
            elif sys.platform == 'darwin':
                # Check if brew is installed
                result = subprocess.run(['which', 'brew'], capture_output=True, text=True)
                if result.stdout.strip():
                    os.system('brew install fluid-synth')
                else:
                    print("Homebrew not found. Please install FluidSynth manually.")
            else:
                print("Please install FluidSynth manually for your platform.")
        except Exception as e:
            print(f"Could not install FluidSynth automatically: {e}")
            print("Please install it manually.")

def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs("static", exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)

# -----------------------------------------------------------------------------
# 4. Music theory helpers (note names ↔︎ MIDI numbers) - ENHANCED REST HANDLING
# -----------------------------------------------------------------------------
NOTE_MAP: Dict[str, int] = {
    "C": 0, "C#": 1, "DB": 1,
    "D": 2, "D#": 3, "EB": 3,
    "E": 4, "F": 5, "F#": 6, "GB": 6,
    "G": 7, "G#": 8, "AB": 8,
    "A": 9, "A#": 10, "BB": 10,
    "B": 11,
}

REST_INDICATORS = ["rest", "r", "Rest", "R", "P", "p", "pause"]

INSTRUMENT_PROGRAMS: Dict[str, int] = {
    "Piano": 0,       "Trumpet": 56,   "Violin": 40,
    "Clarinet": 71,   "Flute": 73,
}

def is_rest(note: str) -> bool:
    """Check if a note string represents a rest."""
    return note.strip().lower() in [r.lower() for r in REST_INDICATORS]

def note_name_to_midi(note: str) -> int:
    """Convert a note name to MIDI note number."""
    if is_rest(note):
        return -1  # Special value for rests
    
    # Allow both scientific (C4) and Helmholtz (C') notation
    match = re.match(r"([A-Ga-g][#b]?)(\\'*)(\d?)", note)
    if not match:
        raise ValueError(f"Invalid note: {note}")
    
    pitch, apostrophes, octave = match.groups()
    pitch = pitch.upper().replace('b', 'B')
    
    # Handle Helmholtz notation (C' = C5, C'' = C6, etc)
    octave_num = 4
    if octave:
        octave_num = int(octave)
    elif apostrophes:
        octave_num = 5 + len(apostrophes)
    
    if pitch not in NOTE_MAP:
        raise ValueError(f"Invalid pitch: {pitch}")
    
    return NOTE_MAP[pitch] + (octave_num + 1) * 12

def midi_to_note_name(midi_num: int) -> str:
    """Convert a MIDI note number to note name."""
    if midi_num == -1:
        return "Rest"
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"

# -----------------------------------------------------------------------------
# 5. Duration scaling: guarantee the output sums to requested total (using integers)
# -----------------------------------------------------------------------------
def scale_json_durations(json_data, target_units: int) -> list:
    """Scales durations so that their sum is exactly target_units (8th notes)."""
    durations = [int(d) for _, d in json_data]
    total = sum(durations)
    if total == 0:
        return json_data

    # Calculate proportional scaling with integer arithmetic
    scaled = []
    remainder = target_units
    for i, (note, d) in enumerate(json_data):
        if i < len(json_data) - 1:
            # Proportional allocation
            portion = max(1, round(d * target_units / total))
            scaled.append([note, portion])
            remainder -= portion
        else:
            # Last note gets all remaining units
            scaled.append([note, max(1, remainder)])

    return scaled

# -----------------------------------------------------------------------------
# 6. MIDI from scaled JSON (using integer durations) - UPDATED REST HANDLING
# -----------------------------------------------------------------------------
def json_to_midi(json_data: list, instrument: str, tempo: int, time_signature: str, measures: int, key: str = "C Major") -> MidiFile:
    """Convert JSON data to MIDI file."""
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack(); mid.tracks.append(track)
    program = INSTRUMENT_PROGRAMS.get(instrument, 56)
    numerator, denominator = map(int, time_signature.split('/'))

    # Add time signature meta message
    track.append(MetaMessage('time_signature', numerator=numerator,
                             denominator=denominator, time=0))
    # Add tempo meta message
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    
    # Add key signature meta message based on the key
    # For MIDI key signatures, the key parameter expects a string like 'C', 'F#m', etc.
    key_parts = key.split(' ')
    key_name = key_parts[0]
    is_minor = len(key_parts) > 1 and key_parts[1].lower() == 'minor'
    
    # Convert key to MIDI key signature format
    key_map = {
        'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
        'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7
    }
    
    # Default to C major if key not found
    key_number = key_map.get(key_name, 0)
    
    track.append(MetaMessage('key_signature', key=key_name, time=0))
    
    # Add program change message for instrument
    track.append(Message('program_change', program=program, time=0))
    
    # Calculate total ticks per measure based on time signature
    ticks_per_measure = TICKS_PER_BEAT * numerator * 4 // denominator
    
    # Convert note durations to ticks and add note events
    current_time = 0
    for note, duration in json_data:
        # Convert duration from 8th notes to ticks
        ticks = int(duration * TICKS_PER_8TH)
        
        if not is_rest(note):
            # Add note_on event
            midi_note = note_name_to_midi(note)
            track.append(Message('note_on', note=midi_note, velocity=64, time=current_time))
            current_time = 0  # Reset for note_off
            
            # Add note_off event
            track.append(Message('note_off', note=midi_note, velocity=64, time=ticks))
        else:
            # For rests, just advance the time
            current_time += ticks
    
    return mid

# -----------------------------------------------------------------------------
# 7. MIDI to MP3 conversion (using FluidSynth)
# -----------------------------------------------------------------------------
def midi_to_mp3(mid: MidiFile, instrument: str, output_path: str = None) -> str:
    """Convert MIDI file to MP3 using FluidSynth."""
    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    temp_midi_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.mid")
    temp_wav_path = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.wav")
    
    # If no output path specified, create one in the temp directory
    if output_path is None:
        output_path = os.path.join(temp_dir, f"output_{uuid.uuid4()}.mp3")
    
    try:
        # Save MIDI file
        mid.save(temp_midi_path)
        
        # Get soundfont path
        soundfont_path = get_soundfont(instrument)
        
        # Convert MIDI to WAV using FluidSynth
        fs = FluidSynth(sound_font=soundfont_path)
        fs.midi_to_audio(temp_midi_path, temp_wav_path)
        
        # Convert WAV to MP3 using pydub
        audio = AudioSegment.from_wav(temp_wav_path)
        audio.export(output_path, format="mp3")
        
        return output_path
    
    except Exception as e:
        print(f"Error converting MIDI to MP3: {e}")
        return None
    
    finally:
        # Clean up temporary files
        try:
            os.remove(temp_midi_path)
            os.remove(temp_wav_path)
            os.rmdir(temp_dir)
        except:
            pass

def get_soundfont(instrument: str) -> str:
    """Get the path to the soundfont for the specified instrument."""
    # Check if we have a local soundfont directory
    soundfont_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soundfonts")
    os.makedirs(soundfont_dir, exist_ok=True)
    
    # Path to the soundfont file
    soundfont_path = os.path.join(soundfont_dir, f"{instrument}.sf2")
    
    # If the soundfont doesn't exist, download it
    if not os.path.exists(soundfont_path):
        url = SOUNDFONT_URLS.get(instrument)
        if url:
            try:
                print(f"Downloading soundfont for {instrument}...")
                response = requests.get(url)
                with open(soundfont_path, "wb") as f:
                    f.write(response.content)
                print(f"Downloaded soundfont to {soundfont_path}")
            except Exception as e:
                print(f"Error downloading soundfont: {e}")
                # Use a default soundfont if available
                default_soundfont = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
                if os.path.exists(default_soundfont):
                    return default_soundfont
                else:
                    raise Exception(f"No soundfont available for {instrument}")
    
    return soundfont_path

# -----------------------------------------------------------------------------
# 8. Visualization
# -----------------------------------------------------------------------------
def create_visualization(json_data: list, instrument: str, tempo: int, time_signature: str, measures: int, key: str = "C Major") -> str:
    """Create a visualization of the exercise."""
    # Convert JSON to MIDI
    mid = json_to_midi(json_data, instrument, tempo, time_signature, measures, key)
    
    # Create a temporary file for the MIDI
    temp_midi_path = os.path.join("temp_audio", f"temp_{uuid.uuid4()}.mid")
    mid.save(temp_midi_path)
    
    # Create a visualization using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract note events from MIDI
    notes = []
    current_time = 0
    for msg in mid.tracks[0]:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append((current_time, msg.note, msg.velocity))
    
    # Plot notes
    for time, note, velocity in notes:
        ax.plot(time, note, 'o', markersize=8, alpha=0.7)
    
    ax.set_title(f"{instrument} Exercise - {key}, {time_signature}, {tempo} BPM")
    ax.set_xlabel("Time (ticks)")
    ax.set_ylabel("MIDI Note Number")
    
    # Save the visualization
    viz_path = os.path.join("static", f"visualization_{uuid.uuid4()}.png")
    plt.savefig(viz_path)
    plt.close()
    
    return viz_path

# -----------------------------------------------------------------------------
# 9. Metronome generation
# -----------------------------------------------------------------------------
def create_metronome_audio(tempo: int, time_signature: str, measures: int) -> str:
    """Create a metronome audio file."""
    # Parse time signature
    numerator, denominator = map(int, time_signature.split('/'))
    
    # Calculate beat duration in seconds
    beat_duration = 60 / tempo
    
    # Calculate total number of beats
    beats_per_measure = numerator
    total_beats = beats_per_measure * measures
    
    # Create a sine wave for the metronome click
    sample_rate = 44100
    click_duration = 0.05  # 50 ms click
    t = np.linspace(0, click_duration, int(click_duration * sample_rate), False)
    
    # Create two different frequencies for accented and normal beats
    accented_click = 0.5 * np.sin(2 * np.pi * 1000 * t)
    normal_click = 0.3 * np.sin(2 * np.pi * 800 * t)
    
    # Create the metronome audio
    metronome = np.zeros(int(total_beats * beat_duration * sample_rate))
    
    for i in range(total_beats):
        # Determine if this is an accented beat (first beat of each measure)
        is_accented = (i % beats_per_measure) == 0
        
        # Calculate the position in the audio array
        pos = int(i * beat_duration * sample_rate)
        
        # Add the click
        if is_accented:
            metronome[pos:pos + len(accented_click)] = accented_click
        else:
            metronome[pos:pos + len(normal_click)] = normal_click
    
    # Normalize the audio
    metronome = metronome / np.max(np.abs(metronome))
    
    # Save the metronome audio
    metronome_path = os.path.join("temp_audio", f"metronome_{uuid.uuid4()}.wav")
    wavfile.write(metronome_path, sample_rate, metronome.astype(np.float32))
    
    return metronome_path

# -----------------------------------------------------------------------------
# 10. Difficulty rating calculation
# -----------------------------------------------------------------------------
def calculate_difficulty_rating(json_data: list, instrument: str, level: str, time_signature: str) -> float:
    """Calculate a difficulty rating for the exercise."""
    # Base difficulty by level
    base_difficulty = {"Beginner": 1.0, "Intermediate": 2.0, "Advanced": 3.0}.get(level, 1.0)
    
    # Calculate complexity factors
    note_range = 0
    note_variety = set()
    rhythm_complexity = 0
    
    midi_notes = []
    for note, _ in json_data:
        if not is_rest(note):
            try:
                midi_note = note_name_to_midi(note)
                midi_notes.append(midi_note)
                note_variety.add(note)
            except:
                pass
    
    # Note range factor
    if midi_notes:
        note_range = max(midi_notes) - min(midi_notes)
    
    # Rhythm complexity factor
    durations = [duration for _, duration in json_data]
    unique_durations = len(set(durations))
    rhythm_complexity = unique_durations / len(durations) if durations else 0
    
    # Time signature complexity
    time_complexity = 1.0
    if time_signature != "4/4":
        time_complexity = 1.2
    
    # Calculate final difficulty
    difficulty = base_difficulty * (1 + 0.1 * note_range/12) * (1 + 0.2 * len(note_variety)/12) * (1 + 0.3 * rhythm_complexity) * time_complexity
    
    # Normalize to 1-10 scale
    normalized_difficulty = min(10, max(1, difficulty * 2))
    
    return round(normalized_difficulty, 1)

# -----------------------------------------------------------------------------
# 11. LLM integration for exercise generation
# -----------------------------------------------------------------------------
def query_mistral(prompt: str) -> str:
    """Query the Mistral API for exercise generation."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {MISTRAL_API_KEY}"
    }
    
    data = {
        "model": "mistral-medium",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying Mistral API: {e}")
        return None

def safe_parse_json(json_str: str) -> list:
    """Safely parse JSON from LLM response."""
    # Try to extract JSON from the response
    json_match = re.search(r'\[\s*\[.*?\]\s*\]', json_str, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
    
    try:
        # Try to parse the JSON directly
        return json.loads(json_str)
    except json.JSONDecodeError:
        # If that fails, try to fix common issues
        try:
            # Replace single quotes with double quotes
            json_str = json_str.replace("'", '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If that still fails, try using ast.literal_eval
            try:
                return ast.literal_eval(json_str)
            except:
                # If all else fails, return None
                return None

def get_fallback_exercise(instrument: str, level: str, key: str) -> list:
    """Get a fallback exercise if LLM generation fails."""
    # Simple fallback exercises by level
    if level == "Beginner":
        if key == "C Major":
            return [["C4", 2], ["D4", 2], ["E4", 2], ["C4", 2]]
        else:
            return [["G4", 2], ["A4", 2], ["B4", 2], ["G4", 2]]
    elif level == "Intermediate":
        if key == "C Major":
            return [["C4", 1], ["E4", 1], ["G4", 1], ["C5", 1], ["G4", 1], ["E4", 1], ["C4", 2]]
        else:
            return [["G4", 1], ["B4", 1], ["D5", 1], ["G5", 1], ["D5", 1], ["B4", 1], ["G4", 2]]
    else:  # Advanced
        if key == "C Major":
            return [["C4", 1], ["E4", 1], ["G4", 1], ["C5", 1], ["D5", 1], ["B4", 1], ["G4", 1], ["E4", 1], ["C4", 1]]
        else:
            return [["G4", 1], ["B4", 1], ["D5", 1], ["G5", 1], ["A5", 1], ["F#5", 1], ["D5", 1], ["B4", 1], ["G4", 1]]

# -----------------------------------------------------------------------------
# 12. Main exercise generation function
# -----------------------------------------------------------------------------
def generate_exercise(instrument: str, level: str, key: str, time_signature: str, measures: int, 
                     difficulty_modifier: int = 0, practice_focus: str = "Balanced", 
                     custom_prompt: str = None, tempo: int = 60) -> Tuple[list, str]:
    """Generate a musical exercise based on specified parameters."""
    # Ensure directories exist
    ensure_directories()
    
    # Calculate total duration in 8th notes based on time signature and measures
    numerator, denominator = map(int, time_signature.split('/'))
    total_duration = numerator * measures * 8 // denominator
    
    # Create the prompt for the LLM
    if custom_prompt:
        prompt = custom_prompt
    else:
        prompt = f"""
Generate a {level} level {instrument} exercise in {key} with the following specifications:
- Time signature: {time_signature}
- Number of measures: {measures}
- Focus: {practice_focus}
- Difficulty modifier: {difficulty_modifier} (from -2 to +2)

The exercise should be monophonic (one note at a time) and appropriate for the instrument's range.

Output the exercise as a JSON array where each element is a pair [note_name, duration].
Note names should use standard notation (e.g., "C4", "F#3", "Bb5").
Durations are in 8th notes (e.g., 1 = eighth note, 2 = quarter note, 4 = half note, 8 = whole note).

The total duration MUST be exactly {total_duration} eighth notes to fit the time signature and number of measures.

Example format:
[
  ["C4", 2],
  ["D4", 2],
  ["E4", 4],
  ["Rest", 2],
  ["G4", 6]
]

Only output the JSON array, no additional text.
"""
    
    # Query the LLM
    response = query_mistral(prompt)
    
    # Parse the response
    if response:
        json_data = safe_parse_json(response)
        if json_data:
            # Scale durations to match the requested total
            json_data = scale_json_durations(json_data, total_duration)
            return json_data, response
    
    # Fallback if LLM generation fails
    fallback = get_fallback_exercise(instrument, level, key)
    fallback = scale_json_durations(fallback, total_duration)
    return fallback, "LLM generation failed, using fallback exercise."