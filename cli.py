#!/usr/bin/env python

"""
Adaptive Music Exercise Generator CLI (Strict Duration Enforcement)
==================================================================
A command-line interface for generating custom musical exercises with LLM.
This CLI version replaces the Gradio web interface with a Typer-based CLI.
"""

import typer
import json
import os
import sys
import re
import random
import requests
import tempfile
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from typing import Optional, List, Tuple, Dict
from enum import Enum
from pathlib import Path
import subprocess
import shutil
import uuid
import base64
from datetime import datetime
import time


# -----------------------------------------------------------------------------
# 1. Runtime-time package installation (for fresh environments)
# -----------------------------------------------------------------------------
def install(packages: List[str]):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install([
    "typer[all]", "rich", "mido", "midi2audio", "pydub",
    "requests", "numpy", "matplotlib", "librosa", "scipy"
])

# Import rich for better CLI output
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Create Typer app
app = typer.Typer(help="Adaptive Music Exercise Generator CLI")
console = Console()

# -----------------------------------------------------------------------------
# 2. Configuration & constants (UPDATED TO USE 8TH NOTES)
# -----------------------------------------------------------------------------
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = "yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX"  # Replace with your key!

SOUNDFONT_URLS = {
    "Trumpet": "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/Trumpet.sf2",
    "Piano": "https://musical-artifacts.com/artifacts/2719/GeneralUser_GS_1.471.sf2",
    "Violin": "https://musical-artifacts.com/artifacts/2744/SalC5Light.sf2",
    "Clarinet": "https://musical-artifacts.com/artifacts/2744/SalC5Light.sf2",
    "Flute": "https://musical-artifacts.com/artifacts/2744/SalC5Light.sf2",
}

SAMPLE_RATE = 44100  # Hz
TICKS_PER_BEAT = 480  # Standard MIDI resolution
TICKS_PER_8TH = TICKS_PER_BEAT // 2  # 240 ticks per 8th note (UPDATED)

os.makedirs("static", exist_ok=True)
os.makedirs("temp_audio", exist_ok=True)
os.makedirs("soundfonts", exist_ok=True)

# -----------------------------------------------------------------------------
# 3. Music theory helpers (note names ↔︎ MIDI numbers) - ENHANCED REST HANDLING
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
    "Piano": 0, "Trumpet": 56, "Violin": 40,
    "Clarinet": 71, "Flute": 73,
}


def is_rest(note: str) -> bool:
    """Check if a note string represents a rest."""
    return note.strip().lower() in [r.lower() for r in REST_INDICATORS]


def note_name_to_midi(note: str) -> int:
    if is_rest(note):
        return -1  # Special value for rests

    # Allow both scientific (C4) and Helmholtz (C') notation
    match = re.match(r"([A-Ga-g][#b]?)(\'*)(\d?)", note)
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
    if midi_num == -1:
        return "Rest"
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"


# -----------------------------------------------------------------------------
# 4. Duration scaling: guarantee the output sums to requested total (using integers)
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
# 5. MIDI from scaled JSON (using integer durations) - UPDATED REST HANDLING
# -----------------------------------------------------------------------------
def json_to_midi(json_data: list, instrument: str, tempo: int, time_signature: str, measures: int,
                 key: str = "C Major") -> MidiFile:
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack();
    mid.tracks.append(track)
    program = INSTRUMENT_PROGRAMS.get(instrument, 56)
    numerator, denominator = map(int, time_signature.split('/'))

    # Add time signature meta message
    track.append(MetaMessage('time_signature', numerator=numerator,
                             denominator=denominator, time=0))
    # Add tempo meta message
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

    # Add key signature meta message based on the key
    key_map = {
        "C Major": "C",
        "G Major": "G",
        "D Major": "D",
        "F Major": "F",
        "Bb Major": "Bb",
        "A Minor": "Am",
        "E Minor": "Em",
    }

    # Use the provided key or default to C major if key not found
    midi_key = key_map.get(key, "C")
    # The 'key' parameter in MetaMessage expects a string like 'C', 'F#m', etc.
    track.append(MetaMessage('key_signature', key=midi_key, time=0))

    # Set instrument program
    track.append(Message('program_change', program=program, time=0))

    # Accumulator for rest durations
    accumulated_rest = 0

    for note_item in json_data:
        try:
            # Handle both formats: [note, duration] and {note, duration}
            if isinstance(note_item, list) and len(note_item) == 2:
                note_name, duration_units = note_item
            elif isinstance(note_item, dict):
                note_name = note_item["note"]
                duration_units = note_item["duration"]
            else:
                print(f"Unsupported note format: {note_item}")
                continue

            ticks = int(duration_units * TICKS_PER_8TH)
            ticks = max(ticks, 1)

            if is_rest(note_name):
                # Accumulate rest duration
                accumulated_rest += ticks
            else:
                # Process any accumulated rest first
                if accumulated_rest > 0:
                    # Add rest by creating a silent note (velocity 0) that won't be heard
                    # Or just skip and use accumulated_rest in timing
                    # We'll just add the time to the next note
                    track.append(Message('note_on', note=0, velocity=0, time=accumulated_rest))
                    track.append(Message('note_off', note=0, velocity=0, time=0))
                    accumulated_rest = 0

                # Process actual note
                note_num = note_name_to_midi(note_name)
                velocity = random.randint(60, 100)
                track.append(Message('note_on', note=note_num, velocity=velocity, time=0))
                track.append(Message('note_off', note=note_num, velocity=velocity, time=ticks))
        except Exception as e:
            print(f"Error parsing note {note_item}: {e}")

    # Handle trailing rest
    if accumulated_rest > 0:
        track.append(Message('note_on', note=0, velocity=0, time=accumulated_rest))
        track.append(Message('note_off', note=0, velocity=0, time=0))

    return mid


# -----------------------------------------------------------------------------
# 6. MIDI → Audio (MP3) helpers
# -----------------------------------------------------------------------------
def get_soundfont(instrument: str) -> str:
    sf2_path = f"soundfonts/{instrument}.sf2"
    if not os.path.exists(sf2_path):
        url = SOUNDFONT_URLS.get(instrument, SOUNDFONT_URLS["Trumpet"])
        print(f"Downloading SoundFont for {instrument}…")
        try:
            response = requests.get(url)
            with open(sf2_path, "wb") as f:
                f.write(response.content)
            # Verify it's a valid SF2 file (not HTML/error)
            if b"<html" in response.content[:100].lower() or b"<!doctype" in response.content[:100].lower():
                print(f"Warning: Downloaded file for {instrument} appears to be HTML or invalid.")
                os.remove(sf2_path)
                return None
        except Exception as e:
            print(f"Failed to download soundfont: {e}")
            return None
    return sf2_path


def midi_to_mp3(midi_obj: MidiFile, instrument: str = "Trumpet", force_fallback: bool = False) -> Tuple[str, float]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as mid_file:
        midi_obj.save(mid_file.name)
        wav_path = mid_file.name.replace(".mid", ".wav")
        mp3_path = mid_file.name.replace(".mid", ".mp3")

    # Use fallback if requested or if no valid soundfont available
    if force_fallback:
        return generate_fallback_audio(midi_obj, mp3_path)

    sf2_path = get_soundfont(instrument)
    if not sf2_path:
        print("No valid soundfont available, using fallback audio generation")
        return generate_fallback_audio(midi_obj, mp3_path)

    try:
        # Try using fluidsynth command first
        result = subprocess.run([
            'fluidsynth', '-ni', sf2_path, mid_file.name,
            '-F', wav_path, '-r', '44100', '-g', '1.0'
        ], check=True, capture_output=True, text=True)

        # Convert to MP3
        from pydub import AudioSegment
        sound = AudioSegment.from_wav(wav_path)
        if instrument == "Trumpet":
            sound = sound.high_pass_filter(200)
        elif instrument == "Violin":
            sound = sound.low_pass_filter(5000)
        sound.export(mp3_path, format="mp3")

        # Move to static directory
        static_mp3_path = os.path.join('static', f'exercise_{uuid.uuid4().hex}.mp3')
        shutil.move(mp3_path, static_mp3_path)
        return static_mp3_path, sound.duration_seconds
    except Exception as e:
        print(f"FluidSynth failed: {e}, trying fallback")
        return generate_fallback_audio(midi_obj, mp3_path)
    finally:
        # Clean up temporary files
        for f in [mid_file.name, wav_path]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


def generate_fallback_audio(midi_obj: MidiFile, output_path: str) -> Tuple[str, float]:
    """Generate simple audio using sine waves as fallback"""
    try:
        import numpy as np
        from scipy.io import wavfile
        from pydub import AudioSegment

        # Parse MIDI to get notes and timing
        notes = []
        for track in midi_obj.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append((msg.note, msg.time))

        # Generate simple sine wave audio
        sample_rate = 44100
        audio_data = np.zeros(sample_rate * 10)  # 10 seconds max

        position = 0
        for note, duration in notes:
            # Convert MIDI note to frequency
            freq = 440 * (2 ** ((note - 69) / 12))
            # Convert ticks to seconds
            duration_sec = duration * (0.5 / TICKS_PER_BEAT)  # Assuming 120 BPM

            # Generate sine wave
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
            wave = 0.5 * np.sin(2 * np.pi * freq * t)

            # Add to audio data
            end_pos = position + len(wave)
            if end_pos < len(audio_data):
                audio_data[position:end_pos] += wave
            position = end_pos

        # Normalize and convert to 16-bit PCM
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

        # Save as WAV then convert to MP3
        wav_path = output_path.replace('.mp3', '.wav')
        wavfile.write(wav_path, sample_rate, audio_data)

        sound = AudioSegment.from_wav(wav_path)
        sound.export(output_path, format="mp3")

        # Move to static directory
        static_mp3_path = os.path.join('static', f'exercise_{uuid.uuid4().hex}.mp3')
        shutil.move(output_path, static_mp3_path)

        return static_mp3_path, sound.duration_seconds
    except Exception as e:
        print(f"Fallback audio generation failed: {e}")
        return None, 0


# -----------------------------------------------------------------------------
# 7. Prompt engineering for variety (using integer durations) - UPDATED DURATION SYSTEM
# -----------------------------------------------------------------------------
def get_fallback_exercise(instrument: str, level: str, key: str,
                          time_sig: str, measures: int) -> str:
    key_notes = {
        "C Major": ["C4", "D4", "E4", "F4", "G4", "A4", "B4"],
        "G Major": ["G3", "A3", "B3", "C4", "D4", "E4", "F#4"],
        "D Major": ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5"],
        "F Major": ["F3", "G3", "A3", "Bb3", "C4", "D4", "E4"],
        "Bb Major": ["Bb3", "C4", "D4", "Eb4", "F4", "G4", "A4"],
        "A Minor": ["A3", "B3", "C4", "D4", "E4", "F4", "G4"],
        "E Minor": ["E3", "F#3", "G3", "A3", "B3", "C4", "D4"],
    }

    # Get fundamental note from key signature
    fundamental_note = key.split()[0]  # Gets 'C' from 'C Major' or 'A' from 'A Minor'
    is_major = "Major" in key

    # Get notes for the key
    notes = key_notes.get(key, key_notes["C Major"])

    # Find fundamental note with octave for ending
    fundamental_with_octave = None
    for note in notes:
        if note.startswith(fundamental_note):
            fundamental_with_octave = note
            break

    # If not found, use the first note (should not happen with our key definitions)
    if not fundamental_with_octave:
        fundamental_with_octave = notes[0]

    numerator, denominator = map(int, time_sig.split('/'))

    # Calculate units based on 8th notes
    units_per_measure = numerator * (8 // denominator)
    target_units = measures * units_per_measure

    # Create a rhythm pattern based on time signature
    if numerator == 3:
        rhythm = [2, 1, 2, 1, 2]  # 3/4 pattern
    else:
        rhythm = [2, 2, 1, 1, 2, 2]  # 4/4 pattern

    # Build exercise
    result = []
    cumulative = 0
    current_units = 0

    # Reserve at least 2 units for the final note
    final_note_duration = min(4, max(2, rhythm[0]))  # Between 2 and 4 units
    available_units = target_units - final_note_duration

    # Generate notes until we reach the available units
    while current_units < available_units:
        # Avoid minor 7th in major keys
        if is_major:
            # Filter out minor 7th notes (e.g., Bb in C major)
            available_notes = [n for n in notes if not (n.startswith("Bb") and key == "C Major") and
                               not (n.startswith("F") and key == "G Major") and
                               not (n.startswith("C") and key == "D Major") and
                               not (n.startswith("Eb") and key == "F Major") and
                               not (n.startswith("Ab") and key == "Bb Major")]
        else:
            available_notes = notes

        note = random.choice(available_notes)
        dur = random.choice(rhythm)

        # Don't exceed available units
        if current_units + dur > available_units:
            dur = available_units - current_units
            if dur <= 0:
                break

        cumulative += dur
        current_units += dur
        result.append({
            "note": note,
            "duration": dur,
            "cumulative_duration": cumulative
        })

    # Add the final note (fundamental of the key)
    final_duration = target_units - current_units
    if final_duration > 0:
        cumulative += final_duration
        result.append({
            "note": fundamental_with_octave,
            "duration": final_duration,
            "cumulative_duration": cumulative
        })

    return json.dumps(result)


def get_style_based_on_level(level: str) -> str:
    styles = {
        "Beginner": ["simple", "legato", "stepwise", "folk-like", "gentle"],
        "Intermediate": ["jazzy", "bluesy", "march-like", "syncopated", "dance-like", "lyrical"],
        "Advanced": ["technical", "chromatic", "fast arpeggios", "wide intervals", "virtuosic", "complex",
                     "contemporary"],
    }
    return random.choice(styles.get(level, ["technical"]))


def get_technique_based_on_level(level: str) -> str:
    techniques = {
        "Beginner": [
            "with long tones", "with simple rhythms", "focusing on tone",
            "with step-wise motion", "with easy intervals", "focusing on breath control",
            "with simple articulation", "with repeated patterns"
        ],
        "Intermediate": [
            "with slurs", "with accents", "using triplets", "with moderate syncopation",
            "with varied articulation", "with moderate interval jumps", "with dynamic contrast",
            "with scale patterns", "with simple ornaments", "with moderate register changes"
        ],
        "Advanced": [
            "with double tonguing", "with extreme registers", "with complex rhythms",
            "with challenging intervals", "with rapid articulation", "with advanced ornaments",
            "with extended techniques", "with complex syncopation", "with virtuosic passages",
            "with extreme dynamic contrast", "with challenging arpeggios"
        ],
    }
    return random.choice(techniques.get(level, ["with slurs"]))


# -----------------------------------------------------------------------------
# 8. Mistral API: query, fallback on errors - UPDATED DURATION SYSTEM
# -----------------------------------------------------------------------------
def query_mistral(prompt: str, instrument: str, level: str, key: str,
                  time_sig: str, measures: int, difficulty_modifier: int = 0,
                  practice_focus: str = "Balanced") -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    numerator, denominator = map(int, time_sig.split('/'))

    # UPDATED: Calculate total required 8th notes
    units_per_measure = numerator * (8 // denominator)
    required_total = measures * units_per_measure

    # UPDATED: Duration explanation in prompt
    duration_constraint = (
        f"Sum of all durations MUST BE EXACTLY {required_total} units (8th notes). "
        f"Each integer duration represents an 8th note (1=8th, 2=quarter, 4=half, 8=whole). "
        f"If it doesn't match, the exercise is invalid."
    )
    system_prompt = (
        f"You are an expert music teacher specializing in {instrument.lower()}. "
        "Create customized exercises using INTEGER durations representing 8th notes."
    )

    if prompt.strip():
        user_prompt = (
            f"{prompt} {duration_constraint} Output ONLY a JSON array of objects with "
            "the following structure: [{{'note': string, 'duration': integer, 'cumulative_duration': integer}}]"
        )
    else:
        # Adjust level based on difficulty modifier
        effective_level = level
        if difficulty_modifier != 0:
            level_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
            level_list = ["Beginner", "Intermediate", "Advanced"]
            base_level_idx = level_map.get(level, 1)
            adjusted_idx = max(0, min(2, base_level_idx + difficulty_modifier))
            effective_level = level_list[adjusted_idx]

        style = get_style_based_on_level(effective_level)
        technique = get_technique_based_on_level(effective_level)

        # Extract fundamental note from key signature
        fundamental_note = key.split()[0]  # Gets 'C' from 'C Major' or 'A' from 'A Minor'
        is_major = "Major" in key

        # Create additional musical constraints
        key_constraints = (
            f"The exercise MUST end on the fundamental note of the key ({fundamental_note}). "
            f"{'' if not is_major else 'For this major key, avoid using the minor 7th degree.'}"
        )

        # Add practice focus constraints
        focus_constraints = ""
        if practice_focus == "Rhythmic Focus":
            focus_constraints = "Include varied rhythmic patterns with syncopation and different note durations. "
        elif practice_focus == "Melodic Focus":
            focus_constraints = "Create a melodically interesting line with good contour and phrasing. "
        elif practice_focus == "Technical Focus":
            focus_constraints = "Include technical challenges like arpeggios, scales, or interval jumps. "
        elif practice_focus == "Expressive Focus":
            focus_constraints = "Design a lyrical exercise with opportunities for dynamic contrast and expression. "

        # Difficulty modifier description for prompt
        difficulty_desc = ""
        if difficulty_modifier > 0:
            difficulty_desc = f"Make this slightly more challenging than a typical {level.lower()} exercise. "
        elif difficulty_modifier < 0:
            difficulty_desc = f"Make this slightly easier than a typical {level.lower()} exercise. "

        user_prompt = (
            f"Create a {style} {instrument.lower()} exercise in {key} with {time_sig} time signature "
            f"{technique} for a {level.lower()} player. {difficulty_desc}{focus_constraints}{duration_constraint} {key_constraints} "
            "Output ONLY a JSON array of objects with the following structure: "
            "[{{'note': string, 'duration': integer, 'cumulative_duration': integer}}] "
            "Use standard note names (e.g., \"Bb4\", \"F#5\"). Monophonic only. "
            "Durations: 1=8th, 2=quarter, 4=half, 8=whole. "
            "Sum must be exactly as specified. ONLY output the JSON array. No prose."
        )

    payload = {
        "model": "mistral-medium",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7 if level == "Advanced" else 0.5,
        "max_tokens": 1000,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        print(f"Error querying Mistral API: {e}")
        return get_fallback_exercise(instrument, level, key, time_sig, measures)


# -----------------------------------------------------------------------------
# 9. Robust JSON parsing for LLM outputs - ENHANCED PARSING
# -----------------------------------------------------------------------------
def safe_parse_json(text: str) -> Optional[list]:
    try:
        text = text.strip().replace("'", '"')

        # Find JSON array in the text
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1:
            return None

        json_str = text[start_idx:end_idx + 1]

        # Fix common JSON issues
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Trailing commas
        json_str = re.sub(r'{\s*(\w+)\s*:', r'{"\1":', json_str)  # Unquoted keys
        json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)(\s*[,}])', r':"\1"\2', json_str)  # Unquoted strings

        parsed = json.loads(json_str)

        # Normalize keys to 'note' and 'duration'
        normalized = []
        for item in parsed:
            if isinstance(item, dict):
                # Find note value - accept multiple keys
                note_val = None
                for key in ['note', 'pitch', 'nota', 'ton']:
                    if key in item:
                        note_val = str(item[key])
                        break

                # Find duration value
                dur_val = None
                for key in ['duration', 'dur', 'length', 'value']:
                    if key in item:
                        try:
                            dur_val = int(item[key])
                        except (TypeError, ValueError):
                            pass

                if note_val is not None and dur_val is not None:
                    normalized.append({"note": note_val, "duration": dur_val})

        return normalized if normalized else None

    except Exception as e:
        print(f"JSON parsing error: {e}\nRaw text: {text}")
        return None


# -----------------------------------------------------------------------------
# 10. Note cleaning function to handle ornamentation
# -----------------------------------------------------------------------------
def clean_note_string(note_str):
    """
    Clean note strings by removing ornamentation symbols that cause parsing errors.
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


# -----------------------------------------------------------------------------
# 11. Main orchestration: talk to API, *scale durations*, build MIDI, UI values - UPDATED
# -----------------------------------------------------------------------------
def generate_exercise(instrument: str, level: str, key: str, tempo: int, time_signature: str,
                      measures: int, custom_prompt: str, mode: str, difficulty_modifier: int = 0,
                      practice_focus: str = "Balanced", force_fallback: bool = False) -> Tuple[
    str, Optional[str], str, MidiFile, str, str, int]:
    try:
        prompt_to_use = custom_prompt if mode == "Exercise Prompt" else ""
        output = query_mistral(prompt_to_use, instrument, level, key, time_signature, measures, difficulty_modifier,
                               practice_focus)
        parsed = safe_parse_json(output)
        if not parsed:
            print("Primary parsing failed, using fallback")
            fallback_str = get_fallback_exercise(instrument, level, key, time_signature, measures)
            parsed = safe_parse_json(fallback_str)
            if not parsed:
                print("Fallback parsing failed, using ultimate fallback")
                # Ultimate fallback: simple scale based on selected key
                key_notes = {
                    "C Major": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
                    "G Major": ["G3", "A3", "B3", "C4", "D4", "E4", "F#4", "G4"],
                    "D Major": ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5"],
                    "F Major": ["F3", "G3", "A3", "Bb3", "C4", "D4", "E4", "F4"],
                    "Bb Major": ["Bb3", "C4", "D4", "Eb4", "F4", "G4", "A4", "Bb4"],
                    "A Minor": ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4"],
                    "E Minor": ["E3", "F#3", "G3", "A3", "B3", "C4", "D4", "E4"],
                }
                notes = key_notes.get(key, key_notes["C Major"])
                numerator, denominator = map(int, time_signature.split('/'))
                units_per_measure = numerator * (8 // denominator)
                target_units = measures * units_per_measure
                note_duration = max(1, target_units // len(notes))
                parsed = [{"note": n, "duration": note_duration} for n in notes]
                # Adjust last note to match total duration
                total = sum(item["duration"] for item in parsed)
                if total < target_units:
                    parsed[-1]["duration"] += target_units - total
                elif total > target_units:
                    parsed[-1]["duration"] -= total - target_units

        # Clean note strings to remove ornamentation
        for item in parsed:
            if 'note' in item:
                item['note'] = clean_note_string(item['note'])

        # Calculate total required 8th notes (UPDATED)
        numerator, denominator = map(int, time_signature.split('/'))
        units_per_measure = numerator * (8 // denominator)
        total_units = measures * units_per_measure

        # Convert to old format for scaling
        old_format = []
        for item in parsed:
            if isinstance(item, dict):
                old_format.append([item["note"], item["duration"]])
            else:
                old_format.append(item)

        # Strict scaling
        parsed_scaled_old = scale_json_durations(old_format, total_units)

        # Convert back to new format with cumulative durations
        cumulative = 0
        parsed_scaled = []
        for note, dur in parsed_scaled_old:
            cumulative += dur
            parsed_scaled.append({
                "note": note,
                "duration": dur,
                "cumulative_duration": cumulative
            })

        # Calculate total duration units
        total_duration = cumulative

        # Generate MIDI and audio
        midi = json_to_midi(parsed_scaled, instrument, tempo, time_signature, measures, key)
        mp3_path, real_duration = midi_to_mp3(midi, instrument, force_fallback)
        output_json_str = json.dumps(parsed_scaled, indent=2)
        return output_json_str, mp3_path, str(
            tempo), midi, f"{real_duration:.2f} seconds", time_signature, total_duration
    except Exception as e:
        return f"Error: {str(e)}", None, str(tempo), None, "0", time_signature, 0


# -----------------------------------------------------------------------------
# 12. Visualization function
# -----------------------------------------------------------------------------
def create_visualization(json_data, time_sig):
    try:
        if not json_data or "Error" in json_data:
            return None

        parsed = json.loads(json_data)
        if not isinstance(parsed, list) or len(parsed) == 0:
            return None

        # Extract notes and durations
        notes = []
        durations = []
        for item in parsed:
            if isinstance(item, dict) and "note" in item and "duration" in item:
                note_name = item["note"]
                if not is_rest(note_name):
                    try:
                        midi_note = note_name_to_midi(note_name)
                        notes.append(midi_note)
                        durations.append(item["duration"])
                    except ValueError:
                        notes.append(60)  # Default to middle C if parsing fails
                        durations.append(item["duration"])
                else:
                    notes.append(None)  # Represent rest
                    durations.append(item["duration"])

        # Create piano roll visualization
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate time positions
        time_positions = [0]
        for dur in durations[:-1]:
            time_positions.append(time_positions[-1] + dur)

        # Plot notes as rectangles
        for i, (note, dur, pos) in enumerate(zip(notes, durations, time_positions)):
            if note is not None:  # Skip rests
                rect = plt.Rectangle((pos, note - 0.4), dur, 0.8, color='blue', alpha=0.7)
                ax.add_patch(rect)
                # Add note name
                ax.text(pos + dur / 2, note + 0.5, midi_to_note_name(note),
                        ha='center', va='bottom', fontsize=8)

        # Add measure lines
        numerator, denominator = map(int, time_sig.split('/'))
        units_per_measure = numerator * (8 // denominator)
        max_time = time_positions[-1] + durations[-1]
        for measure in range(1, int(max_time / units_per_measure) + 1):
            measure_pos = measure * units_per_measure
            if measure_pos <= max_time:
                ax.axvline(x=measure_pos, color='gray', linestyle='--', alpha=0.5)

        # Set axis limits and labels
        if notes and None not in notes:
            ax.set_ylim(min(notes) - 5, max(notes) + 5)
        else:
            ax.set_ylim(55, 75)
        ax.set_xlim(0, max_time)
        ax.set_ylabel('MIDI Note')
        ax.set_xlabel('Time (8th note units)')
        ax.set_title('Exercise Visualization')

        # Add piano keyboard on y-axis
        ax.set_yticks([60, 62, 64, 65, 67, 69, 71, 72])  # C4 to C5
        ax.set_yticklabels(['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'])
        ax.grid(True, axis='y', alpha=0.3)

        # Save figure to temporary file
        temp_img_path = os.path.join('static', f'visualization_{uuid.uuid4().hex}.png')
        plt.tight_layout()
        plt.savefig(temp_img_path)
        plt.close()

        return temp_img_path
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


# -----------------------------------------------------------------------------
# 13. Metronome function
# -----------------------------------------------------------------------------
def create_metronome_audio(tempo, time_sig, measures):
    try:
        numerator, denominator = map(int, time_sig.split('/'))
        # Create a MIDI file with metronome clicks
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

        # Save and convert to audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as mid_file:
            mid.save(mid_file.name)
            wav_path = mid_file.name.replace(".mid", ".wav")
            mp3_path = mid_file.name.replace(".mid", ".mp3")

        # Use fallback audio generation for metronome
        from pydub import AudioSegment
        from pydub.generators import Sine

        # Create metronome clicks with pydub
        click_duration = 50  # milliseconds
        strong_click = Sine(1000).to_audio_segment(duration=click_duration).apply_gain(-3)
        weak_click = Sine(800).to_audio_segment(duration=click_duration).apply_gain(-6)

        silence_duration = 60000 / tempo - click_duration  # ms per beat minus click

        metronome_audio = AudioSegment.silent(duration=0)

        for beat in range(total_beats):
            click = strong_click if beat % beats_per_measure == 0 else weak_click
            metronome_audio += click + AudioSegment.silent(duration=silence_duration)

        # Export to MP3
        metronome_audio.export(mp3_path, format="mp3")

        # Move to static directory
        static_mp3_path = os.path.join('static', f'metronome_{uuid.uuid4().hex}.mp3')
        shutil.move(mp3_path, static_mp3_path)

        # Clean up temporary files
        for f in [mid_file.name, wav_path]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

        return static_mp3_path
    except Exception as e:
        print(f"Error creating metronome: {e}")
        return None


# -----------------------------------------------------------------------------
# 14. Function to calculate difficulty rating
# -----------------------------------------------------------------------------
def calculate_difficulty_rating(json_data, level, difficulty_modifier=0, practice_focus="Balanced"):
    try:
        if not json_data or "Error" in json_data:
            return 0

        parsed = json.loads(json_data)
        if not isinstance(parsed, list) or len(parsed) == 0:
            return 0

        # Extract notes and durations
        notes = []
        durations = []
        for item in parsed:
            if isinstance(item, dict) and "note" in item and "duration" in item:
                note_name = item["note"]
                if not is_rest(note_name):
                    try:
                        midi_note = note_name_to_midi(note_name)
                        notes.append(midi_note)
                        durations.append(item["duration"])
                    except ValueError:
                        pass

        if not notes:
            return 0

        # Calculate difficulty factors
        # 1. Range (wider range = harder)
        note_range = max(notes) - min(notes) if notes else 0
        range_factor = min(note_range / 12, 1.0)  # Normalize to octave

        # 2. Rhythmic complexity (more varied durations = harder)
        unique_durations = len(set(durations))
        rhythm_factor = min(unique_durations / 4, 1.0)  # Normalize

        # 3. Interval jumps (larger jumps = harder)
        jumps = [abs(notes[i] - notes[i - 1]) for i in range(1, len(notes))]
        avg_jump = sum(jumps) / len(jumps) if jumps else 0
        jump_factor = min(avg_jump / 7, 1.0)  # Normalize to perfect fifth

        # 4. Speed factor (shorter durations = harder)
        avg_duration = sum(durations) / len(durations) if durations else 0
        speed_factor = min(2.0 / avg_duration if avg_duration > 0 else 1.0, 1.0)  # Normalize

        # Adjust weights based on practice focus
        weights = {"range": 0.25, "rhythm": 0.25, "jump": 0.25, "speed": 0.25}

        if practice_focus == "Rhythmic Focus":
            weights = {"range": 0.15, "rhythm": 0.55, "jump": 0.15, "speed": 0.15}
        elif practice_focus == "Melodic Focus":
            weights = {"range": 0.40, "rhythm": 0.15, "jump": 0.30, "speed": 0.15}
        elif practice_focus == "Technical Focus":
            weights = {"range": 0.25, "rhythm": 0.15, "jump": 0.40, "speed": 0.20}
        elif practice_focus == "Expressive Focus":
            weights = {"range": 0.35, "rhythm": 0.25, "jump": 0.25, "speed": 0.15}

        # Calculate base difficulty with adjusted weights
        base_difficulty = (
                range_factor * weights["range"] +
                rhythm_factor * weights["rhythm"] +
                jump_factor * weights["jump"] +
                speed_factor * weights["speed"]
        )

        # Apply level multiplier
        level_multiplier = {
            "Beginner": 0.7,
            "Intermediate": 1.0,
            "Advanced": 1.3
        }.get(level, 1.0)

        # Apply difficulty modifier (each step is about 15% change)
        modifier_multiplier = 1.0 + (difficulty_modifier * 0.15)

        # Calculate final rating (1-10 scale)
        rating = round(base_difficulty * level_multiplier * modifier_multiplier * 10)
        return max(1, min(rating, 10))  # Ensure between 1-10
    except Exception as e:
        print(f"Error calculating difficulty: {e}")
        return 0


# -----------------------------------------------------------------------------
# 15. Define enums for CLI options
# -----------------------------------------------------------------------------
class Instrument(str, Enum):
    TRUMPET = "Trumpet"
    PIANO = "Piano"
    VIOLIN = "Violin"
    CLARINET = "Clarinet"
    FLUTE = "Flute"


class Level(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class Key(str, Enum):
    C_MAJOR = "C Major"
    G_MAJOR = "G Major"
    D_MAJOR = "D Major"
    F_MAJOR = "F Major"
    BB_MAJOR = "Bb Major"
    A_MINOR = "A Minor"
    E_MINOR = "E Minor"


class TimeSignature(str, Enum):
    THREE_FOUR = "3/4"
    FOUR_FOUR = "4/4"


class PracticeFocus(str, Enum):
    BALANCED = "Balanced"
    RHYTHMIC = "Rhythmic Focus"
    MELODIC = "Melodic Focus"
    TECHNICAL = "Technical Focus"
    EXPRESSIVE = "Expressive Focus"


class OutputFormat(str, Enum):
    JSON = "json"
    MIDI = "midi"
    MP3 = "mp3"
    ALL = "all"


# -----------------------------------------------------------------------------
# 16. CLI Commands
# -----------------------------------------------------------------------------
@app.command("generate")
def generate(
        instrument: Instrument = typer.Option(Instrument.TRUMPET, help="Instrument to generate exercise for"),
        level: Level = typer.Option(Level.INTERMEDIATE, help="Difficulty level"),
        key: Key = typer.Option(Key.C_MAJOR, help="Key signature"),
        time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
        measures: int = typer.Option(4, help="Number of measures", min=1, max=16),
        difficulty_modifier: int = typer.Option(0, help="Difficulty modifier (-2 to +2)", min=-2, max=2),
        practice_focus: PracticeFocus = typer.Option(PracticeFocus.BALANCED, help="Practice focus"),
        output_format: OutputFormat = typer.Option(OutputFormat.ALL, help="Output format"),
        output_dir: str = typer.Option("./output", help="Directory to save output files"),
        custom_prompt: Optional[str] = typer.Option(None, help="Custom prompt for exercise generation"),
        tempo: int = typer.Option(60, help="Tempo in BPM", min=40, max=200),
        force_fallback: bool = typer.Option(False, help="Force using fallback audio generation instead of soundfonts"),
):
    """Generate a musical exercise based on specified parameters."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Show parameters
    console.print("[bold green]Generating exercise with the following parameters:[/bold green]")
    params_table = Table(show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter")
    params_table.add_column("Value")
    params_table.add_row("Instrument", str(instrument))
    params_table.add_row("Level", str(level))
    params_table.add_row("Key", str(key))
    params_table.add_row("Time Signature", str(time_signature))
    params_table.add_row("Measures", str(measures))
    params_table.add_row("Difficulty Modifier", str(difficulty_modifier))
    params_table.add_row("Practice Focus", str(practice_focus))
    params_table.add_row("Tempo", f"{tempo} BPM")
    console.print(params_table)

    # Generate exercise
    with console.status("[bold green]Generating exercise...[/bold green]"):
        mode = "Exercise Prompt" if custom_prompt else "Exercise Parameters"
        # Extract string values from enums
        instrument_str = instrument.value
        level_str = level.value
        key_str = key.value
        time_sig_str = time_signature.value
        practice_focus_str = practice_focus.value

        json_data, mp3_path, tempo_str, midi_obj, duration, time_sig, total_duration = generate_exercise(
            instrument_str, level_str, key_str, tempo, time_sig_str,
            measures, custom_prompt or "", mode, difficulty_modifier, practice_focus_str,
            force_fallback=force_fallback
        )

    # Calculate difficulty rating
    difficulty_rating = calculate_difficulty_rating(json_data, level_str, difficulty_modifier, practice_focus_str)

    # Save outputs based on format
    # Create safe filename components by replacing problematic characters
    safe_instrument = instrument_str.replace(' ', '_')
    safe_level = level_str.replace(' ', '_')
    base_filename = f"exercise_{safe_instrument}_{safe_level}_{measures}m"
    output_files = []

    if output_format in [OutputFormat.JSON, OutputFormat.ALL]:
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, "w") as f:
            f.write(json_data)
        output_files.append(("JSON", json_path))

    if output_format in [OutputFormat.MIDI, OutputFormat.ALL]:
        midi_path = os.path.join(output_dir, f"{base_filename}.mid")
        midi_obj.save(midi_path)
        output_files.append(("MIDI", midi_path))

    if output_format in [OutputFormat.MP3, OutputFormat.ALL]:
        if mp3_path:
            # Copy the MP3 file to the output directory
            new_mp3_path = os.path.join(output_dir, f"{base_filename}.mp3")
            if os.path.exists(mp3_path):
                import shutil
                shutil.copy(mp3_path, new_mp3_path)
                output_files.append(("MP3", new_mp3_path))

    # Generate visualization if all formats are requested
    if output_format == OutputFormat.ALL:
        try:
            viz_path = create_visualization(json_data, str(time_signature))
            if viz_path:
                viz_output = os.path.join(output_dir, f"{base_filename}_viz.png")
                import shutil
                shutil.copy(viz_path, viz_output)
                output_files.append(("Visualization", viz_output))
        except Exception as e:
            console.print(f"[bold red]Error generating visualization: {e}[/bold red]")

    # Display results
    console.print("\n[bold green]Exercise generated successfully![/bold green]")
    console.print(f"[bold]Difficulty Rating:[/bold] {difficulty_rating}/10")
    console.print(f"[bold]Duration:[/bold] {duration} seconds")
    console.print(f"[bold]Total Duration Units:[/bold] {total_duration} (8th notes)")

    # Show output files
    if output_files:
        console.print("\n[bold]Output Files:[/bold]")
        for file_type, file_path in output_files:
            console.print(f"[bold]{file_type}:[/bold] {file_path}")

    # Show JSON preview
    if output_format in [OutputFormat.JSON, OutputFormat.ALL]:
        console.print("\n[bold]JSON Preview:[/bold]")
        try:
            parsed_json = json.loads(json_data)
            console.print_json(json.dumps(parsed_json[:5] if len(parsed_json) > 5 else parsed_json))
            if len(parsed_json) > 5:
                console.print("[italic](showing first 5 notes only)[/italic]")
        except:
            console.print(json_data[:200] + "..." if len(json_data) > 200 else json_data)


@app.command("metronome")
def metronome(
        tempo: int = typer.Option(60, help="Tempo in BPM", min=40, max=200),
        time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
        measures: int = typer.Option(4, help="Number of measures", min=1, max=16),
        output_dir: str = typer.Option("./output", help="Directory to save output file"),
):
    """Generate a metronome audio file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate metronome
    with console.status(f"[bold green]Generating metronome at {tempo} BPM...[/bold green]"):
        # Extract the actual time signature string from the enum
        time_sig_str = time_signature.value
        metronome_path = create_metronome_audio(tempo, time_sig_str, measures)

    if metronome_path:
        # Copy the metronome file to the output directory
        # Replace slash with underscore to avoid file path issues
        safe_time_sig = time_sig_str.replace('/', '_')
        output_path = os.path.join(output_dir, f"metronome_{tempo}bpm_{safe_time_sig}_{measures}m.mp3")
        import shutil
        shutil.copy(metronome_path, output_path)

        console.print("\n[bold green]Metronome generated successfully![/bold green]")
        console.print(f"Output File: \n{output_path}")
        return output_path
    else:
        console.print("[bold red]Failed to generate metronome.[/bold red]")


@app.command("convert")
def convert(
        input_file: str = typer.Argument(..., help="Input JSON file path"),
        output_format: OutputFormat = typer.Option(OutputFormat.MIDI, help="Output format"),
        instrument: Instrument = typer.Option(Instrument.TRUMPET, help="Instrument for audio generation"),
        time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
        key: Key = typer.Option(Key.C_MAJOR, help="Key signature"),
        tempo: int = typer.Option(60, help="Tempo in BPM", min=40, max=200),
        output_dir: str = typer.Option("./output", help="Directory to save output files"),
        force_fallback: bool = typer.Option(False, help="Force using fallback audio generation instead of soundfonts"),
):
    """Convert a JSON exercise file to MIDI or MP3."""
    # Check if input file exists
    if not os.path.exists(input_file):
        console.print(f"[bold red]Input file not found: {input_file}[/bold red]")
        raise typer.Exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read JSON file
    try:
        with open(input_file, "r") as f:
            json_data = f.read()
        parsed = safe_parse_json(json_data)
        if not parsed:
            console.print("[bold red]Failed to parse JSON file.[/bold red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error reading JSON file: {e}[/bold red]")
        raise typer.Exit(1)

    # Clean the note data
    cleaned_parsed = []
    for note in parsed:
        cleaned_note = note.copy()
        if 'note' in cleaned_note:
            cleaned_note['note'] = clean_note_string(cleaned_note['note'])
        cleaned_parsed.append(cleaned_note)

    # Generate MIDI
    with console.status("[bold green]Converting to MIDI...[/bold green]"):
        # Calculate measures from JSON data
        total_units = sum(item["duration"] for item in cleaned_parsed)
        # Extract the actual time signature string from the enum
        time_sig_str = time_signature.value
        numerator, denominator = map(int, time_sig_str.split('/'))
        units_per_measure = numerator * (8 // denominator)
        measures = max(1, round(total_units / units_per_measure))

        # Extract string values from enums
        instrument_str = instrument.value
        key_str = key.value

        # Generate MIDI
        midi_obj = json_to_midi(cleaned_parsed, instrument_str, tempo, time_sig_str, measures, key=key_str)
    # Base filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Save outputs based on format
    output_files = []

    if output_format in [OutputFormat.MIDI, OutputFormat.ALL]:
        midi_path = os.path.join(output_dir, f"{base_name}.mid")
        midi_obj.save(midi_path)
        output_files.append(("MIDI", midi_path))

    if output_format in [OutputFormat.MP3, OutputFormat.ALL]:
        with console.status("[bold green]Converting to MP3...[/bold green]"):
            mp3_path, duration = midi_to_mp3(midi_obj, instrument_str, force_fallback=force_fallback)
            if mp3_path:
                new_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")
                import shutil
                shutil.copy(mp3_path, new_mp3_path)
                output_files.append(("MP3", new_mp3_path))
                console.print(
                    f"[bold green]MP3 conversion successful![/bold green] {'(Using fallback audio generation)' if force_fallback else ''}")
            else:
                console.print("[bold red]MP3 conversion failed.[/bold red]")
                console.print("Trying fallback audio generation...")
                mp3_path, duration = midi_to_mp3(midi_obj, instrument_str, force_fallback=True)
                if mp3_path:
                    new_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")
                    import shutil
                    shutil.copy(mp3_path, new_mp3_path)
                    output_files.append(("MP3", new_mp3_path))
                    console.print("[bold green]MP3 conversion successful using fallback audio generation![/bold green]")
                else:
                    console.print("[bold red]MP3 conversion failed even with fallback audio generation.[/bold red]")

    # Display results
    if output_files:
        console.print("\n[bold green]Conversion completed successfully![/bold green]")
        console.print("\n[bold]Output Files:[/bold]")
        for file_type, file_path in output_files:
            console.print(f"[bold]{file_type}:[/bold] {file_path}")
    else:
        console.print("[bold red]No output files were generated.[/bold red]")


@app.command("info")
def info():
    """Display information about available options."""
    console.print("[bold green]Adaptive Music Exercise Generator CLI[/bold green]")
    console.print("\n[bold]Available Instruments:[/bold]")
    for instrument in Instrument:
        console.print(f"- {instrument.value}")

    console.print("\n[bold]Difficulty Levels:[/bold]")
    for level in Level:
        console.print(f"- {level.value}")

    console.print("\n[bold]Key Signatures:[/bold]")
    for key in Key:
        console.print(f"- {key.value}")

    console.print("\n[bold]Time Signatures:[/bold]")
    for ts in TimeSignature:
        console.print(f"- {ts.value}")

    console.print("\n[bold]Practice Focus Options:[/bold]")
    for focus in PracticeFocus:
        console.print(f"- {focus.value}")

    console.print("\n[bold]Output Formats:[/bold]")
    for fmt in OutputFormat:
        console.print(f"- {fmt.value}")


if __name__ == "__main__":
    app()