#!/usr/bin/env python

"""
Adaptive Music Exercise Generator CLI (Strict Duration Enforcement)
==================================================================
A command-line interface for generating custom musical exercises with LLM.
This version uses the strict duration enforcement and fallback mechanism from the V2 backup.
JSON is used for the output.
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
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX")  # Use environment variable

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
# 3. Music theory helpers (note names ↔︎ MIDI numbers)
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
    match = re.match(r"([A-Ga-g][#b]?)(\d)", note)
    if not match:
        raise ValueError(f"Invalid note: {note}")
    pitch, octave = match.groups()
    pitch = pitch.upper().replace('b', 'B')
    if pitch not in NOTE_MAP:
        raise ValueError(f"Invalid pitch: {pitch}")
    return NOTE_MAP[pitch] + (int(octave) + 1) * 12


def midi_to_note_name(midi_num: int) -> str:
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
# 5. MIDI from scaled JSON (using integer durations) - UPDATED TO USE 8TH NOTES
# -----------------------------------------------------------------------------
def json_to_midi(json_data: list, instrument: str, tempo: int, time_signature: str, measures: int) -> MidiFile:
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack();
    mid.tracks.append(track)
    program = INSTRUMENT_PROGRAMS.get(instrument, 56)
    numerator, denominator = map(int, time_signature.split('/'))

    track.append(MetaMessage('time_signature', numerator=numerator,
                             denominator=denominator, time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    track.append(Message('program_change', program=program, time=0))

    for note_name, duration_units in json_data:
        try:
            note_num = note_name_to_midi(note_name)
            ticks = int(duration_units * TICKS_PER_8TH)  # UPDATED TO USE 8TH NOTES
            ticks = max(ticks, 1)
            velocity = random.randint(60, 100)
            track.append(Message('note_on', note=note_num, velocity=velocity, time=0))
            track.append(Message('note_off', note=note_num, velocity=velocity, time=ticks))
        except Exception as e:
            print(f"Error parsing note {note_name}: {e}")
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
            # Check if response is valid (not HTML/error)
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type or response.status_code != 200:
                print(f"Warning: Invalid response for {instrument} soundfont.")
                return None

            with open(sf2_path, "wb") as f:
                f.write(response.content)
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
        current_time = 0
        for track in midi_obj.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append((msg.note, current_time, msg.time))
                current_time += msg.time

        # Generate simple sine wave audio
        sample_rate = 44100
        total_ticks = sum(msg.time for track in midi_obj.tracks for msg in track)
        total_seconds = total_ticks * (0.5 / TICKS_PER_BEAT)  # Assuming 120 BPM
        audio_data = np.zeros(int(sample_rate * total_seconds) + sample_rate)  # Add 1 second buffer

        position = 0
        for note, start_time, duration in notes:
            # Convert MIDI note to frequency
            freq = 440 * (2 ** ((note - 69) / 12))
            # Convert ticks to seconds
            start_sec = start_time * (0.5 / TICKS_PER_BEAT)
            duration_sec = duration * (0.5 / TICKS_PER_BEAT)

            # Generate sine wave
            t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
            wave = 0.5 * np.sin(2 * np.pi * freq * t)

            # Add to audio data
            start_sample = int(start_sec * sample_rate)
            end_sample = start_sample + len(wave)
            if end_sample < len(audio_data):
                audio_data[start_sample:end_sample] += wave

        # Normalize and convert to 16-bit PCM
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = np.int16(audio_data / max_val * 32767)
        else:
            audio_data = np.int16(audio_data)

        # Save as WAV then convert to MP3
        wav_path = output_path.replace('.mp3', '.wav')
        wavfile.write(wav_path, sample_rate, audio_data)

        sound = AudioSegment.from_wav(wav_path)
        sound.export(output_path, format="mp3")

        # Move to static directory
        static_mp3_path = os.path.join('static', f'exercise_{uuid.uuid4().hex}.mp3')
        shutil.move(output_path, static_mp3_path)

        return static_mp3_path, sound.duration_seconds
    except ImportError as e:
        print(f"Required packages not available for fallback audio: {e}")
        return None, 0
    except Exception as e:
        print(f"Fallback audio generation failed: {e}")
        return None, 0


# -----------------------------------------------------------------------------
# 7. Prompt engineering for variety (using integer durations) - UPDATED DURATION SYSTEM
# -----------------------------------------------------------------------------
def get_fallback_exercise(instrument: str, level: str, key: str,
                          time_sig: str, measures: int) -> str:
    instrument_patterns = {
        "Trumpet": ["C4", "D4", "E4", "G4", "E4", "C4"],
        "Piano": ["C4", "E4", "G4", "C5", "G4", "E4"],
        "Violin": ["G4", "A4", "B4", "D5", "B4", "G4"],
        "Clarinet": ["E4", "F4", "G4", "Bb4", "G4", "E4"],
        "Flute": ["A4", "B4", "C5", "E5", "C5", "A4"],
    }
    pattern = instrument_patterns.get(instrument, instrument_patterns["Trumpet"])
    numerator, denominator = map(int, time_sig.split('/'))

    # UPDATED: Calculate units based on 8th notes
    units_per_measure = numerator * (8 // denominator)  # 8th notes per measure
    target_units = measures * units_per_measure
    notes, durs = [], []
    i = 0

    # Use quarter notes (2 units) as base duration
    while sum(durs) < target_units:
        notes.append(pattern[i % len(pattern)])
        # Use quarter notes (2 units) by default
        durs.append(2)
        i += 1

    # Adjust last duration to match total exactly
    total_units = sum(durs)
    if total_units > target_units:
        durs[-1] = durs[-1] - (total_units - target_units)
    elif total_units < target_units:
        durs[-1] = durs[-1] + (target_units - total_units)

    return json.dumps([[n, d] for n, d in zip(notes, durs)])


def get_style_based_on_level(level: str) -> str:
    styles = {
        "Beginner": ["simple", "legato", "stepwise"],
        "Intermediate": ["jazzy", "bluesy", "march-like", "syncopated"],
        "Advanced": ["technical", "chromatic", "fast arpeggios", "wide intervals"],
    }
    return random.choice(styles.get(level, ["technical"]))


def get_technique_based_on_level(level: str) -> str:
    techniques = {
        "Beginner": ["with long tones", "with simple rhythms", "focusing on tone"],
        "Intermediate": ["with slurs", "with accents", "using triplets"],
        "Advanced": ["with double tonguing", "with extreme registers", "complex rhythms"],
    }
    return random.choice(techniques.get(level, ["with slurs"]))


# -----------------------------------------------------------------------------
# 8. Mistral API: query, fallback on errors - UPDATED DURATION SYSTEM
# -----------------------------------------------------------------------------
def query_mistral(prompt: str, instrument: str, level: str, key: str,
                  time_sig: str, measures: int) -> str:
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
            f"{prompt} {duration_constraint} Output ONLY a JSON array of [note, duration] pairs."
        )
    else:
        style = get_style_based_on_level(level)
        technique = get_technique_based_on_level(level)
        user_prompt = (
            f"Create a {style} {instrument.lower()} exercise in {key} with {time_sig} time signature "
            f"{technique} for a {level.lower()} player. {duration_constraint} "
            "Output ONLY a JSON array of [note, duration] pairs following these rules: "
            "Use standard note names (e.g., \"Bb4\", \"F#5\"). Monophonic only. "
            "Durations: 1=8th, 2=quarter, 4=half, 8=whole. "  # UPDATED
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
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.replace("```json", "").replace("```", "").strip()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Mistral API: {e}")
        return get_fallback_exercise(instrument, level, key, time_sig, measures)
    except (KeyError, IndexError) as e:
        print(f"Error parsing Mistral API response: {e}")
        return get_fallback_exercise(instrument, level, key, time_sig, measures)


# -----------------------------------------------------------------------------
# 9. Robust JSON parsing for LLM outputs
# -----------------------------------------------------------------------------
def safe_parse_json(text: str) -> Optional[list]:
    try:
        text = text.replace("'", '"')
        match = re.search(r"\[(\s*\[.*?\]\s*,?)*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except json.JSONDecodeError as e:
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
                      measures: int, custom_prompt: str, mode: str, force_fallback: bool = False) -> Tuple[
    str, Optional[str], str, MidiFile, str, str, int]:
    try:
        prompt_to_use = custom_prompt if mode == "Exercise Prompt" else ""
        output = query_mistral(prompt_to_use, instrument, level, key, time_signature, measures)
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
                note_duration = max(1, target_units / len(notes))
                parsed = [[n, note_duration] for n in notes]
                # Adjust last note to match total duration
                total = sum(d for _, d in parsed)
                if total < target_units:
                    parsed[-1][1] += target_units - total
                elif total > target_units:
                    parsed[-1][1] -= total - target_units

        # Clean note strings to remove ornamentation
        for i, (note, dur) in enumerate(parsed):
            parsed[i][0] = clean_note_string(note)

        # Calculate total required 8th notes (UPDATED)
        numerator, denominator = map(int, time_signature.split('/'))
        units_per_measure = numerator * (8 // denominator)
        total_units = measures * units_per_measure

        # Strict scaling
        parsed_scaled = scale_json_durations(parsed, total_units)

        # Calculate total duration units
        total_duration = sum(d for _, d in parsed_scaled)

        # Generate MIDI and audio
        midi = json_to_midi(parsed_scaled, instrument, tempo, time_signature, measures)
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
        for note, dur in parsed:
            try:
                midi_note = note_name_to_midi(note)
                notes.append(midi_note)
                durations.append(dur)
            except ValueError:
                notes.append(60)  # Default to middle C if parsing fails
                durations.append(dur)

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
        if notes:
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
    except ImportError as e:
        print(f"Visualization requires matplotlib: {e}")
        return None
    except ValueError as e:
        print(f"Error in visualization data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating visualization: {e}")
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

        # Save and convert to audio using fallback method
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as mid_file:
            mid.save(mid_file.name)
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
        try:
            os.remove(mid_file.name)
        except FileNotFoundError:
            pass

        return static_mp3_path
    except ImportError as e:
        print(f"Metronome requires pydub: {e}")
        return None
    except Exception as e:
        print(f"Error creating metronome: {e}")
        return None


# -----------------------------------------------------------------------------
# 14. Define enums for CLI options
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


class OutputFormat(str, Enum):
    JSON = "json"
    MIDI = "midi"
    MP3 = "mp3"
    ALL = "all"


# -----------------------------------------------------------------------------
# 15. CLI Commands
# -----------------------------------------------------------------------------
@app.command("generate")
def generate(
        instrument: Instrument = typer.Option(Instrument.TRUMPET, help="Instrument to generate exercise for"),
        level: Level = typer.Option(Level.INTERMEDIATE, help="Difficulty level"),
        key: Key = typer.Option(Key.C_MAJOR, help="Key signature"),
        time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
        measures: int = typer.Option(4, help="Number of measures", min=1, max=16),
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

        json_data, mp3_path, tempo_str, midi_obj, duration, time_sig, total_duration = generate_exercise(
            instrument_str, level_str, key_str, tempo, time_sig_str,
            measures, custom_prompt or "", mode, force_fallback
        )

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
            viz_path = create_visualization(json_data, time_sig_str)
            if viz_path:
                viz_output = os.path.join(output_dir, f"{base_filename}_viz.png")
                import shutil
                shutil.copy(viz_path, viz_output)
                output_files.append(("Visualization", viz_output))
        except Exception as e:
            console.print(f"[bold red]Error generating visualization: {e}[/bold red]")

    # Display results
    console.print("\n[bold green]Exercise generated successfully![/bold green]")
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
        input_file: str = typer.Option(..., help="Input JSON file path"),
        output_format: OutputFormat = typer.Option(OutputFormat.MIDI, help="Output format"),
        instrument: Instrument = typer.Option(Instrument.TRUMPET, help="Instrument for audio generation"),
        time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
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
    for note, dur in parsed:
        cleaned_note = clean_note_string(note)
        cleaned_parsed.append([cleaned_note, dur])

    # Generate MIDI
    with console.status("[bold green]Converting to MIDI...[/bold green]"):
        # Calculate measures from JSON data
        total_units = sum(d for _, d in cleaned_parsed)
        # Extract the actual time signature string from the enum
        time_sig_str = time_signature.value
        numerator, denominator = map(int, time_sig_str.split('/'))
        units_per_measure = numerator * (8 // denominator)
        measures = max(1, round(total_units / units_per_measure))

        # Extract string values from enums
        instrument_str = instrument.value

        # Generate MIDI
        midi_obj = json_to_midi(cleaned_parsed, instrument_str, tempo, time_sig_str, measures)
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

    console.print("\n[bold]Output Formats:[/bold]")
    for fmt in OutputFormat:
        console.print(f"- {fmt.value}")


if __name__ == "__main__":
    app()