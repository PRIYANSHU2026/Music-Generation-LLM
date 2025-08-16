"""
Adaptive Music Exercise Generator (Strict Duration Enforcement)
==============================================================
Generates custom musical exercises with LLM, perfectly fit to user-specified number of measures
AND time signature, guaranteeing exact durations in MIDI and in the UI!
Major updates:
- Changed base duration unit from 16th notes to 8th notes (1 unit = 8th note)
- Updated all calculations and prompts to use new duration system
- Duration sum display now shows total in 8th notes
- Maintained all original functionality
- Added cumulative duration tracking
- Enforced JSON output format with note, duration, cumulative_duration
- Enhanced rest handling and JSON parsing
- Fixed JSON parsing errors for 8-measure exercises
- Added robust error handling for MIDI generation
"""

# -----------------------------------------------------------------------------
# 1. Runtime-time package installation (for fresh containers/Colab/etc)
# -----------------------------------------------------------------------------
import sys
import subprocess
from typing import Dict, Optional, Tuple, List

def install(packages: List[str]):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install([
    "mido", "midi2audio", "pydub", "gradio",
    "requests", "numpy", "matplotlib", "librosa", "scipy",
    "uuid", "datetime"
])

# -----------------------------------------------------------------------------
# 2. Static imports
# -----------------------------------------------------------------------------
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
import gradio as gr
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
# 3. Configuration & constants (UPDATED TO USE 8TH NOTES)
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
# 7. MIDI → Audio (MP3) helpers
# -----------------------------------------------------------------------------

def cleanup_invalid_soundfonts():
    """Check and remove any invalid soundfont files"""
    if not os.path.exists("soundfonts"):
        os.makedirs("soundfonts", exist_ok=True)
        return
        
    for instrument in SOUNDFONT_URLS.keys():
        sf2_path = f"soundfonts/{instrument}.sf2"
        if os.path.exists(sf2_path):
            # Check if file is valid (not HTML and has reasonable size)
            if os.path.getsize(sf2_path) < 10000 or "HTML" in open(sf2_path, 'r', errors='ignore').read(100):
                print(f"Removing invalid soundfont: {sf2_path}")
                os.remove(sf2_path)

# Run cleanup on startup
cleanup_invalid_soundfonts()
def get_soundfont(instrument: str) -> str:
    os.makedirs("soundfonts", exist_ok=True)
    sf2_path = f"soundfonts/{instrument}.sf2"
    if not os.path.exists(sf2_path) or os.path.getsize(sf2_path) < 10000 or "HTML" in open(sf2_path, 'r', errors='ignore').read(100):
        url = SOUNDFONT_URLS.get(instrument, SOUNDFONT_URLS["Trumpet"])
        print(f"Downloading SoundFont for {instrument}…")
        
        # Try to download using requests
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(sf2_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded soundfont for {instrument} using requests")
            else:
                print(f"Failed to download soundfont: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error downloading soundfont: {str(e)}")
            
        # Verify the downloaded file is not HTML
        if not os.path.exists(sf2_path) or os.path.getsize(sf2_path) < 10000 or "HTML" in open(sf2_path, 'r', errors='ignore').read(100):
            print(f"Warning: Downloaded file for {instrument} appears to be HTML or invalid.")
            # Use a fallback local soundfont if available
            fallback_paths = [
                "/usr/share/sounds/sf2/FluidR3_GM.sf2",
                "/usr/share/sounds/sf2/default.sf2",
                "/usr/local/share/soundfonts/default.sf2"
            ]
            for fallback_path in fallback_paths:
                if os.path.exists(fallback_path):
                    print(f"Using fallback soundfont: {fallback_path}")
                    return fallback_path
            
            # If no fallback found, create a minimal soundfont in memory
            print("No valid soundfont found. Using minimal synthesized sounds.")
            return ""  # Empty string signals to use synthesized fallback
    return sf2_path

def midi_to_mp3(midi_obj: MidiFile, instrument: str = "Trumpet", force_fallback: bool = False) -> Tuple[str, float]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as mid_file:
        midi_obj.save(mid_file.name)
        wav_path = mid_file.name.replace(".mid", ".wav")
        mp3_path = mid_file.name.replace(".mid", ".mp3")
    sf2_path = get_soundfont(instrument)
    
    # If soundfont path is empty or force_fallback is True, skip FluidSynth and go directly to fallback
    conversion_success = False
    
    if sf2_path and not force_fallback:  # Only try FluidSynth if we have a valid soundfont and not forcing fallback
        # Check if fluidsynth is available
        fluidsynth_available = False
        try:
            result = sp.run(['which', 'fluidsynth'], capture_output=True, text=True)
            fluidsynth_available = result.returncode == 0 and result.stdout.strip() != ''
        except Exception:
            fluidsynth_available = False
        
        # Try using fluidsynth command line first if available
        if fluidsynth_available:
            try:
                print(f"Using fluidsynth command line with soundfont: {sf2_path}")
                result = sp.run([
                    'fluidsynth', '-ni', sf2_path, mid_file.name,
                    '-F', wav_path, '-r', '44100', '-g', '1.0'
                ], check=True, capture_output=True, text=True)
                conversion_success = os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000
                if not conversion_success:
                    print(f"FluidSynth command failed or produced empty file: {result.stdout}\n{result.stderr}")
            except Exception as e:
                print(f"FluidSynth command error: {str(e)}")
        
        # If fluidsynth command failed or not available, try pyFluidSynth
        if not conversion_success and os.path.exists(sf2_path) and os.path.getsize(sf2_path) > 10000:
            try:
                print(f"Using pyFluidSynth with soundfont: {sf2_path}")
                fs = FluidSynth(sf2_path, sample_rate=44100, gain=1.0)
                fs.midi_to_audio(mid_file.name, wav_path)
                conversion_success = os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000
                if not conversion_success:
                    print("pyFluidSynth failed to produce a valid WAV file")
            except Exception as e:
                print(f"pyFluidSynth error: {str(e)}")
    else:
        if force_fallback:
            print("Forcing fallback audio generation as requested")
        else:
            print("No valid soundfont available, skipping FluidSynth methods")
    
    # If both FluidSynth methods failed, create a more sophisticated sine wave audio as fallback
    if not conversion_success:
        try:
            print("Using fallback sine wave audio generation")
            # Parse MIDI file to get notes with their durations and velocities
            mid = mido.MidiFile(mid_file.name)
            notes = []
            current_notes = {}
            current_time = 0
            
            for track in mid.tracks:
                for msg in track:
                    current_time += msg.time
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # Note starts
                        current_notes[msg.note] = (current_time, msg.velocity)
                    elif (msg.type == 'note_off' or 
                          (msg.type == 'note_on' and msg.velocity == 0)) and \
                          msg.note in current_notes:
                        # Note ends
                        start_time, velocity = current_notes[msg.note]
                        duration = current_time - start_time
                        notes.append((msg.note, start_time, duration, velocity))
                        del current_notes[msg.note]
            
            # Sort notes by start time
            notes.sort(key=lambda x: x[1])
            
            if not notes:
                # If no notes were found, create a simple beep
                sample_rate = 44100
                duration = 1.0  # seconds
                t = np.linspace(0, duration, int(duration * sample_rate), False)
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz = A4
            else:
                # Calculate total duration based on the last note end time
                total_duration = max(note[1] + note[2] for note in notes) + 0.5  # Add a bit of padding
                
                # Generate audio
                sample_rate = 44100
                t = np.linspace(0, total_duration, int(total_duration * sample_rate), False)
                audio = np.zeros_like(t)
                
                # Add sine waves for each note with envelope
                for note, start_time, note_duration, velocity in notes:
                    # Convert MIDI note to frequency
                    freq = 440.0 * (2.0 ** ((note - 69) / 12.0))
                    
                    # Calculate sample indices
                    start_idx = int(start_time * sample_rate)
                    end_idx = int((start_time + note_duration) * sample_rate)
                    
                    if end_idx <= len(t):
                        # Create time array for this note
                        t_note = t[start_idx:end_idx] - t[start_idx]
                        
                        # Create envelope (attack, decay, sustain, release)
                        attack = 0.05  # seconds
                        decay = 0.1    # seconds
                        release = 0.1  # seconds
                        sustain_level = 0.7
                        
                        envelope = np.ones(len(t_note))
                        attack_samples = int(attack * sample_rate)
                        decay_samples = int(decay * sample_rate)
                        release_samples = int(release * sample_rate)
                        
                        # Apply attack
                        if attack_samples > 0:
                            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
                        
                        # Apply decay to sustain level
                        if decay_samples > 0 and attack_samples < len(envelope):
                            decay_end = min(attack_samples + decay_samples, len(envelope))
                            envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_end - attack_samples)
                        
                        # Apply release
                        if release_samples > 0 and len(envelope) > release_samples:
                            envelope[-release_samples:] = np.linspace(envelope[-release_samples], 0, release_samples)
                        
                        # Generate note with harmonics for richer sound
                        note_signal = np.zeros_like(t_note)
                        
                        # Fundamental frequency
                        note_signal += 0.7 * np.sin(2 * np.pi * freq * t_note)
                        
                        # Add harmonics (overtones)
                        note_signal += 0.2 * np.sin(2 * np.pi * (2 * freq) * t_note)  # 1st harmonic (octave)
                        note_signal += 0.05 * np.sin(2 * np.pi * (3 * freq) * t_note)  # 2nd harmonic
                        note_signal += 0.02 * np.sin(2 * np.pi * (4 * freq) * t_note)  # 3rd harmonic
                        
                        # Apply envelope and velocity
                        note_signal *= envelope * (velocity / 127.0)
                        
                        # Add to audio
                        audio[start_idx:end_idx] += note_signal
            
            # Normalize audio to prevent clipping
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude * 0.9
            
            # Apply a slight reverb effect
            reverb_length = int(0.1 * sample_rate)  # 100ms reverb
            if reverb_length > 0:
                reverb = np.exp(-np.linspace(0, 5, reverb_length))
                audio_with_reverb = np.zeros(len(audio) + reverb_length)
                audio_with_reverb[:len(audio)] = audio
                
                # Convolve with reverb impulse response
                for i in range(len(audio)):
                    audio_with_reverb[i:i+reverb_length] += audio[i] * reverb * 0.3
                
                # Trim to original length
                audio = audio_with_reverb[:len(audio)]
                
                # Normalize again after reverb
                max_amplitude = np.max(np.abs(audio))
                if max_amplitude > 0:
                    audio = audio / max_amplitude * 0.9
            
            # Write to WAV file
            import scipy.io.wavfile as wavfile
            wavfile.write(wav_path, sample_rate, (audio * 32767).astype(np.int16))
            conversion_success = os.path.exists(wav_path) and os.path.getsize(wav_path) > 1000
        except Exception as e:
            print(f"Fallback audio generation error: {str(e)}")
            # Create a very simple beep as last resort
            try:
                sample_rate = 44100
                duration = 2.0  # seconds
                t = np.linspace(0, duration, int(duration * sample_rate), False)
                audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz = A4
                import scipy.io.wavfile as wavfile
                wavfile.write(wav_path, sample_rate, (audio * 32767).astype(np.int16))
                conversion_success = True
            except Exception as e2:
                print(f"Even simple beep generation failed: {str(e2)}")
                conversion_success = False
    try:
        sound = AudioSegment.from_wav(wav_path)
        if instrument == "Trumpet":
            sound = sound.high_pass_filter(200)
        elif instrument == "Violin":
            sound = sound.low_pass_filter(5000)
        sound.export(mp3_path, format="mp3")
        static_mp3_path = os.path.join('static', os.path.basename(mp3_path))
        shutil.move(mp3_path, static_mp3_path)
        return static_mp3_path, sound.duration_seconds
    finally:
        for f in [mid_file.name, wav_path]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

# -----------------------------------------------------------------------------
# 8. Prompt engineering for variety (using integer durations) - UPDATED DURATION SYSTEM
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
        "Advanced": ["technical", "chromatic", "fast arpeggios", "wide intervals", "virtuosic", "complex", "contemporary"],
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
# 9. Mistral API: query, fallback on errors - UPDATED DURATION SYSTEM
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
        return content.replace("```json","").replace("```","").strip()
    except Exception as e:
        print(f"Error querying Mistral API: {e}")
        return get_fallback_exercise(instrument, level, key, time_sig, measures)

# -----------------------------------------------------------------------------
# 10. Robust JSON parsing for LLM outputs - ENHANCED PARSING
# -----------------------------------------------------------------------------
def safe_parse_json(text: str) -> Optional[list]:
    try:
        text = text.strip().replace("'", '"')
        
        # Find JSON array in the text
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx == -1 or end_idx == -1:
            return None
            
        json_str = text[start_idx:end_idx+1]
        
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
# 11. Main orchestration: talk to API, *scale durations*, build MIDI, UI values - UPDATED
# -----------------------------------------------------------------------------
def generate_exercise(instrument: str, level: str, key: str, tempo: int, time_signature: str,
                      measures: int, custom_prompt: str, mode: str, difficulty_modifier: int = 0, 
                      practice_focus: str = "Balanced", force_fallback: bool = False) -> Tuple[str, Optional[str], str, MidiFile, str, str, int]:
    try:
        prompt_to_use = custom_prompt if mode == "Exercise Prompt" else ""
        output = query_mistral(prompt_to_use, instrument, level, key, time_signature, measures, difficulty_modifier, practice_focus)
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
        mp3_path, real_duration = midi_to_mp3(midi, instrument, force_fallback=force_fallback)
        output_json_str = json.dumps(parsed_scaled, indent=2)
        return output_json_str, mp3_path, str(tempo), midi, f"{real_duration:.2f} seconds", time_signature, total_duration
    except Exception as e:
        return f"Error: {str(e)}", None, str(tempo), None, "0", time_signature, 0

# -----------------------------------------------------------------------------
# 12. Simple AI chat assistant (optional, shares LLM)
# -----------------------------------------------------------------------------
def handle_chat(message: str, history: List, instrument: str, level: str):
    if not message.strip():
        return "", history
    messages = [{"role": "system", "content": f"You are a {instrument} teacher for {level} students."}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "mistral-medium", "messages": messages, "temperature": 0.7, "max_tokens": 500}
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        history.append((message, content))
        return "", history
    except Exception as e:
        history.append((message, f"Error: {str(e)}"))
        return "", history

# -----------------------------------------------------------------------------
# 13. New features: Visualization, Metronome, and Exercise Library
# -----------------------------------------------------------------------------

# Visualization function to create a piano roll representation of the exercise
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
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate time positions
        time_positions = [0]
        for dur in durations[:-1]:
            time_positions.append(time_positions[-1] + dur)
        
        # Plot notes as rectangles
        for i, (note, dur, pos) in enumerate(zip(notes, durations, time_positions)):
            if note is not None:  # Skip rests
                rect = plt.Rectangle((pos, note-0.4), dur, 0.8, color='blue', alpha=0.7)
                ax.add_patch(rect)
                # Add note name
                ax.text(pos + dur/2, note+0.5, midi_to_note_name(note), 
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
        ax.set_ylim(min(notes) - 5 if None not in notes else 55, 
                   max(notes) + 5 if None not in notes else 75)
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

# VexFlow music notation visualization function
def create_vexflow_notation(json_data, time_sig, key_sig):
    # Helper function to convert duration units to VexFlow duration
    def durationToVex(units):
        if units == 1:
            return "8"
        elif units == 2:
            return "4"
        elif units == 3:
            return "4d"
        elif units == 4:
            return "2"
        elif units == 6:
            return "2d"
        elif units == 8:
            return "1"
        else:
            return "8"
    
    if not json_data or "Error" in json_data:
        return None
        
    try:
        parsed = json.loads(json_data)
        if not isinstance(parsed, list) or len(parsed) == 0:
            return None
        
        # Create HTML content with VexFlow notation
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Music Notation</title>
            <script src="https://cdn.jsdelivr.net/npm/vexflow@4.2.2/build/cjs/vexflow.js"></script>
            <style>
                #output {{width: 100%; overflow: auto;}}
                body {{font-family: Arial, sans-serif;}}
                h2 {{color: #333;}}
            </style>
        </head>
        <body>
            <h2>Exercise in {key_sig}, {time_sig}</h2>
            <div id="output"></div>
            <script>
                const {{Factory, EasyScore, System}} = Vex.Flow;
                
                // Create VexFlow factory and context
                const vf = new Factory({{renderer: {{elementId: 'output', width: 1200, height: 200}}}});
                const score = vf.EasyScore();
                const system = vf.System();
                
                // Parse notes from JSON
                const jsonData = {json.dumps(parsed)};
                
                // Convert to VexFlow notation
                let vexNotes = [];
                let currentMeasure = [];
                let currentDuration = 0;
                const timeSignature = "{time_sig}";
                const [numerator, denominator] = timeSignature.split('/').map(Number);
                const unitsPerMeasure = numerator * (8 / denominator);
                
                // Helper function to convert duration units to VexFlow duration
                function durationToVex(units) {{
                    if (units === 1) return "8";
                    if (units === 2) return "4";
                    if (units === 3) return "4d";
                    if (units === 4) return "2";
                    if (units === 6) return "2d";
                    if (units === 8) return "1";
                    return "8";
                }}
                
                // Process notes
                jsonData.forEach(item => {{
                    const noteName = item.note;
                    const duration = item.duration;
                    
                    // Skip invalid notes
                    if (!noteName || duration <= 0) return;
                    
                    // Handle rests
                    const isRest = /rest|r|p/i.test(noteName);
                    let vexNote;
                    
                    if (isRest) {{
                        vexNote = `B4/${{durationToVex(duration)}}/r`;
                    }} else {{
                        // Convert scientific notation to VexFlow format
                        // VexFlow uses lowercase for note names
                        const noteRegex = /([A-Ga-g][#b]?)(\d)/;
                        const match = noteName.match(noteRegex);
                        if (match) {{
                            const [_, pitch, octave] = match;
                            vexNote = `${{pitch.toLowerCase()}}${{octave}}/${{durationToVex(duration)}}`;
                        }} else {{
                            // Default if parsing fails
                            vexNote = `c4/${{durationToVex(duration)}}`;
                        }}
                    }}
                    
                    currentMeasure.push(vexNote);
                    currentDuration += duration;
                    
                    // Check if measure is complete
                    if (currentDuration >= unitsPerMeasure) {{
                        vexNotes.push(currentMeasure);
                        currentMeasure = [];
                        currentDuration = 0;
                    }}
                }});
                
                // Add any remaining notes
                if (currentMeasure.length > 0) {{
                    vexNotes.push(currentMeasure);
                }}
                
                // Create staves and add notes
                const staves = [];
                const measuresPerLine = 4;
                
                for (let i = 0; i < vexNotes.length; i += measuresPerLine) {{
                    const lineStaves = [];
                    const lineNotes = vexNotes.slice(i, i + measuresPerLine);
                    
                    // Create a new system for each line
                    const lineSystem = vf.System({{width: 1100}});
                    
                    // Add staves for each measure in the line
                    lineNotes.forEach((measure, index) => {{
                        const stave = lineSystem.addStave({{
                            voices: [
                                score.voice(score.notes(measure.join(', ')))
                            ]
                        }});
                        
                        // Add time signature and key to first measure of first line
                        if (i === 0 && index === 0) {{
                            stave.addTimeSignature(timeSignature);
                            stave.addKeySignature("{key_sig.split()[0]}");
                        }}
                    }});
                    
                    lineSystem.addConnector("singleRight");
                    staves.push(lineSystem);
                }}
                
                // Format and draw
                vf.draw();
            </script>
        </body>
        </html>
        '''
        
        # For Hugging Face environment, return HTML content directly
        # Also save a copy to file for compatibility with existing code
        try:
            html_path = os.path.join('static', f'notation_{uuid.uuid4().hex}.html')
            with open(html_path, 'w') as f:
                f.write(html_content)
        except Exception as file_error:
            print(f"Warning: Could not save notation file: {file_error}")
        
        # Return HTML content directly
        return html_content
    except Exception as e:
        print(f"Error creating VexFlow notation: {e}")
        return "<p>Failed to generate music notation. Error: " + str(e) + "</p>"

# Metronome function
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
        
        # Use piano soundfont for metronome
        sf2_path = get_soundfont("Piano")
        try:
            sp.run([
                'fluidsynth', '-ni', sf2_path, mid_file.name,
                '-F', wav_path, '-r', '44100', '-g', '1.0'
            ], check=True, capture_output=True)
        except Exception:
            fs = FluidSynth(sf2_path, sample_rate=44100, gain=1.0)
            fs.midi_to_audio(mid_file.name, wav_path)
        
        # Convert to MP3
        sound = AudioSegment.from_wav(wav_path)
        sound.export(mp3_path, format="mp3")
        
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



# Function to calculate difficulty rating
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
        jumps = [abs(notes[i] - notes[i-1]) for i in range(1, len(notes))]
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
# 14. Module exports for CLI usage
# -----------------------------------------------------------------------------
# Export all necessary functions for CLI usage

        mode = gr.Radio(["Exercise Parameters","Exercise Prompt"], value="Exercise Parameters", label="Generation Mode")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(visible=True) as params_group:
                    gr.Markdown("### Exercise Parameters")
                    instrument = gr.Dropdown([
                        "Trumpet", "Piano", "Violin", "Clarinet", "Flute",
                    ], value="Trumpet", label="Instrument")
                    level = gr.Radio([
                        "Beginner", "Intermediate", "Advanced",
                    ], value="Intermediate", label="Difficulty Level")
                    difficulty_modifier = gr.Slider(minimum=-2, maximum=2, value=0, step=1, 
                                                  label="Difficulty Modifier", 
                                                  info="Fine-tune the difficulty: -2 (easier) to +2 (harder)")
                    practice_focus = gr.Dropdown([
                        "Balanced", "Rhythmic Focus", "Melodic Focus", "Technical Focus", "Expressive Focus"
                    ], value="Balanced", label="Practice Focus")
                    key = gr.Dropdown([
                        "C Major", "G Major", "D Major", "F Major", "Bb Major", "A Minor", "E Minor",
                    ], value="C Major", label="Key Signature")
                    time_signature = gr.Dropdown(["3/4", "4/4"], value="4/4", label="Time Signature")
                    measures = gr.Radio([4, 8, 12, 16], value=4, label="Length (measures)")
                with gr.Group(visible=False) as prompt_group:
                    gr.Markdown("### Exercise Prompt")
                    custom_prompt = gr.Textbox("", label="Enter your custom prompt", lines=3)
                    measures_prompt = gr.Radio([4, 8, 12, 16], value=4, label="Length (measures)")
                generate_btn = gr.Button("Generate Exercise", variant="primary")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Exercise Player"):
                        audio_output = gr.Audio(label="Generated Exercise", autoplay=True, type="filepath")
                        with gr.Row():
                            bpm_display = gr.Textbox(label="Tempo (BPM)")
                            time_sig_display = gr.Textbox(label="Time Signature")
                            duration_display = gr.Textbox(label="Audio Duration", interactive=False)
                        with gr.Row():
                            difficulty_rating = gr.Number(label="Difficulty Rating (1-10)", interactive=False, precision=1)
                        
                        # Metronome section
                        gr.Markdown("### Metronome")
                        with gr.Row():
                            metronome_tempo = gr.Slider(minimum=40, maximum=200, value=60, step=1, label="Metronome Tempo")
                            metronome_btn = gr.Button("Generate Metronome", variant="secondary")
                        metronome_audio = gr.Audio(label="Metronome", type="filepath")
                        
                    with gr.TabItem("Exercise Data"):
                        json_output = gr.Code(label="JSON Representation", language="json")
                        duration_sum = gr.Number(
                            label="Total Duration Units (8th notes)",
                            interactive=False,
                            precision=0
                        )
                        
                    with gr.TabItem("Visualization"):
                        visualization_output = gr.Image(label="Exercise Visualization", type="filepath")
                        visualize_btn = gr.Button("Generate Visualization", variant="secondary")
                        
                    with gr.TabItem("Music Notation"):
                        notation_html = gr.HTML(label="Music Notation")
                        notation_btn = gr.Button("Generate Music Notation", variant="secondary")
                        
                    with gr.TabItem("MIDI Export"):
                        midi_output = gr.File(label="MIDI File")
                        download_midi = gr.Button("Generate MIDI File")
                        

                        
                    with gr.TabItem("AI Chat"):
                        chat_history = gr.Chatbot(label="Practice Assistant", height=400)
                        chat_message = gr.Textbox(label="Ask the AI anything about your practice")
                        send_chat_btn = gr.Button("Send")
        # Toggle UI groups
        mode.change(
            fn=lambda m: {
                params_group: gr.update(visible=(m == "Exercise Parameters")),
                prompt_group: gr.update(visible=(m == "Exercise Prompt")),
            },
            inputs=[mode], outputs=[params_group, prompt_group]
        )
        def generate_caller(mode_val, instrument_val, level_val, key_val,
                    time_sig_val, measures_val, prompt_val, measures_prompt_val,
                    difficulty_modifier_val, practice_focus_val):
            real_measures = measures_prompt_val if mode_val == "Exercise Prompt" else measures_val
            fixed_tempo = 60
            json_data, mp3_path, tempo, midi, duration, time_sig, total_duration = generate_exercise(
                instrument_val, level_val, key_val, fixed_tempo, time_sig_val,
                real_measures, prompt_val, mode_val, difficulty_modifier_val, practice_focus_val
            )
            
            # Calculate difficulty rating
            rating = calculate_difficulty_rating(json_data, level_val, difficulty_modifier_val, practice_focus_val)
            
            # Generate visualization
            viz_path = create_visualization(json_data, time_sig_val)
            
            # Generate music notation
            html_content = create_vexflow_notation(json_data, time_sig_val, key_val)
            if not html_content:
                html_content = ""
            
            return json_data, mp3_path, tempo, midi, duration, time_sig, total_duration, rating, viz_path, mp3_path, html_content
            
        generate_btn.click(
            fn=generate_caller,
            inputs=[mode, instrument, level, key, time_signature, measures, custom_prompt, measures_prompt, 
                   difficulty_modifier, practice_focus],
            outputs=[json_output, audio_output, bpm_display, current_midi, duration_display, 
                    time_sig_display, duration_sum, difficulty_rating, visualization_output, current_audio_path, notation_html]
        )
        
        # Visualization button
        visualize_btn.click(
            fn=create_visualization,
            inputs=[json_output, time_signature],
            outputs=[visualization_output]
        )
        
        # Music Notation button
        def display_notation(json_data, time_sig, key_val):
            html_content = create_vexflow_notation(json_data, time_sig, key_val)
            if html_content:
                return html_content
            return "<p>Failed to generate music notation.</p>"
            
        notation_btn.click(
            fn=display_notation,
            inputs=[json_output, time_signature, key],
            outputs=[notation_html]
        )
        
        # Metronome generation
        def generate_metronome(tempo, time_sig, measures_val):
            return create_metronome_audio(tempo, time_sig, measures_val)
            
        metronome_btn.click(
            fn=generate_metronome,
            inputs=[metronome_tempo, time_signature, measures],
            outputs=[metronome_audio]
        )
        

        

        
        def save_midi(json_data, instr, time_sig, key_sig="C Major"):
            try:
                if not json_data or "Error" in json_data:
                    return None
                    
                parsed = json.loads(json_data)
                
                # Validate JSON structure
                if not isinstance(parsed, list):
                    return None
                    
                old_format = []
                for item in parsed:
                    if isinstance(item, dict) and "note" in item and "duration" in item:
                        old_format.append([item["note"], item["duration"]])
                
                if not old_format:
                    return None
                    
                # Calculate total units
                total_units = sum(d[1] for d in old_format)
                numerator, denominator = map(int, time_sig.split('/'))
                units_per_measure = numerator * (8 // denominator)
                measures_est = max(1, round(total_units / units_per_measure))
                
                # Generate MIDI
                cumulative = 0
                scaled_new = []
                for note, dur in old_format:
                    cumulative += dur
                    scaled_new.append({
                        "note": note,
                        "duration": dur,
                        "cumulative_duration": cumulative
                    })
                    
                midi_obj = json_to_midi(scaled_new, instr, 60, time_sig, measures_est, key=key_sig)
                midi_path = os.path.join("static", "exercise.mid")
                midi_obj.save(midi_path)
                return midi_path
            except Exception as e:
                print(f"Error saving MIDI: {e}")
                return None
                
        download_midi.click(
            fn=save_midi,
            inputs=[json_output, instrument, time_signature, key],
            outputs=[midi_output],
        )
        send_chat_btn.click(
            fn=handle_chat,
            inputs=[chat_message, chat_history, instrument, level],
            outputs=[chat_message, chat_history],
        )
    return demo

# -----------------------------------------------------------------------------
# 14. Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module provides music exercise generation functionality.")
    print("Please use the CLI interface by running: python cli.py --help")