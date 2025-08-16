import streamlit as st
import requests
import json
import re
import os
import sys
import subprocess
from io import BytesIO
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
import scipy.io.wavfile
from pydub import AudioSegment

# Import functions from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import json_to_midi, note_name_to_midi, is_rest

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def install(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing missing package: {package}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except Exception as e:
                print(f"Error installing {package}: {e}")
                st.warning(f"Failed to install {package}. Please install it manually with: pip install {package}")
                continue

install(["streamlit", "requests", "mido", "pydub", "numpy", "scipy"])

# Define constants
OLLAMA_URL = "http://localhost:11434"  # Change this to your Ollama server URL if different

SYSTEM_PROMPT = """
You are a music assistant that generates monophonic trumpet exercises using a specific JSON format.

Each exercise must be output in the following structure:

[
  ["C4", 0.5],
  ["D4", 0.25],
  ["E4", 1]
]

Format rules:
- Each element is a pair: [note_name, duration]
- Use standard English note names (e.g., "Bb3", "F#4", "C5")
- The notes must be monophonic (one at a time)
- Use only notes playable on a standard Bb trumpet (range: E3 to C6)
- Duration values in beats: 0.125 = eighth, 0.25 = quarter, 0.5 = half, 1 = whole, 2 = breve

Only output the JSON array. Do not include explanation, markdown, or other formatting.
""".strip()

def get_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        return [m['name'] for m in response.json().get("models", [])]
    except Exception as e:
        st.error(f"Error connecting to Ollama server: {e}")
        return []

def query_model(model_name, user_prompt):
    st.info(f"Querying model '{model_name}' with prompt: {user_prompt}")

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        if "message" not in response_json or "content" not in response_json["message"]:
            raise Exception("Unexpected response format")

        response_text = response_json["message"]["content"]
        return response_text
    except Exception as e:
        st.error(f"Error querying model '{model_name}': {e}")
        raise e

def run_models(models, prompt):
    results = {}
    for model in models:
        try:
            with st.spinner(f"Running model {model}..."):
                results[model] = query_model(model, prompt)
        except Exception as e:
            results[model] = e
    return results

def safe_parse_json(text):
    try:
        # Try to extract JSON array if embedded in other text
        match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception as e:
        st.error(f"JSON parsing error: {e}\nRaw text: {text}")
        return None

# Enhanced midi_to_mp3 function with improved fallback audio generation
def midi_to_mp3(midi_obj, instrument_program=56, force_fallback=False):  # 56 is the MIDI program for trumpet
    # Save MIDI to a temporary file
    with BytesIO() as mid_buffer:
        midi_obj.save(file=mid_buffer)
        mid_buffer.seek(0)
        mid_data = mid_buffer.read()
    
    # Parse MIDI to extract notes
    midi_obj.save(file=BytesIO())
    notes = []
    current_notes = {}
    current_time = 0
    
    for track in midi_obj.tracks:
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
    
    # Generate audio
    sample_rate = 44100
    
    if not notes:
        # If no notes were found, create a simple beep
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz = A4
    else:
        # Calculate total duration based on the last note end time
        total_duration = max(note[1] + note[2] for note in notes) + 0.5  # Add a bit of padding
        
        # Generate audio
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
    
    # Convert to MP3 using pydub
    with BytesIO() as wav_buffer:
        scipy.io.wavfile.write(wav_buffer, sample_rate, (audio * 32767).astype(np.int16))
        wav_buffer.seek(0)
        sound = AudioSegment.from_wav(wav_buffer)
        
        # Apply trumpet-specific processing
        # High-pass filter to remove low frequencies
        sound = sound.high_pass_filter(300)
        
        # Add some brightness with a slight boost in high frequencies
        sound = sound.high_shelf_filter(cutoff=2000, gain=3.0)
        
        # Normalize volume
        sound = sound.normalize(headroom=0.1)
        
        # Export to MP3
        mp3_buffer = BytesIO()
        sound.export(mp3_buffer, format="mp3", bitrate="192k")
        mp3_buffer.seek(0)
        return mp3_buffer.read()

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Trumpet LLM Arena", layout="wide")
st.title("Trumpet Exercise LLM Evaluator")

suggestions = [
    "Write a simple warm-up exercise for trumpet.",
    "Write a beginner trumpet exercise focusing on stepwise motion.",
    "Write a trumpet arpeggio exercise.",
    "Write a valve combination practice for trumpet.",
    "Write a trumpet exercise with rhythmic variation."
]

prompt = st.selectbox("Choose a suggested prompt or type your own:", suggestions, index=0)
custom_prompt = st.text_input("Or enter your own prompt here:", value=prompt)

if "available_models" not in st.session_state:
    st.session_state.available_models = get_models()

if "selected_models" not in st.session_state:
    st.session_state.selected_models = []

selected_models = st.multiselect(
    "Select which models to run:",
    st.session_state.available_models,
    default=st.session_state.selected_models
)

st.session_state.selected_models = selected_models

if st.button("Generate Exercises"):
    if not selected_models:
        st.warning("Please select at least one model.")
    else:
        with st.spinner("Running selected models..."):
            results = run_models(selected_models, custom_prompt)

            for model, output in results.items():
                st.subheader(f"Model: `{model}`")

                if isinstance(output, Exception):
                    st.error(f"Error while querying model '{model}': {output}")
                    continue

                st.code(output, language="json")
                parsed = safe_parse_json(output)

                if parsed:
                    try:
                        # Create a MIDI file from the parsed JSON
                        midi = json_to_midi(parsed, instrument=56)  # 56 is the MIDI program for trumpet
                        
                        # Convert MIDI to MP3 using our enhanced function
                        mp3_data = midi_to_mp3(midi, force_fallback=False)
                        
                        # Display audio player
                        st.audio(mp3_data, format="audio/mp3")
                    except Exception as e:
                        st.error(f"Failed to generate audio: {e}")
                else:
                    st.warning("Could not parse output as valid JSON.")

st.markdown("---")
st.markdown("### About This App")
st.markdown("""
    This app evaluates different LLM models on their ability to generate trumpet exercises.
    It uses a custom audio synthesis engine to create high-quality audio from the generated exercises,
    even when soundfonts are not available.
    
    The audio generation includes:
    - Proper note timing and duration
    - ADSR envelope for natural note articulation
    - Harmonic overtones for richer sound
    - Reverb for spatial depth
    - Trumpet-specific audio processing
""")