"""Audio, MIDI, and image processing utilities for the Adaptive Music Generator.

This module provides functionality for processing MIDI files, converting to audio,
and generating music notation visualizations.
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple

import mido
from mido import MidiFile

from .exceptions import AudioConversionError, VisualizationError

# Constants
SAMPLE_RATE = 44100  # Hz
SOUNDFONT_URLS = {
    "Trumpet": "https://musical-artifacts.com/artifacts/6471",
    "Piano": "https://musical-artifacts.com/artifacts/6739",
    "Violin": "https://musical-artifacts.com/artifacts/6308",
    "Clarinet": "https://musical-artifacts.com/artifacts/5492",
    "Flute": "https://musical-artifacts.com/artifacts/6742",
}


def ensure_directories() -> None:
    """Ensure that necessary directories exist."""
    os.makedirs("static", exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)
    os.makedirs("soundfonts", exist_ok=True)


def download_all_soundfonts() -> None:
    """Download all available soundfonts for offline use.
    
    This function downloads all soundfonts defined in SOUNDFONT_URLS
    to ensure they're available for offline use.
    """
    print("Downloading all soundfonts for offline use...")
    ensure_directories()
    
    for instrument, url in SOUNDFONT_URLS.items():
        try:
            print(f"Downloading {instrument} soundfont...")
            soundfont_filename = os.path.basename(url)
            soundfont_path = os.path.join("soundfonts", soundfont_filename)
            
            if os.path.exists(soundfont_path):
                print(f"{instrument} soundfont already exists at {soundfont_path}")
                continue
                
            import requests
            response = requests.get(url)
            response.raise_for_status()
            
            with open(soundfont_path, "wb") as f:
                f.write(response.content)
                
            print(f"Successfully downloaded {instrument} soundfont to {soundfont_path}")
            
        except Exception as e:
            print(f"Error downloading {instrument} soundfont: {str(e)}")
            
    print("Soundfont download process completed.")
    print("Available soundfonts:")
    for sf in os.listdir("soundfonts"):
        if sf.endswith(".sf2"):
            print(f" - {sf}")



def download_soundfont(instrument: str) -> str:
    """Download a soundfont for the specified instrument if not already available.
    
    Args:
        instrument: Name of the instrument
        
    Returns:
        Path to the downloaded soundfont file
    """
    ensure_directories()
    
    instrument = instrument.capitalize()
    if instrument not in SOUNDFONT_URLS:
        instrument = "Piano"  # Default to piano if instrument not found
        
    soundfont_url = SOUNDFONT_URLS[instrument]
    soundfont_filename = os.path.basename(soundfont_url)
    soundfont_path = os.path.join("soundfonts", soundfont_filename)
    
    # Download if not exists
    if not os.path.exists(soundfont_path):
        try:
            import requests
            print(f"Downloading soundfont for {instrument} from {soundfont_url}")
            response = requests.get(soundfont_url)
            response.raise_for_status()
            
            with open(soundfont_path, "wb") as f:
                f.write(response.content)
            
            print(f"Successfully downloaded soundfont to {soundfont_path}")
                
        except Exception as e:
            print(f"Error downloading {instrument} soundfont: {str(e)}")
            print(f"Attempting to use fallback soundfont...")
            
            # Try to find any existing soundfont in the directory
            soundfont_dir = "soundfonts"
            existing_soundfonts = [f for f in os.listdir(soundfont_dir) if f.endswith(".sf2")]
            
            if existing_soundfonts:
                fallback_sf = os.path.join(soundfont_dir, existing_soundfonts[0])
                print(f"Using existing soundfont: {fallback_sf}")
                return fallback_sf
                
            # Fall back to default piano soundfont if download fails
            default_sf = os.path.join("soundfonts", os.path.basename(SOUNDFONT_URLS["Piano"]))
            if os.path.exists(default_sf):
                print(f"Using default piano soundfont: {default_sf}")
                return default_sf
                
            # If no soundfonts are available, try downloading the piano soundfont
            try:
                print("Attempting to download default piano soundfont...")
                piano_url = SOUNDFONT_URLS["Piano"]
                piano_filename = os.path.basename(piano_url)
                piano_path = os.path.join("soundfonts", piano_filename)
                
                response = requests.get(piano_url)
                response.raise_for_status()
                
                with open(piano_path, "wb") as f:
                    f.write(response.content)
                    
                print(f"Successfully downloaded piano soundfont to {piano_path}")
                return piano_path
                
            except Exception as piano_error:
                raise AudioConversionError(f"Failed to download any soundfont. Original error: {str(e)}. Piano fallback error: {str(piano_error)}")
    
    return soundfont_path


def midi_to_audio(midi_file: Union[MidiFile, str], output_path: str, 
                 instrument: str = "Piano", format: str = "mp3") -> str:
    """Convert a MIDI file to audio using FluidSynth.
    
    Args:
        midi_file: MidiFile object or path to MIDI file
        output_path: Path to save the output audio file
        instrument: Instrument name for soundfont selection
        format: Output audio format (mp3, wav)
        
    Returns:
        Path to the generated audio file
    """
    ensure_directories()
    
    # Save MIDI file if it's an object
    if isinstance(midi_file, MidiFile):
        temp_midi = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        midi_file.save(temp_midi.name)
        midi_path = temp_midi.name
    else:
        midi_path = midi_file
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Get soundfont
        soundfont_path = download_soundfont(instrument)
        
        # Convert MIDI to audio using FluidSynth
        try:
            from midi2audio import FluidSynth
            fs = FluidSynth(soundfont_path)
            
            if format.lower() == "mp3":
                fs.midi_to_audio(midi_path, output_path)
            else:  # Default to WAV
                wav_path = output_path.rsplit(".", 1)[0] + ".wav"
                fs.midi_to_audio(midi_path, wav_path)
                
                # Convert WAV to desired format if not WAV
                if format.lower() != "wav":
                    try:
                        from pydub import AudioSegment
                        audio = AudioSegment.from_wav(wav_path)
                        audio.export(output_path, format=format.lower())
                        os.remove(wav_path)  # Clean up WAV file
                    except ImportError:
                        # Fall back to ffmpeg if pydub not available
                        subprocess.run(["ffmpeg", "-i", wav_path, output_path], 
                                      check=True, capture_output=True)
                        os.remove(wav_path)  # Clean up WAV file
                else:
                    # If format is WAV, just use the wav_path
                    output_path = wav_path
                    
        except ImportError:
            # Fallback to direct FluidSynth command if midi2audio not available
            wav_path = output_path.rsplit(".", 1)[0] + ".wav"
            subprocess.run(["fluidsynth", "-ni", soundfont_path, midi_path, "-F", wav_path, 
                          "-r", str(SAMPLE_RATE)], check=True, capture_output=True)
            
            # Convert WAV to desired format if not WAV
            if format.lower() != "wav":
                subprocess.run(["ffmpeg", "-i", wav_path, output_path], 
                              check=True, capture_output=True)
                os.remove(wav_path)  # Clean up WAV file
            else:
                # If format is WAV, just use the wav_path
                output_path = wav_path
                
    except Exception as e:
        raise AudioConversionError(f"Failed to convert MIDI to audio: {str(e)}")
    finally:
        # Clean up temporary MIDI file if created
        if isinstance(midi_file, MidiFile) and os.path.exists(midi_path):
            os.remove(midi_path)
    
    return output_path


def generate_sheet_music(exercise_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """Generate sheet music visualization from exercise data.
    
    Args:
        exercise_data: Dictionary containing exercise data
        output_path: Path to save the output image file (optional)
        
    Returns:
        HTML content for rendering the sheet music
    """
    ensure_directories()
    
    try:
        # Generate VexFlow HTML for sheet music visualization
        notes = exercise_data.get("notes", [])
        if not notes:
            raise VisualizationError("No notes found in exercise data")
        
        # Sort notes by start time
        notes.sort(key=lambda x: x.get("start_time", 0))
        
        # Get time signature and key
        time_signature = exercise_data.get("time_signature", "4/4")
        key = exercise_data.get("key", "C Major")
        
        # Generate VexFlow HTML
        vexflow_html = generate_vexflow_html(notes, time_signature, key)
        
        # Save to file if output_path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(vexflow_html)
        
        return vexflow_html
        
    except Exception as e:
        raise VisualizationError(f"Failed to generate sheet music: {str(e)}")


def generate_vexflow_html(notes: list, time_signature: str, key: str) -> str:
    """Generate VexFlow HTML for sheet music visualization.
    
    Args:
        notes: List of note dictionaries
        time_signature: Time signature string (e.g., "4/4")
        key: Musical key (e.g., "C Major")
        
    Returns:
        HTML content for rendering the sheet music
    """
    # Convert time signature
    time_parts = time_signature.split("/")
    if len(time_parts) != 2:
        raise VisualizationError(f"Invalid time signature: {time_signature}")
    
    numerator, denominator = int(time_parts[0]), int(time_parts[1])
    
    # Convert key
    key_parts = key.split(" ")
    if len(key_parts) != 2:
        raise VisualizationError(f"Invalid key: {key}")
    
    key_note, key_mode = key_parts[0], key_parts[1].lower()
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Sheet Music Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/vexflow@4.2.2/build/cjs/vexflow.js"></script>
        <style>
            #output {{
                width: 100%;
                overflow: auto;
            }}
        </style>
    </head>
    <body>
        <div id="output"></div>
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const {{ Renderer, Stave, StaveNote, Formatter, Accidental, TimeSignature, KeySignature }} = Vex.Flow;
                
                // Create renderer
                const div = document.getElementById('output');
                const renderer = new Renderer(div, Renderer.Backends.SVG);
                
                // Configure renderer
                renderer.resize(800, 200);
                const context = renderer.getContext();
                
                // Create stave
                const stave = new Stave(10, 40, 780);
                stave.addTimeSignature('{numerator}/{denominator}');
                stave.addKeySignature('{key_note}');
                stave.setContext(context).draw();
                
                // Create notes
                const notes = [
    """
    
    # Process notes
    note_html = []
    current_time = 0
    
    for note in notes:
        pitch = note.get("pitch", "C4")
        duration = note.get("duration", 1)  # In eighth notes
        start_time = note.get("start_time", 0)
        
        # Add rest if there's a gap
        if start_time > current_time:
            rest_duration = start_time - current_time
            note_html.append(f'new StaveNote({{clef: "treble", keys: ["b/4"], duration: "{rest_duration}r"}})')
        
        # Add the note
        note_html.append(f'new StaveNote({{clef: "treble", keys: ["{pitch.lower()}"], duration: "{duration}"}})')
        
        # Update current time
        current_time = start_time + duration
    
    html += ",\n                    ".join(note_html)
    
    html += f"""
                ];
                
                // Format and draw notes
                Formatter.FormatAndDraw(context, stave, notes);
            }});
        </script>
    </body>
    </html>
    """
    
    return html