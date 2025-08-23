#!/usr/bin/env python

"""
Audio Converter
==============
Functions for converting MIDI to audio formats.
"""

import os
import uuid
import shutil
import tempfile
import subprocess
import requests
from typing import Tuple, Optional

from mido import MidiFile
from lib.music_generation.constants import SOUNDFONT_URLS, TICKS_PER_BEAT, SAMPLE_RATE


def get_soundfont(instrument: str) -> Optional[str]:
    """
    Download or retrieve a soundfont for the specified instrument.
    
    Args:
        instrument: Instrument name
        
    Returns:
        Path to soundfont file or None if download fails
    """
    os.makedirs("soundfonts", exist_ok=True)
    sf2_path = f"soundfonts/{instrument}.sf2"
    
    # If soundfont already exists, return its path
    if os.path.exists(sf2_path) and os.path.getsize(sf2_path) > 10240:  # At least 10KB
        return sf2_path
    
    # Alternative URLs for common instruments if primary source fails
    alternative_urls = {
        "Trumpet": [
            "https://freepats.zenvoid.org/Brass/trumpet-classique.sf2",
            "https://musical-artifacts.com/artifacts/6471/download/Trumpet.sf2",
            "https://freepats.zenvoid.org/Brass/trumpet.sf2"
        ],
        "Piano": [
            "https://freepats.zenvoid.org/Piano/acoustic-grand-piano.sf2",
            "https://musical-artifacts.com/artifacts/6739/download/Piano.sf2",
            "https://freepats.zenvoid.org/Piano/YDP-GrandPiano.sf2"
        ],
        "Violin": [
            "https://freepats.zenvoid.org/Orchestra/strings.sf2",
            "https://musical-artifacts.com/artifacts/6308/download/Violin.sf2"
        ],
        "Clarinet": [
            "https://freepats.zenvoid.org/Reed/clarinet.sf2",
            "https://musical-artifacts.com/artifacts/5492/download/Clarinet.sf2"
        ],
        "Flute": [
            "https://freepats.zenvoid.org/Woodwind/flute.sf2",
            "https://musical-artifacts.com/artifacts/6742/download/Flute.sf2"
        ]
    }
    
    # Get URLs to try for this instrument
    urls_to_try = alternative_urls.get(instrument, [])
    
    # Add the URL from constants if available
    main_url = SOUNDFONT_URLS.get(instrument)
    if main_url and main_url not in urls_to_try:
        urls_to_try.insert(0, main_url)  # Try the main URL first
    
    # If no URLs for this instrument, try Piano
    if not urls_to_try:
        print(f"No soundfont URLs defined for {instrument}, trying Piano instead.")
        urls_to_try = alternative_urls.get("Piano", [])
        main_url = SOUNDFONT_URLS.get("Piano")
        if main_url and main_url not in urls_to_try:
            urls_to_try.insert(0, main_url)
    
    # If still no URLs, give up
    if not urls_to_try:
        print("No soundfont URLs available.")
        return None
    
    # Try each URL until one works
    print(f"Downloading SoundFont for {instrument}…")
    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=30)
            
            # Check if response is valid
            if response.status_code != 200:
                print(f"Warning: Failed to download from {url} (HTTP {response.status_code}).")
                continue
                
            # Check content type and size
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type or 'text' in content_type:
                print(f"Warning: Invalid response from {url} (received HTML/text instead of binary data).")
                continue
                
            # Check if file size is reasonable for a soundfont (at least 10KB)
            if len(response.content) < 10240:
                print(f"Warning: Downloaded file from {url} is too small to be a valid soundfont.")
                continue

            # Write the soundfont file
            with open(sf2_path, "wb") as f:
                f.write(response.content)
                
            # Verify the file was written successfully
            if os.path.exists(sf2_path) and os.path.getsize(sf2_path) > 10240:
                print(f"Successfully downloaded soundfont for {instrument}")
                return sf2_path
            else:
                print(f"Warning: Failed to write {instrument} soundfont file.")
                # Continue to next URL
                
        except requests.exceptions.Timeout:
            print(f"Warning: Timeout while downloading from {url}.")
            # Continue to next URL
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            # Continue to next URL
    
    # If we get here, all URLs failed
    print(f"Warning: All download attempts failed for {instrument} soundfont.")
    return None


def midi_to_mp3(midi_obj: MidiFile, instrument: str = "Piano", force_fallback: bool = False) -> Tuple[Optional[str], float]:
    """
    Convert a MIDI object to MP3 audio file.
    
    Args:
        midi_obj: MidiFile object to convert
        instrument: Instrument name for soundfont selection
        force_fallback: Whether to force using fallback audio generation
        
    Returns:
        Tuple of (path to MP3 file, duration in seconds) or (None, 0) if conversion fails
    """
    os.makedirs("static", exist_ok=True)
    os.makedirs("temp_audio", exist_ok=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as mid_file:
        midi_obj.save(mid_file.name)
        wav_path = mid_file.name.replace(".mid", ".wav")
        mp3_path = mid_file.name.replace(".mid", ".mp3")

    # Use fallback if requested
    if force_fallback:
        print(f"Fallback audio generation requested for {instrument}")
        return generate_fallback_audio(midi_obj, mp3_path)

    # Try to get soundfont
    sf2_path = None
    
    # First try the requested instrument
    sf2_path = get_soundfont(instrument)
        
    # If that fails and instrument is not Piano, try Piano as a fallback instrument
    if not sf2_path and instrument != "Piano":
        print(f"No valid soundfont available for {instrument}, trying Piano instead...")
        sf2_path = get_soundfont("Piano")
        
    # If still no soundfont, use fallback audio generation
    if not sf2_path:
        print(f"No valid soundfont available, using fallback audio generation")
        return generate_fallback_audio(midi_obj, mp3_path)

    try:
        # Check if fluidsynth is available
        try:
            subprocess.run(['fluidsynth', '--version'], check=True, capture_output=True, text=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("FluidSynth not available, using fallback audio generation")
            return generate_fallback_audio(midi_obj, mp3_path)
            
        # Try using fluidsynth command
        result = subprocess.run([
            'fluidsynth', '-ni', sf2_path, mid_file.name,
            '-F', wav_path, '-r', '44100', '-g', '1.0'
        ], check=True, capture_output=True, text=True)

        # Check if WAV file was created successfully
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1024:
            print(f"FluidSynth did not generate a valid WAV file for {instrument}, using fallback")
            return generate_fallback_audio(midi_obj, mp3_path)

        # Convert to MP3
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_wav(wav_path)
            
            # Apply instrument-specific filters
            if instrument == "Trumpet":
                sound = sound.high_pass_filter(200)
            elif instrument == "Violin":
                sound = sound.low_pass_filter(5000)
                
            sound.export(mp3_path, format="mp3")
            
            # Check if MP3 file was created successfully
            if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) < 1024:
                print(f"Failed to create MP3 file for {instrument}, using fallback")
                return generate_fallback_audio(midi_obj, mp3_path)
                
            # Move to static directory
            static_mp3_path = os.path.join('static', f'exercise_{uuid.uuid4().hex}.mp3')
            shutil.move(mp3_path, static_mp3_path)
            return static_mp3_path, sound.duration_seconds
        except ImportError as e:
            print(f"Required audio libraries not available: {e}, using fallback")
            return generate_fallback_audio(midi_obj, mp3_path)
        except Exception as e:
            print(f"MP3 conversion failed: {e}, using fallback")
            return generate_fallback_audio(midi_obj, mp3_path)
    except subprocess.SubprocessError as e:
        print(f"FluidSynth process error: {e}, using fallback")
        return generate_fallback_audio(midi_obj, mp3_path)
    except Exception as e:
        print(f"FluidSynth failed: {e}, using fallback")
        return generate_fallback_audio(midi_obj, mp3_path)
    finally:
        # Clean up temporary files
        for f in [mid_file.name, wav_path]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


def generate_fallback_audio(midi_obj: MidiFile, output_path: str) -> Tuple[Optional[str], float]:
    """
    Generate simple audio using sine waves as fallback when FluidSynth is unavailable.
    
    Args:
        midi_obj: MidiFile object to convert
        output_path: Path to save the output MP3
        
    Returns:
        Tuple of (path to MP3 file, duration in seconds) or (None, 0) if generation fails
    """
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
        sample_rate = SAMPLE_RATE
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


def create_metronome_audio(tempo: int, time_sig: str, measures: int) -> Optional[str]:
    """
    Create a metronome audio file.
    
    Args:
        tempo: Tempo in BPM
        time_sig: Time signature (e.g., "4/4")
        measures: Number of measures
        
    Returns:
        Path to MP3 file or None if generation fails
    """
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
        
        os.makedirs("static", exist_ok=True)
        
        numerator, denominator = map(int, time_sig.split('/'))
        
        # Create metronome clicks with pydub
        click_duration = 50  # milliseconds
        strong_click = Sine(1000).to_audio_segment(duration=click_duration).apply_gain(-3)
        weak_click = Sine(800).to_audio_segment(duration=click_duration).apply_gain(-6)

        # Calculate silence duration between clicks
        silence_duration = 60000 / tempo - click_duration  # ms per beat minus click

        # Calculate total beats
        beats_per_measure = numerator
        total_beats = beats_per_measure * measures

        metronome_audio = AudioSegment.silent(duration=0)

        for beat in range(total_beats):
            click = strong_click if beat % beats_per_measure == 0 else weak_click
            metronome_audio += click + AudioSegment.silent(duration=silence_duration)

        # Export to MP3
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            mp3_path = temp_file.name
            
        metronome_audio.export(mp3_path, format="mp3")

        # Move to static directory
        static_mp3_path = os.path.join('static', f'metronome_{uuid.uuid4().hex}.mp3')
        shutil.move(mp3_path, static_mp3_path)

        return static_mp3_path
    except ImportError as e:
        print(f"Metronome requires pydub: {e}")
        return None
    except Exception as e:
        print(f"Error creating metronome: {e}")
        return None