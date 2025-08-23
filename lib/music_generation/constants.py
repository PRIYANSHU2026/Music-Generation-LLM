#!/usr/bin/env python

"""
Music Generation Constants
=======================
Constants and configuration values for music generation.
"""

from typing import Dict

# MIDI configuration
TICKS_PER_BEAT = 480  # Standard MIDI resolution
TICKS_PER_8TH = TICKS_PER_BEAT // 2  # 240 ticks per 8th note
SAMPLE_RATE = 44100  # Hz

# Soundfont URLs
SOUNDFONT_URLS: Dict[str, str] = {
    "Trumpet": "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2",
    "Piano": "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2",
    "Violin": "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2",
    "Clarinet": "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2",
    "Flute": "https://github.com/FluidSynth/fluidsynth/raw/master/sf2/VintageDreamsWaves-v2.sf2",
}

# Instrument MIDI program numbers
INSTRUMENT_PROGRAMS: Dict[str, int] = {
    "Piano": 0, "Trumpet": 56, "Violin": 40,
    "Clarinet": 71, "Flute": 73,
}

# API configuration
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"