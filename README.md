# Adaptive Music Exercise Generator

## Overview
This application generates custom musical exercises using LLM technology, perfectly fit to user-specified parameters like number of measures and time signature. It guarantees exact durations in MIDI and provides visual music notation using VexFlow.

## Features
- Generates musical exercises based on user preferences
- Visualizes music notation using VexFlow
- Converts exercises to MIDI and audio
- Supports multiple instruments
- Provides both web interface and CLI options
- Optimized for Hugging Face Spaces deployment

## VexFlow Integration
The application uses VexFlow for music notation visualization with multiple fallback mechanisms:
1. Primary CDN: jsdelivr.net
2. Local copy in static directory
3. Secondary CDN: unpkg.com

## CLI Usage
The application provides a command-line interface for generating exercises:

```bash
# Generate an exercise with default parameters
python cli.py generate

# Generate an exercise with custom parameters
python cli.py generate --instrument Trumpet --level Intermediate --key "C Major" --time-signature 4/4 --measures 4 --output-format mp3

# Force fallback audio generation (useful when soundfonts are unavailable)
python cli.py generate --force-fallback

# Generate a metronome track
python cli.py metronome --tempo 120 --time-signature 4/4 --measures 4

# Convert a JSON exercise file to MIDI or MP3
python cli.py convert --input-file exercise.json --output-format mp3

# Display available options
python cli.py info
```

## Web Interface
To run the web interface locally:

```bash
python app.py
```

## Docker Deployment
Build and run the Docker container:

```bash
docker build -t music-exercise-generator .
docker run -p 7860:7860 music-exercise-generator
```

## Dependencies
See requirements.txt for a full list of dependencies.



