# Music Generation LLM

A modular application for generating adaptive music exercises using LLMs.

## Project Structure

The project has been refactored into a modular structure:

```
├── lib/                    # Core music generation functionality
│   └── music_generation/   # Music generation modules
│       ├── constants.py    # Configuration and constants
│       ├── generator.py    # Exercise generation logic
│       └── theory.py       # Music theory helpers
├── processing/             # Processing modules
│   ├── audio/              # Audio processing
│   │   └── converter.py    # MIDI to audio conversion
│   ├── midi/               # MIDI processing
│   │   └── converter.py    # JSON to MIDI conversion
│   └── visualization/      # Visualization tools
│       └── visualizer.py   # Piano roll visualization
├── tests/                  # Test suite
│   ├── lib/                # Tests for lib modules
│   └── processing/         # Tests for processing modules
├── cli.py                  # Command-line interface
└── requirements.txt        # Project dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate a music exercise

```bash
python cli.py generate --instrument Trumpet --level Intermediate --key "C Major" --time-signature "4/4" --measures 4 --output-format all
```

### Generate a metronome track

```bash
python cli.py metronome --tempo 60 --time-signature "4/4" --measures 4
```

### Convert a JSON exercise to MIDI or MP3

```bash
python cli.py convert --input-file exercise.json --output-format mp3 --instrument Piano
```

### Display available options

```bash
python cli.py info
```

## Module Overview

### lib/music_generation

- **constants.py**: Configuration values and constants
- **generator.py**: Core music generation logic using LLM
- **theory.py**: Music theory helpers for note conversion

### processing/midi

- **converter.py**: Convert JSON note data to MIDI files

### processing/audio

- **converter.py**: Convert MIDI files to MP3 audio

### processing/visualization

- **visualizer.py**: Generate piano roll visualizations

## Testing

Run the test suite:

```bash
python -m unittest discover tests
```

## Error Handling

The application is designed to fail gracefully when errors occur, with no automatic fallbacks. Error messages are displayed to help diagnose issues.



