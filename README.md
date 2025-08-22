# HarmonyHub: Adaptive Music Exercise Generator

## Overview

HarmonyHub is an intelligent music education tool that uses generative AI to create personalized music exercises. It adapts to different skill levels and provides real-time feedback, making music learning more engaging and effective.

## Features

- **Adaptive Exercise Generation**: Creates music exercises tailored to your skill level and learning goals
- **Multiple Output Formats**: Generates exercises as MIDI files, audio files, and sheet music
- **Metronome Generation**: Creates customizable metronome tracks to practice with
- **Format Conversion**: Converts between different music file formats
- **VexFlow Integration**: Visualizes sheet music with VexFlow

## Project Structure

```
adaptive-music-generator/
├── adaptive_music_generator/
│   ├── __init__.py         # Package initialization
│   ├── lib.py              # Core music generation logic
│   ├── processors.py       # MIDI/audio/image processing
│   ├── cli.py              # CLI interface
│   └── exceptions.py       # Custom exceptions
├── tests/
│   ├── test_lib.py         # Tests for core logic
│   └── test_processors.py  # Tests for processors
├── requirements.txt        # Project dependencies
├── Dockerfile              # Container definition
└── README.md               # This file
```

## Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/adaptive-music-generator.git
cd adaptive-music-generator

# Install dependencies
pip install -r requirements.txt

# Set Mistral API Key (required for exercise generation)
export MISTRAL_API_KEY="your_api_key_here"
# On Windows use: set MISTRAL_API_KEY=your_api_key_here
```

### Docker Installation

```bash
# Build the Docker image
docker build -t harmonyhub .

# Run the container with your API key
docker run -it --rm -e MISTRAL_API_KEY="your_api_key_here" harmonyhub
```

## Usage

### CLI Usage

```bash
# Generate a music exercise
python cli.py generate --instrument Piano --level Beginner --key "C Major" --time-signature 4/4 --measures 4 --output-format mp3

# Generate a metronome track
python cli.py metronome --tempo 120 --time-signature 4/4 --measures 8

# Convert a MIDI file to audio
python cli.py convert --input-file exercise.mid --output-format wav

# Show available options
python cli.py info
```

### Docker Usage

```bash
# Run with specific command
docker run -it --rm -v $(pwd)/output:/app/output -e MISTRAL_API_KEY="your_api_key_here" harmonyhub generate --instrument Piano --level Intermediate

# Show help
docker run -it --rm harmonyhub --help
```

## Troubleshooting

### API Key Error

If you see an error like:
```
Error generating exercise: MISTRAL_API_KEY environment variable not set
```

Make sure to set your Mistral API key as an environment variable:
```bash
# Linux/macOS
export MISTRAL_API_KEY="yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX"

# Windows
set MISTRAL_API_KEY=yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX
```

### Command Parameter Issues

If you see an error like:
```
No such option: --difficulty
```

Make sure to use the correct parameter names. Use `--level` instead of `--difficulty` for specifying the exercise difficulty level.

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Features

1. Implement your feature in the appropriate module
2. Add tests in the `tests/` directory
3. Update documentation as needed

## License

MIT

## Acknowledgements

- FluidSynth for MIDI to audio conversion
- VexFlow for music notation rendering
- Mistral AI for the language model backend



