<div align="center">
  <img width="561" height="212" alt="image" src="https://github.com/user-attachments/assets/f9a451aa-8237-4aa6-a335-df192f5682a7" />

</div>

<h1 align="center">GSoC 2025(INCF)</h1>


## **Project Title:** *HarmonyHub: Using Generative AI for Adaptive Learning in Music CLI Version*  
**Organization:** INCF  
**Contributor:** **Priyanshu Tiwari**  
**Mentors:** Alberto Acquilino • Mirko D'Andrea • Keerthi Reddy Kambham • Thrun • Oscar  
**Hugging Face Repo:** [🔗 Music LLM](https://huggingface.co/spaces/SHIKARICHACHA/adaptive-music-exercise-generator)

---

## 📜 **Executive Summary**

**HarmonyHub** is an **AI-driven adaptive music education platform** that leverages the **Mistral LLM API** to generate **personalized, rhythmically precise, and melodically coherent** practice exercises in real time. Designed for **students, educators, and self-taught musicians**, the system dynamically adapts to user-defined parameters:

- 🎹 **Instrument**: Piano, Violin, Trumpet, Clarinet, Flute
- 🔤 **Difficulty Level**: Beginner, Intermediate, Advanced
- ⏱ **Time Signature & Key**: e.g., 4/4 in C Major, 6/8 in A Minor
- 🎯 **Practice Focus**: Rhythmic, Melodic, Technical, Expressive, Sight-Reading, Improvisation
- 🎼 **Rhythmic Complexity**: Basic, Syncopated, Polyrhythmic

Generated exercises are delivered in **MIDI**, **MP3**, and **JSON** formats, accompanied by:
- Real-time **sheet music visualization** via VexFlow
- Interactive **AI music theory assistant**
- No-code **Gradio interface** for instant access

HarmonyHub bridges **generative AI** and **music cognition**, offering an intelligent, accessible, and scalable tool for modern music pedagogy.

## VexFlow Integration
The application uses VexFlow for music notation visualization with multiple fallback mechanisms:
1. Primary CDN: jsdelivr.net
2. Local copy in static directory
3. Secondary CDN: unpkg.com


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



