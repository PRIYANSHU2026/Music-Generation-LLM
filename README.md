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
## Docker Deployment
Build and run the Docker container:

```bash
docker build -t music-exercise-generator .
docker run -p 7860:7860 music-exercise-generator
```

## Dependencies
See requirements.txt for a full list of dependencies.



