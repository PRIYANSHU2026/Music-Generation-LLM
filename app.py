"""Adaptive Music Exercise Generator (Strict Duration Enforcement)
==============================================================
Generates custom musical exercises with LLM, perfectly fit to user-specified number of measures
AND time signature, guaranteeing exact durations in MIDI and in the UI!

This file contains the web interface using Gradio.
The business logic has been moved to lib.py.
"""

# Import business logic from lib.py
from lib import (
    install, ensure_fluidsynth, ensure_directories,
    is_rest, note_name_to_midi, midi_to_note_name,
    scale_json_durations, json_to_midi, midi_to_mp3,
    create_visualization, create_metronome_audio,
    calculate_difficulty_rating, query_mistral,
    safe_parse_json, get_fallback_exercise,
    generate_exercise
)

# Install required packages
install([
    "mido", "midi2audio", "pydub", "gradio",
    "requests", "numpy", "matplotlib", "librosa", "scipy",
    "uuid", "datetime"
])

# Ensure FluidSynth is installed
ensure_fluidsynth()

# Ensure necessary directories exist
ensure_directories()

# Import required packages for the web interface
import gradio as gr
import os
import base64
import json
import time
from datetime import datetime
import uuid

# -----------------------------------------------------------------------------
# Gradio Web Interface
# -----------------------------------------------------------------------------

def generate_exercise_ui(instrument, level, key, time_signature, measures, 
                      difficulty_modifier, practice_focus, custom_prompt, tempo):
    """Generate a musical exercise and create the UI elements."""
    start_time = time.time()
    
    # Generate the exercise
    json_data, llm_response = generate_exercise(
        instrument, level, key, time_signature, measures,
        difficulty_modifier, practice_focus, custom_prompt, tempo
    )
    
    # Calculate difficulty rating
    difficulty = calculate_difficulty_rating(json_data, instrument, level, time_signature)
    
    # Create MIDI file
    mid = json_to_midi(json_data, instrument, tempo, time_signature, measures, key)
    
    # Create MP3 file
    mp3_path = midi_to_mp3(mid, instrument)
    
    # Create visualization
    viz_path = create_visualization(json_data, instrument, tempo, time_signature, measures, key)
    
    # Create metronome audio
    metronome_path = create_metronome_audio(tempo, time_signature, measures)
    
    # Calculate total duration
    total_duration = sum(duration for _, duration in json_data)
    
    # Format the exercise for display
    formatted_exercise = "\n".join([f"{note}: {duration}" for note, duration in json_data])
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Return the UI elements
    return (
        json.dumps(json_data, indent=2),
        mp3_path,
        metronome_path,
        viz_path,
        f"Difficulty Rating: {difficulty}/10",
        f"Total Duration: {total_duration} eighth notes",
        f"Generation Time: {generation_time:.2f} seconds",
        llm_response
    )

# Create the Gradio interface
with gr.Blocks(title="Adaptive Music Exercise Generator") as demo:
    gr.Markdown("# 🎵 Adaptive Music Exercise Generator")
    gr.Markdown("Generate custom musical exercises with LLM, perfectly fit to your specifications!")
    
    with gr.Row():
        with gr.Column(scale=1):
            instrument = gr.Dropdown(
                ["Trumpet", "Piano", "Violin", "Clarinet", "Flute"],
                label="Instrument",
                value="Trumpet"
            )
            level = gr.Dropdown(
                ["Beginner", "Intermediate", "Advanced"],
                label="Level",
                value="Intermediate"
            )
            key = gr.Dropdown(
                ["C Major", "G Major", "D Major", "F Major", "Bb Major", "A Minor", "E Minor"],
                label="Key",
                value="C Major"
            )
            time_signature = gr.Dropdown(
                ["4/4", "3/4"],
                label="Time Signature",
                value="4/4"
            )
        
        with gr.Column(scale=1):
            measures = gr.Slider(
                minimum=1,
                maximum=8,
                step=1,
                label="Number of Measures",
                value=4
            )
            difficulty_modifier = gr.Slider(
                minimum=-2,
                maximum=2,
                step=1,
                label="Difficulty Modifier",
                value=0
            )
            practice_focus = gr.Dropdown(
                ["Balanced", "Rhythmic Focus", "Melodic Focus", "Technical Focus", "Expressive Focus"],
                label="Practice Focus",
                value="Balanced"
            )
            tempo = gr.Slider(
                minimum=40,
                maximum=200,
                step=5,
                label="Tempo (BPM)",
                value=60
            )
    
    custom_prompt = gr.Textbox(
        label="Custom Prompt (Optional)",
        placeholder="Leave blank to use default prompt",
        lines=3
    )
    
    generate_btn = gr.Button("Generate Exercise", variant="primary")
    
    with gr.Row():
        with gr.Column(scale=1):
            json_output = gr.Code(
                label="Exercise JSON",
                language="json"
            )
            llm_response = gr.Textbox(
                label="LLM Response",
                lines=5
            )
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Exercise Audio",
                type="filepath"
            )
            metronome_output = gr.Audio(
                label="Metronome",
                type="filepath"
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            viz_output = gr.Image(
                label="Visualization",
                type="filepath"
            )
        
        with gr.Column(scale=1):
            difficulty_output = gr.Textbox(
                label="Difficulty"
            )
            duration_output = gr.Textbox(
                label="Duration"
            )
            time_output = gr.Textbox(
                label="Generation Time"
            )
    
    # Connect the button to the generate function
    generate_btn.click(
        generate_exercise_ui,
        inputs=[
            instrument, level, key, time_signature, measures,
            difficulty_modifier, practice_focus, custom_prompt, tempo
        ],
        outputs=[
            json_output, audio_output, metronome_output, viz_output,
            difficulty_output, duration_output, time_output, llm_response
        ]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()