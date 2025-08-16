#!/usr/bin/env python

"""
Adaptive Music Exercise Generator CLI (Typer-based)
==================================================
A command-line interface for generating custom musical exercises using LLM.
This CLI version replaces the Gradio web interface with a Typer-based CLI.
"""

import typer
import json
import os
import sys
from typing import Optional, List, Tuple
from enum import Enum
import mido
from pathlib import Path

# Import core functionality from lib.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib import (
    generate_exercise, json_to_midi, midi_to_mp3, create_visualization,
    create_metronome_audio, calculate_difficulty_rating, query_mistral,
    safe_parse_json, get_fallback_exercise, install
)

# Ensure required packages are installed
install(["typer[all]", "rich", "mido", "midi2audio", "pydub"])

# Import rich for better CLI output
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Create Typer app
app = typer.Typer(help="Adaptive Music Exercise Generator CLI")
console = Console()

# Define enums for CLI options
class Instrument(str, Enum):
    TRUMPET = "Trumpet"
    PIANO = "Piano"
    VIOLIN = "Violin"
    CLARINET = "Clarinet"
    FLUTE = "Flute"

class Level(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"

class Key(str, Enum):
    C_MAJOR = "C Major"
    G_MAJOR = "G Major"
    D_MAJOR = "D Major"
    F_MAJOR = "F Major"
    BB_MAJOR = "Bb Major"
    A_MINOR = "A Minor"
    E_MINOR = "E Minor"

class TimeSignature(str, Enum):
    THREE_FOUR = "3/4"
    FOUR_FOUR = "4/4"

class PracticeFocus(str, Enum):
    BALANCED = "Balanced"
    RHYTHMIC = "Rhythmic Focus"
    MELODIC = "Melodic Focus"
    TECHNICAL = "Technical Focus"
    EXPRESSIVE = "Expressive Focus"

class OutputFormat(str, Enum):
    JSON = "json"
    MIDI = "midi"
    MP3 = "mp3"
    ALL = "all"

@app.command("generate")
def generate(
    instrument: Instrument = typer.Option(Instrument.TRUMPET, help="Instrument to generate exercise for"),
    level: Level = typer.Option(Level.INTERMEDIATE, help="Difficulty level"),
    key: Key = typer.Option(Key.C_MAJOR, help="Key signature"),
    time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
    measures: int = typer.Option(4, help="Number of measures", min=1, max=16),
    difficulty_modifier: int = typer.Option(0, help="Difficulty modifier (-2 to +2)", min=-2, max=2),
    practice_focus: PracticeFocus = typer.Option(PracticeFocus.BALANCED, help="Practice focus"),
    output_format: OutputFormat = typer.Option(OutputFormat.ALL, help="Output format"),
    output_dir: str = typer.Option("./output", help="Directory to save output files"),
    custom_prompt: Optional[str] = typer.Option(None, help="Custom prompt for exercise generation"),
    tempo: int = typer.Option(60, help="Tempo in BPM", min=40, max=200),
    force_fallback: bool = typer.Option(False, help="Force using fallback audio generation instead of soundfonts"),
):
    """Generate a musical exercise based on specified parameters."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Show parameters
    console.print("[bold green]Generating exercise with the following parameters:[/bold green]")
    params_table = Table(show_header=True, header_style="bold magenta")
    params_table.add_column("Parameter")
    params_table.add_column("Value")
    params_table.add_row("Instrument", str(instrument))
    params_table.add_row("Level", str(level))
    params_table.add_row("Key", str(key))
    params_table.add_row("Time Signature", str(time_signature))
    params_table.add_row("Measures", str(measures))
    params_table.add_row("Difficulty Modifier", str(difficulty_modifier))
    params_table.add_row("Practice Focus", str(practice_focus))
    params_table.add_row("Tempo", f"{tempo} BPM")
    console.print(params_table)
    
    # Generate exercise
    with console.status("[bold green]Generating exercise...[/bold green]"):
        mode = "Exercise Prompt" if custom_prompt else "Exercise Parameters"
        # Extract string values from enums
        instrument_str = instrument.value
        level_str = level.value
        key_str = key.value
        time_sig_str = time_signature.value
        practice_focus_str = practice_focus.value
        
        json_data, mp3_path, tempo_str, midi_obj, duration, time_sig, total_duration = generate_exercise(
            instrument_str, level_str, key_str, tempo, time_sig_str,
            measures, custom_prompt or "", mode, difficulty_modifier, practice_focus_str,
            force_fallback=force_fallback
        )
    
    # Calculate difficulty rating
    difficulty_rating = calculate_difficulty_rating(json_data, level_str, difficulty_modifier, practice_focus_str)
    
    # Save outputs based on format
    # Create safe filename components by replacing problematic characters
    safe_instrument = instrument_str.replace(' ', '_')
    safe_level = level_str.replace(' ', '_')
    base_filename = f"exercise_{safe_instrument}_{safe_level}_{measures}m"
    output_files = []
    
    if output_format in [OutputFormat.JSON, OutputFormat.ALL]:
        json_path = os.path.join(output_dir, f"{base_filename}.json")
        with open(json_path, "w") as f:
            f.write(json_data)
        output_files.append(("JSON", json_path))
    
    if output_format in [OutputFormat.MIDI, OutputFormat.ALL]:
        midi_path = os.path.join(output_dir, f"{base_filename}.mid")
        midi_obj.save(midi_path)
        output_files.append(("MIDI", midi_path))
    
    if output_format in [OutputFormat.MP3, OutputFormat.ALL]:
        if mp3_path:
            # Copy the MP3 file to the output directory
            new_mp3_path = os.path.join(output_dir, f"{base_filename}.mp3")
            if os.path.exists(mp3_path):
                import shutil
                shutil.copy(mp3_path, new_mp3_path)
                output_files.append(("MP3", new_mp3_path))
    
    # Generate visualization if all formats are requested
    if output_format == OutputFormat.ALL:
        try:
            viz_path = create_visualization(json_data, str(time_signature))
            if viz_path:
                viz_output = os.path.join(output_dir, f"{base_filename}_viz.png")
                import shutil
                shutil.copy(viz_path, viz_output)
                output_files.append(("Visualization", viz_output))
        except Exception as e:
            console.print(f"[bold red]Error generating visualization: {e}[/bold red]")
    
    # Display results
    console.print("\n[bold green]Exercise generated successfully![/bold green]")
    console.print(f"[bold]Difficulty Rating:[/bold] {difficulty_rating}/10")
    console.print(f"[bold]Duration:[/bold] {duration} seconds")
    console.print(f"[bold]Total Duration Units:[/bold] {total_duration} (8th notes)")
    
    # Show output files
    if output_files:
        console.print("\n[bold]Output Files:[/bold]")
        for file_type, file_path in output_files:
            console.print(f"[bold]{file_type}:[/bold] {file_path}")
    
    # Show JSON preview
    if output_format in [OutputFormat.JSON, OutputFormat.ALL]:
        console.print("\n[bold]JSON Preview:[/bold]")
        try:
            parsed_json = json.loads(json_data)
            console.print_json(json.dumps(parsed_json[:5] if len(parsed_json) > 5 else parsed_json))
            if len(parsed_json) > 5:
                console.print("[italic](showing first 5 notes only)[/italic]")
        except:
            console.print(json_data[:200] + "..." if len(json_data) > 200 else json_data)

@app.command("metronome")
def metronome(
    tempo: int = typer.Option(60, help="Tempo in BPM", min=40, max=200),
    time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
    measures: int = typer.Option(4, help="Number of measures", min=1, max=16),
    output_dir: str = typer.Option("./output", help="Directory to save output file"),
):
    """Generate a metronome audio file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate metronome
    with console.status(f"[bold green]Generating metronome at {tempo} BPM...[/bold green]"):
        # Extract the actual time signature string from the enum
        time_sig_str = time_signature.value
        metronome_path = create_metronome_audio(tempo, time_sig_str, measures)
    
    if metronome_path:
        # Copy the metronome file to the output directory
        # Replace slash with underscore to avoid file path issues
        safe_time_sig = time_sig_str.replace('/', '_')
        output_path = os.path.join(output_dir, f"metronome_{tempo}bpm_{safe_time_sig}_{measures}m.mp3")
        import shutil
        shutil.copy(metronome_path, output_path)
        
        console.print("\n[bold green]Metronome generated successfully![/bold green]")
        console.print(f"Output File: \n{output_path}")
        return output_path
    else:
        console.print("[bold red]Failed to generate metronome.[/bold red]")

@app.command("convert")
def convert(
    input_file: str = typer.Argument(..., help="Input JSON file path"),
    output_format: OutputFormat = typer.Option(OutputFormat.MIDI, help="Output format"),
    instrument: Instrument = typer.Option(Instrument.TRUMPET, help="Instrument for audio generation"),
    time_signature: TimeSignature = typer.Option(TimeSignature.FOUR_FOUR, help="Time signature"),
    key: Key = typer.Option(Key.C_MAJOR, help="Key signature"),
    tempo: int = typer.Option(60, help="Tempo in BPM", min=40, max=200),
    output_dir: str = typer.Option("./output", help="Directory to save output files"),
    force_fallback: bool = typer.Option(False, help="Force using fallback audio generation instead of soundfonts"),
):
    """Convert a JSON exercise file to MIDI or MP3."""
    # Check if input file exists
    if not os.path.exists(input_file):
        console.print(f"[bold red]Input file not found: {input_file}[/bold red]")
        raise typer.Exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read JSON file
    try:
        with open(input_file, "r") as f:
            json_data = f.read()
        parsed = safe_parse_json(json_data)
        if not parsed:
            console.print("[bold red]Failed to parse JSON file.[/bold red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error reading JSON file: {e}[/bold red]")
        raise typer.Exit(1)
    
    # Generate MIDI
    with console.status("[bold green]Converting to MIDI...[/bold green]"):
        # Calculate measures from JSON data
        total_units = sum(item["duration"] for item in parsed)
        # Extract the actual time signature string from the enum
        time_sig_str = time_signature.value
        numerator, denominator = map(int, time_sig_str.split('/'))
        units_per_measure = numerator * (8 // denominator)
        measures = max(1, round(total_units / units_per_measure))
        
        # Extract string values from enums
        instrument_str = instrument.value
        key_str = key.value
        
        # Generate MIDI
        midi_obj = json_to_midi(parsed, instrument_str, tempo, time_sig_str, measures, key=key_str)
    # Base filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Save outputs based on format
    output_files = []
    
    if output_format in [OutputFormat.MIDI, OutputFormat.ALL]:
        midi_path = os.path.join(output_dir, f"{base_name}.mid")
        midi_obj.save(midi_path)
        output_files.append(("MIDI", midi_path))
    
    if output_format in [OutputFormat.MP3, OutputFormat.ALL]:
        with console.status("[bold green]Converting to MP3...[/bold green]"):
            mp3_path, duration = midi_to_mp3(midi_obj, instrument_str, force_fallback=force_fallback)
            if mp3_path:
                new_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")
                import shutil
                shutil.copy(mp3_path, new_mp3_path)
                output_files.append(("MP3", new_mp3_path))
                console.print(f"[bold green]MP3 conversion successful![/bold green] {'(Using fallback audio generation)' if force_fallback else ''}")
            else:
                console.print("[bold red]MP3 conversion failed.[/bold red]")
                console.print("Trying fallback audio generation...")
                mp3_path, duration = midi_to_mp3(midi_obj, instrument_str, force_fallback=True)
                if mp3_path:
                    new_mp3_path = os.path.join(output_dir, f"{base_name}.mp3")
                    import shutil
                    shutil.copy(mp3_path, new_mp3_path)
                    output_files.append(("MP3", new_mp3_path))
                    console.print("[bold green]MP3 conversion successful using fallback audio generation![/bold green]")
                else:
                    console.print("[bold red]MP3 conversion failed even with fallback audio generation.[/bold red]")
    
    # Display results
    if output_files:
        console.print("\n[bold green]Conversion completed successfully![/bold green]")
        console.print("\n[bold]Output Files:[/bold]")
        for file_type, file_path in output_files:
            console.print(f"[bold]{file_type}:[/bold] {file_path}")
    else:
        console.print("[bold red]No output files were generated.[/bold red]")

@app.command("info")
def info():
    """Display information about available options."""
    console.print("[bold green]Adaptive Music Exercise Generator CLI[/bold green]")
    console.print("\n[bold]Available Instruments:[/bold]")
    for instrument in Instrument:
        console.print(f"- {instrument.value}")
    
    console.print("\n[bold]Difficulty Levels:[/bold]")
    for level in Level:
        console.print(f"- {level.value}")
    
    console.print("\n[bold]Key Signatures:[/bold]")
    for key in Key:
        console.print(f"- {key.value}")
    
    console.print("\n[bold]Time Signatures:[/bold]")
    for ts in TimeSignature:
        console.print(f"- {ts.value}")
    
    console.print("\n[bold]Practice Focus Options:[/bold]")
    for focus in PracticeFocus:
        console.print(f"- {focus.value}")
    
    console.print("\n[bold]Output Formats:[/bold]")
    for fmt in OutputFormat:
        console.print(f"- {fmt.value}")

if __name__ == "__main__":
    app()