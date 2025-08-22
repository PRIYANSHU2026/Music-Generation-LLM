"""Command-line interface for the Adaptive Music Generator.

This module provides a CLI for generating music exercises, converting files,
and other utility functions.
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from .lib import generate_music, generate_metronome
from .processors import midi_to_audio, generate_sheet_music
from .exceptions import MusicGenerationError, AudioConversionError, VisualizationError

# Create Typer app
app = typer.Typer(help="Adaptive Music Generator CLI")
console = Console()


def ensure_dependencies() -> None:
    """Ensure that all required dependencies are installed."""
    try:
        import mido
        import requests
    except ImportError:
        console.print("[bold red]Missing required dependencies. Installing...[/bold red]")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            console.print("[bold green]Dependencies installed successfully![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Failed to install dependencies: {str(e)}[/bold red]")
            sys.exit(1)


@app.command("generate")
def generate(
    instrument: str = typer.Option("Piano", help="Instrument to generate for"),
    level: str = typer.Option("Beginner", help="Difficulty level"),
    key: str = typer.Option("C Major", help="Musical key"),
    time_signature: str = typer.Option("4/4", help="Time signature"),
    measures: int = typer.Option(4, help="Number of measures"),
    focus: str = typer.Option("Melodic", help="Practice focus"),
    output_format: str = typer.Option("midi", help="Output format (midi, mp3, json)"),
    output_dir: str = typer.Option("output", help="Output directory"),
    force_fallback: bool = typer.Option(False, help="Force fallback audio generation")
) -> None:
    """Generate a music exercise with the specified parameters."""
    ensure_dependencies()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique ID for this exercise
    exercise_id = str(uuid.uuid4())[:8]
    
    try:
        # Show parameters
        console.print(f"[bold]Generating exercise with ID: {exercise_id}[/bold]")
        console.print(f"Instrument: {instrument}")
        console.print(f"Level: {level}")
        console.print(f"Key: {key}")
        console.print(f"Time Signature: {time_signature}")
        console.print(f"Measures: {measures}")
        console.print(f"Focus: {focus}")
        
        # Generate music
        params = {
            "instrument": instrument,
            "level": level,
            "key": key,
            "time_signature": time_signature,
            "measures": measures,
            "focus": focus
        }
        
        result = generate_music(params)
        exercise_data = result["exercise_data"]
        midi_file = result["midi_file"]
        
        # Save outputs based on format
        base_path = os.path.join(output_dir, f"exercise_{exercise_id}")
        
        # Always save JSON
        json_path = f"{base_path}.json"
        with open(json_path, "w") as f:
            json.dump(exercise_data, f, indent=2)
        
        # Save MIDI
        midi_path = f"{base_path}.mid"
        midi_file.save(midi_path)
        
        # Generate sheet music
        html_path = f"{base_path}.html"
        generate_sheet_music(exercise_data, html_path)
        
        # Convert to audio if requested
        if output_format.lower() in ["mp3", "wav"]:
            audio_path = f"{base_path}.{output_format.lower()}"
            midi_to_audio(midi_file, audio_path, instrument, output_format.lower())
            console.print(f"[bold green]Audio saved to: {audio_path}[/bold green]")
        
        console.print(f"[bold green]Exercise generated successfully![/bold green]")
        console.print(f"MIDI saved to: {midi_path}")
        console.print(f"JSON saved to: {json_path}")
        console.print(f"Sheet music saved to: {html_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error generating exercise: {str(e)}[/bold red]")
        sys.exit(1)


@app.command("metronome")
def metronome(
    tempo: int = typer.Option(120, help="Tempo in BPM"),
    time_signature: str = typer.Option("4/4", help="Time signature"),
    measures: int = typer.Option(4, help="Number of measures"),
    output_format: str = typer.Option("midi", help="Output format (midi, mp3)"),
    output_dir: str = typer.Option("output", help="Output directory")
) -> None:
    """Generate a metronome track with the specified parameters."""
    ensure_dependencies()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Show parameters
        console.print(f"[bold]Generating metronome track[/bold]")
        console.print(f"Tempo: {tempo} BPM")
        console.print(f"Time Signature: {time_signature}")
        console.print(f"Measures: {measures}")
        
        # Generate metronome
        midi_file = generate_metronome(tempo, time_signature, measures)
        
        # Save outputs based on format
        base_path = os.path.join(output_dir, f"metronome_{tempo}bpm_{time_signature.replace('/', '-')}")
        
        # Save MIDI
        midi_path = f"{base_path}.mid"
        midi_file.save(midi_path)
        
        # Convert to audio if requested
        if output_format.lower() in ["mp3", "wav"]:
            audio_path = f"{base_path}.{output_format.lower()}"
            midi_to_audio(midi_file, audio_path, "Piano", output_format.lower())
            console.print(f"[bold green]Audio saved to: {audio_path}[/bold green]")
        
        console.print(f"[bold green]Metronome generated successfully![/bold green]")
        console.print(f"MIDI saved to: {midi_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error generating metronome: {str(e)}[/bold red]")
        sys.exit(1)


@app.command("convert")
def convert(
    input_file: str = typer.Argument(..., help="Input file path (MIDI or JSON)"),
    output_format: str = typer.Option("mp3", help="Output format (midi, mp3, json, html)"),
    instrument: str = typer.Option("Piano", help="Instrument for audio conversion")
) -> None:
    """Convert between different file formats."""
    ensure_dependencies()


@app.command("download-soundfonts")
def download_soundfonts() -> None:
    """Download all available soundfonts for offline use."""
    from .processors import download_all_soundfonts
    
    try:
        console.print("[bold]Downloading all soundfonts for offline use...[/bold]")
        download_all_soundfonts()
        console.print("[bold green]All soundfonts downloaded successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error downloading soundfonts: {str(e)}[/bold red]")
        sys.exit(1)
    
    try:
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[bold red]Input file not found: {input_file}[/bold red]")
            sys.exit(1)
        
        # Determine input format
        input_format = input_path.suffix.lower()[1:]
        output_path = input_path.with_suffix(f".{output_format.lower()}")
        
        console.print(f"[bold]Converting {input_format} to {output_format}[/bold]")
        
        if input_format == "json" and output_format.lower() in ["midi", "mid"]:
            # JSON to MIDI
            with open(input_path, "r") as f:
                exercise_data = json.load(f)
            
            from .lib import create_midi_file
            midi_file = create_midi_file(exercise_data)
            midi_file.save(output_path)
            
        elif input_format == "json" and output_format.lower() == "html":
            # JSON to HTML (sheet music)
            with open(input_path, "r") as f:
                exercise_data = json.load(f)
            
            generate_sheet_music(exercise_data, output_path)
            
        elif input_format in ["midi", "mid"] and output_format.lower() in ["mp3", "wav"]:
            # MIDI to audio
            import mido
            midi_file = mido.MidiFile(input_path)
            midi_to_audio(midi_file, output_path, instrument, output_format.lower())
            
        elif input_format in ["midi", "mid"] and output_format.lower() == "json":
            # MIDI to JSON
            console.print("[bold yellow]MIDI to JSON conversion not yet implemented[/bold yellow]")
            sys.exit(1)
            
        else:
            console.print(f"[bold red]Unsupported conversion: {input_format} to {output_format}[/bold red]")
            sys.exit(1)
        
        console.print(f"[bold green]Conversion successful![/bold green]")
        console.print(f"Output saved to: {output_path}")
        
    except Exception as e:
        console.print(f"[bold red]Error during conversion: {str(e)}[/bold red]")
        sys.exit(1)


@app.command("info")
def info() -> None:
    """Display information about available options."""
    table = Table(title="Adaptive Music Generator Options")
    
    table.add_column("Parameter", style="cyan")
    table.add_column("Options", style="green")
    table.add_column("Description", style="yellow")
    
    table.add_row(
        "Instrument",
        "Piano, Trumpet, Violin, Clarinet, Flute",
        "Instrument to generate for"
    )
    
    table.add_row(
        "Level",
        "Beginner, Intermediate, Advanced",
        "Difficulty level"
    )
    
    table.add_row(
        "Key",
        "C Major, G Major, D Major, A Major, E Major, B Major, F# Major, C# Major, F Major, Bb Major, Eb Major, Ab Major, Db Major, Gb Major, Cb Major, A Minor, E Minor, B Minor, F# Minor, C# Minor, G# Minor, D# Minor, A# Minor, D Minor, G Minor, C Minor, F Minor, Bb Minor, Eb Minor, Ab Minor",
        "Musical key"
    )
    
    table.add_row(
        "Time Signature",
        "4/4, 3/4, 2/4, 6/8, 9/8, 12/8, 5/4, 7/8",
        "Time signature"
    )
    
    table.add_row(
        "Focus",
        "Rhythmic, Melodic, Technical, Expressive, Sight-Reading, Improvisation",
        "Practice focus"
    )
    
    table.add_row(
        "Output Format",
        "midi, mp3, json, html",
        "Output file format"
    )
    
    console.print(table)
    
    console.print("\n[bold]Example Commands:[/bold]")
    console.print("  Generate an exercise: python -m adaptive_music_generator.cli generate --instrument Piano --level Beginner --key 'C Major' --time-signature 4/4 --measures 4 --output-format mp3")
    console.print("  Generate a metronome: python -m adaptive_music_generator.cli metronome --tempo 120 --time-signature 4/4 --measures 4 --output-format mp3")
    console.print("  Convert a file: python -m adaptive_music_generator.cli convert exercise.json --output-format mp3")


if __name__ == "__main__":
    app()