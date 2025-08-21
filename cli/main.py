import typer
import asyncio
import json
import requests
from typing import Optional, List
from pathlib import Path
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from app.services.music import MusicGenerationService
from app.core.config import settings

# Initialize Typer app
app = typer.Typer(
    name="music-llm",
    help="Music Generation LLM CLI Tool",
    add_completion=False
)

# Initialize console and services
console = Console()
music_service = MusicGenerationService()

# Global variables for API configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = None

def get_auth_headers():
    """Get authentication headers for API requests."""
    if API_TOKEN:
        return {"Authorization": f"Bearer {API_TOKEN}"}
    return {}

@app.command()
def login(
    username: str = typer.Option(..., "--username", "-u", help="Username for authentication"),
    password: str = typer.Option(..., "--password", "-p", help="Password for authentication"),
    api_url: str = typer.Option("http://localhost:8000", "--api-url", help="API base URL")
):
    """Login to the Music Generation LLM service."""
    global API_BASE_URL, API_TOKEN
    
    try:
        API_BASE_URL = api_url.rstrip('/')
        
        with console.status("[bold green]Logging in..."):
            response = requests.post(
                f"{API_BASE_URL}/api/v1/auth/login",
                data={"username": username, "password": password}
            )
        
        if response.status_code == 200:
            data = response.json()
            API_TOKEN = data["access_token"]
            console.print(f"[bold green]✓ Successfully logged in as {username}[/bold green]")
            
            # Save token to file
            token_file = Path.home() / ".music_llm_token"
            token_file.write_text(API_TOKEN)
            console.print(f"[dim]Token saved to {token_file}[/dim]")
            
        else:
            console.print(f"[bold red]✗ Login failed: {response.text}[/bold red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[bold red]✗ Error during login: {str(e)}[/bold red]")
        raise typer.Exit(1)

@app.command()
def logout():
    """Logout and clear stored token."""
    global API_TOKEN
    
    # Remove token file
    token_file = Path.home() / ".music_llm_token"
    if token_file.exists():
        token_file.unlink()
        console.print("[dim]Stored token removed[/dim]")
    
    API_TOKEN = None
    console.print("[bold green]✓ Successfully logged out[/bold green]")

@app.command()
def generate(
    genre: str = typer.Option(..., "--genre", "-g", help="Music genre"),
    mood: str = typer.Option(..., "--mood", "-m", help="Music mood"),
    length: int = typer.Option(30, "--length", "-l", help="Length in seconds"),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="Additional notes"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    api_mode: bool = typer.Option(False, "--api", help="Use API instead of local generation")
):
    """Generate music using the LLM service."""
    
    if api_mode:
        _generate_via_api(genre, mood, length, notes, output_file)
    else:
        _generate_locally(genre, mood, length, notes, output_file)

def _generate_via_api(genre: str, mood: str, length: int, notes: Optional[str], output_file: Optional[str]):
    """Generate music via API."""
    if not API_TOKEN:
        console.print("[bold red]✗ Not authenticated. Please login first.[/bold red]")
        raise typer.Exit(1)
    
    try:
        with console.status("[bold green]Generating music via API..."):
            response = requests.post(
                f"{API_BASE_URL}/api/v1/music/generate",
                headers=get_auth_headers(),
                data={
                    "genre": genre,
                    "mood": mood,
                    "length": length,
                    "additional_notes": notes or ""
                }
            )
        
        if response.status_code == 200:
            data = response.json()
            if data["success"]:
                _display_music_result(data["data"], output_file)
            else:
                console.print(f"[yellow]⚠ Warning: {data.get('warning', 'Unknown warning')}[/yellow]")
                _display_music_result(data["data"], output_file)
        else:
            console.print(f"[bold red]✗ API request failed: {response.text}[/bold red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[bold red]✗ Error during API generation: {str(e)}[/bold red]")
        raise typer.Exit(1)

async def _generate_locally(genre: str, mood: str, length: int, notes: Optional[str], output_file: Optional[str]):
    """Generate music locally using the service."""
    try:
        with console.status("[bold green]Generating music locally..."):
            result = await music_service.generate_music_description(
                genre=genre,
                mood=mood,
                length=length,
                additional_notes=notes
            )
        
        if result["success"]:
            _display_music_result(result["data"], output_file)
        else:
            console.print(f"[yellow]⚠ LLM generation failed, using fallback[/yellow]")
            _display_music_result(result["fallback"], output_file)
            
    except Exception as e:
        console.print(f"[bold red]✗ Error during local generation: {str(e)}[/bold red]")
        raise typer.Exit(1)

def _display_music_result(music_data: dict, output_file: Optional[str]):
    """Display music generation result."""
    # Create a rich table for the result
    table = Table(title="Generated Music", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in music_data.items():
        if isinstance(value, dict):
            # Handle nested structures
            table.add_row(key, json.dumps(value, indent=2))
        elif isinstance(value, list):
            table.add_row(key, ", ".join(map(str, value)))
        else:
            table.add_row(key, str(value))
    
    console.print(table)
    
    # Save to file if specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(music_data, f, indent=2)
            console.print(f"[green]✓ Result saved to {output_file}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Error saving to file: {str(e)}[/red]")

@app.command()
def genres():
    """List supported music genres."""
    if API_TOKEN:
        # Get from API
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/music/genres",
                headers=get_auth_headers()
            )
            if response.status_code == 200:
                data = response.json()
                genres_list = data["genres"]
            else:
                genres_list = music_service.get_supported_genres()
        except:
            genres_list = music_service.get_supported_genres()
    else:
        genres_list = music_service.get_supported_genres()
    
    # Display genres
    table = Table(title="Supported Genres", show_header=True, header_style="bold magenta")
    table.add_column("Genre", style="cyan")
    
    for genre in genres_list:
        table.add_row(genre)
    
    console.print(table)

@app.command()
def formats():
    """List supported output formats."""
    if API_TOKEN:
        # Get from API
        try:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/music/formats",
                headers=get_auth_headers()
            )
            if response.status_code == 200:
                data = response.json()
                formats_list = data["formats"]
            else:
                formats_list = music_service.get_supported_formats()
        except:
            formats_list = music_service.get_supported_formats()
    else:
        formats_list = music_service.get_supported_formats()
    
    # Display formats
    table = Table(title="Supported Output Formats", show_header=True, header_style="bold magenta")
    table.add_column("Format", style="cyan")
    
    for format_type in formats_list:
        table.add_row(format_type)
    
    console.print(table)

@app.command()
def status():
    """Show service status and configuration."""
    # Create status panel
    status_info = []
    
    # Authentication status
    if API_TOKEN:
        status_info.append(f"Authentication: [green]✓ Authenticated[/green]")
        status_info.append(f"API URL: {API_BASE_URL}")
    else:
        status_info.append("Authentication: [red]✗ Not authenticated[/red]")
    
    # Local service status
    status_info.append(f"Local LLM API Key: {'[green]✓ Configured[/green]' if settings.LLM_API_KEY else '[red]✗ Not configured[/red]'}")
    status_info.append(f"Supported Genres: {len(music_service.get_supported_genres())}")
    status_info.append(f"Supported Formats: {len(music_service.get_supported_formats())}")
    status_info.append(f"Max Music Length: {settings.MAX_MUSIC_LENGTH}s")
    
    status_panel = Panel(
        "\n".join(status_info),
        title="Service Status",
        border_style="blue"
    )
    
    console.print(status_panel)

@app.command()
def batch(
    input_file: str = typer.Argument(..., help="JSON file with batch generation requests"),
    output_dir: str = typer.Option("./output", "--output-dir", "-o", help="Output directory for results")
):
    """Generate multiple music pieces from a batch file."""
    
    if not API_TOKEN:
        console.print("[bold red]✗ Not authenticated. Please login first.[/bold red]")
        raise typer.Exit(1)
    
    # Read batch file
    try:
        with open(input_file, 'r') as f:
            batch_requests = json.load(f)
        
        if not isinstance(batch_requests, list):
            console.print("[bold red]✗ Batch file must contain a list of requests[/bold red]")
            raise typer.Exit(1)
        
        console.print(f"[green]✓ Loaded {len(batch_requests)} requests from {input_file}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Error reading batch file: {str(e)}[/bold red]")
        raise typer.Exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process batch requests
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing batch requests...", total=len(batch_requests))
        
        for i, request in enumerate(batch_requests):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/music/generate",
                    headers=get_auth_headers(),
                    data={
                        "genre": request.get("genre", "pop"),
                        "mood": request.get("mood", "happy"),
                        "length": request.get("length", 30),
                        "additional_notes": request.get("additional_notes", "")
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Save individual result
                    result_file = output_path / f"result_{i+1:03d}.json"
                    with open(result_file, 'w') as f:
                        json.dump(data, f, indent=2)
                
                progress.advance(task)
                
            except Exception as e:
                console.print(f"[red]✗ Error processing request {i+1}: {str(e)}[/red]")
                progress.advance(task)
    
    console.print(f"[green]✓ Batch processing complete. Results saved to {output_dir}[/green]")

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload")
):
    """Start the FastAPI server."""
    import uvicorn
    
    console.print(f"[bold green]Starting Music Generation LLM server on {host}:{port}[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

def main():
    """Main entry point."""
    # Try to load stored token
    global API_TOKEN
    token_file = Path.home() / ".music_llm_token"
    if token_file.exists():
        try:
            API_TOKEN = token_file.read_text().strip()
            console.print("[dim]Loaded stored authentication token[/dim]")
        except:
            pass
    
    app()

if __name__ == "__main__":
    main()
