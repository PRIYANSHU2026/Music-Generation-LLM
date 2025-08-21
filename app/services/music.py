from typing import Dict, List, Optional, Any
import logging
import json
import asyncio
from datetime import datetime
import openai
from app.core.config import settings

logger = logging.getLogger(__name__)

class MusicGenerationService:
    """Service for generating music using LLM models."""
    
    def __init__(self):
        self.openai_client = None
        if settings.LLM_API_KEY:
            openai.api_key = settings.LLM_API_KEY
            self.openai_client = openai
        
        self.supported_genres = settings.SUPPORTED_GENRES
        self.output_formats = settings.OUTPUT_FORMATS
        self.max_length = settings.MAX_MUSIC_LENGTH
    
    async def generate_music_description(
        self, 
        genre: str, 
        mood: str, 
        length: int,
        additional_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate music description using LLM."""
        try:
            if not self.openai_client:
                raise Exception("OpenAI API key not configured")
            
            prompt = self._build_music_prompt(genre, mood, length, additional_notes)
            
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=settings.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a music composition expert. Generate detailed music descriptions in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE
            )
            
            content = response.choices[0].message.content
            # Try to parse JSON from the response
            try:
                music_data = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                music_data = {
                    "description": content,
                    "genre": genre,
                    "mood": mood,
                    "length": length,
                    "generated_at": datetime.utcnow().isoformat()
                }
            
            return {
                "success": True,
                "data": music_data,
                "model_used": settings.LLM_MODEL_NAME
            }
            
        except Exception as e:
            logger.error(f"Error generating music description: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback": self._generate_fallback_description(genre, mood, length)
            }
    
    def _build_music_prompt(self, genre: str, mood: str, length: int, additional_notes: Optional[str]) -> str:
        """Build the prompt for music generation."""
        prompt = f"""
        Generate a detailed music composition description for:
        - Genre: {genre}
        - Mood: {mood}
        - Length: {length} seconds
        """
        
        if additional_notes:
            prompt += f"\n- Additional Notes: {additional_notes}"
        
        prompt += """
        
        Please provide the response in JSON format with the following structure:
        {
            "title": "Song Title",
            "genre": "genre",
            "mood": "mood",
            "length_seconds": length,
            "tempo_bpm": 120,
            "key": "C major",
            "structure": {
                "intro": "description",
                "verse": "description",
                "chorus": "description",
                "bridge": "description",
                "outro": "description"
            },
            "instruments": ["piano", "guitar", "drums"],
            "chord_progression": ["C", "Am", "F", "G"],
            "melody_description": "description of the main melody",
            "dynamics": "description of volume and intensity changes"
        }
        """
        
        return prompt
    
    def _generate_fallback_description(self, genre: str, mood: str, length: int) -> Dict[str, Any]:
        """Generate a fallback description when LLM fails."""
        return {
            "title": f"{mood.title()} {genre.title()} Song",
            "genre": genre,
            "mood": mood,
            "length_seconds": length,
            "tempo_bpm": 120,
            "key": "C major",
            "structure": {
                "intro": f"Gentle {genre} introduction with {mood} atmosphere",
                "verse": f"Main {genre} verse with {mood} emotional content",
                "chorus": f"Memorable {genre} chorus with {mood} intensity",
                "bridge": f"Transitional {genre} bridge",
                "outro": f"Fading {genre} conclusion"
            },
            "instruments": ["piano", "guitar", "drums"],
            "chord_progression": ["C", "Am", "F", "G"],
            "melody_description": f"A {mood} {genre} melody that flows naturally",
            "dynamics": "Gradual build-up and gentle fade-out",
            "fallback_generated": True
        }
    
    async def generate_midi_from_description(self, music_description: Dict[str, Any]) -> bytes:
        """Generate MIDI file from music description (placeholder)."""
        # This is a placeholder - in a real implementation, you would:
        # 1. Parse the music description
        # 2. Use a music generation library (like pretty_midi, mido, etc.)
        # 3. Generate actual MIDI data
        
        logger.info(f"Generating MIDI for: {music_description.get('title', 'Unknown')}")
        
        # Placeholder MIDI data (empty MIDI file)
        midi_data = b'MThd\x00\x00\x00\x06\x00\x01\x00\x60\x00\x80MTrk\x00\x00\x00\x0b\x00\xff\x2f\x00'
        
        return midi_data
    
    def validate_genre(self, genre: str) -> bool:
        """Validate if the genre is supported."""
        return genre.lower() in [g.lower() for g in self.supported_genres]
    
    def validate_length(self, length: int) -> bool:
        """Validate if the length is within acceptable range."""
        return 10 <= length <= self.max_length
    
    def get_supported_genres(self) -> List[str]:
        """Get list of supported genres."""
        return self.supported_genres.copy()
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported output formats."""
        return self.output_formats.copy()
