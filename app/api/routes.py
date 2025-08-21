from fastapi import APIRouter, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List
import io
import json
from datetime import datetime

from app.services.music import MusicGenerationService
from app.models.user import User
from app.core.security import get_current_user
from app.core.config import settings

router = APIRouter()

# Initialize music service
music_service = MusicGenerationService()

# Authentication endpoints
@router.post("/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """User login endpoint."""
    # This is a simplified login - in production, validate against database
    if username == "admin" and password == "password":
        from app.core.security import create_access_token
        token = create_access_token(data={"sub": username})
        return {"access_token": token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

@router.post("/auth/register")
async def register(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    """User registration endpoint."""
    # This is a simplified registration - in production, save to database
    return {"message": "User registered successfully", "username": username}

# Music generation endpoints
@router.post("/music/generate")
async def generate_music(
    genre: str = Form(...),
    mood: str = Form(...),
    length: int = Form(30),
    additional_notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Generate music description using LLM."""
    try:
        # Validate inputs
        if not music_service.validate_genre(genre):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported genre. Supported genres: {music_service.get_supported_genres()}"
            )
        
        if not music_service.validate_length(length):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Length must be between 10 and {settings.MAX_MUSIC_LENGTH} seconds"
            )
        
        # Generate music description
        result = await music_service.generate_music_description(
            genre=genre,
            mood=mood,
            length=length,
            additional_notes=additional_notes
        )
        
        if result["success"]:
            return {
                "success": True,
                "data": result["data"],
                "user": current_user.username,
                "generated_at": datetime.utcnow().isoformat()
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "success": False,
                    "warning": "LLM generation failed, using fallback",
                    "data": result["fallback"],
                    "user": current_user.username,
                    "generated_at": datetime.utcnow().isoformat()
                }
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating music: {str(e)}"
        )

@router.post("/music/generate-midi")
async def generate_midi(
    music_description: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Generate MIDI file from music description."""
    try:
        # Parse music description
        try:
            description_data = json.loads(music_description)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format for music description"
            )
        
        # Generate MIDI
        midi_data = await music_service.generate_midi_from_description(description_data)
        
        # Return MIDI file as streaming response
        return StreamingResponse(
            io.BytesIO(midi_data),
            media_type="audio/midi",
            headers={
                "Content-Disposition": f"attachment; filename=generated_music_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mid"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating MIDI: {str(e)}"
        )

@router.get("/music/genres")
async def get_supported_genres(current_user: User = Depends(get_current_user)):
    """Get list of supported music genres."""
    return {
        "genres": music_service.get_supported_genres(),
        "total": len(music_service.get_supported_genres())
    }

@router.get("/music/formats")
async def get_supported_formats(current_user: User = Depends(get_current_user)):
    """Get list of supported output formats."""
    return {
        "formats": music_service.get_supported_formats(),
        "total": len(music_service.get_supported_formats())
    }

@router.get("/music/health")
async def music_service_health():
    """Check music service health."""
    return {
        "service": "Music Generation Service",
        "status": "healthy",
        "llm_configured": bool(settings.LLM_API_KEY),
        "supported_genres": len(music_service.get_supported_genres()),
        "supported_formats": len(music_service.get_supported_formats())
    }

# User management endpoints
@router.get("/users/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active
    }

@router.put("/users/me")
async def update_current_user(
    full_name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    """Update current user information."""
    # This is a simplified update - in production, update database
    return {
        "message": "User updated successfully",
        "username": current_user.username,
        "updated_fields": [k for k, v in locals().items() if v is not None and k not in ['current_user']]
    }

# Batch music generation endpoint
@router.post("/music/batch-generate")
async def batch_generate_music(
    requests: str = Form(...),  # JSON string of multiple requests
    current_user: User = Depends(get_current_user)
):
    """Generate multiple music pieces in batch."""
    try:
        # Parse batch requests
        try:
            batch_requests = json.loads(requests)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format for batch requests"
            )
        
        if not isinstance(batch_requests, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch requests must be a list"
            )
        
        if len(batch_requests) > 10:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 10 requests"
            )
        
        # Process batch requests
        results = []
        for i, request in enumerate(batch_requests):
            try:
                result = await music_service.generate_music_description(
                    genre=request.get("genre", "pop"),
                    mood=request.get("mood", "happy"),
                    length=request.get("length", 30),
                    additional_notes=request.get("additional_notes")
                )
                results.append({
                    "request_id": i,
                    "success": True,
                    "data": result.get("data", result.get("fallback"))
                })
            except Exception as e:
                results.append({
                    "request_id": i,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "batch_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "total_requests": len(batch_requests),
            "successful": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing batch requests: {str(e)}"
        )
