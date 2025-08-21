import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from app.main import app
from app.core.config import settings

client = TestClient(app)

class TestAuthentication:
    """Test authentication endpoints."""
    
    def test_login_success(self):
        """Test successful login."""
        response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_failure(self):
        """Test failed login."""
        response = client.post("/api/v1/auth/login", data={
            "username": "wrong",
            "password": "wrong"
        })
        assert response.status_code == 401
    
    def test_register(self):
        """Test user registration."""
        response = client.post("/api/v1/auth/register", data={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpass123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"

class TestMusicGeneration:
    """Test music generation endpoints."""
    
    def test_generate_music_unauthorized(self):
        """Test music generation without authentication."""
        response = client.post("/api/v1/music/generate", data={
            "genre": "pop",
            "mood": "happy",
            "length": 30
        })
        assert response.status_code == 401
    
    @patch('app.services.music.MusicGenerationService.generate_music_description')
    def test_generate_music_success(self, mock_generate):
        """Test successful music generation."""
        # Mock the service response
        mock_generate.return_value = {
            "success": True,
            "data": {
                "title": "Test Song",
                "genre": "pop",
                "mood": "happy",
                "length_seconds": 30
            }
        }
        
        # First login to get token
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test music generation with token
        response = client.post("/api/v1/music/generate", 
            data={
                "genre": "pop",
                "mood": "happy",
                "length": 30
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["title"] == "Test Song"
    
    def test_generate_music_invalid_genre(self):
        """Test music generation with invalid genre."""
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test with invalid genre
        response = client.post("/api/v1/music/generate",
            data={
                "genre": "invalid_genre",
                "mood": "happy",
                "length": 30
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Unsupported genre" in response.json()["detail"]
    
    def test_generate_music_invalid_length(self):
        """Test music generation with invalid length."""
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test with invalid length
        response = client.post("/api/v1/music/generate",
            data={
                "genre": "pop",
                "mood": "happy",
                "length": 5  # Too short
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Length must be between" in response.json()["detail"]

class TestMusicEndpoints:
    """Test other music-related endpoints."""
    
    def test_get_genres_unauthorized(self):
        """Test getting genres without authentication."""
        response = client.get("/api/v1/music/genres")
        assert response.status_code == 401
    
    def test_get_genres_authorized(self):
        """Test getting genres with authentication."""
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test getting genres
        response = client.get("/api/v1/music/genres",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "genres" in data
        assert "total" in data
        assert isinstance(data["genres"], list)
    
    def test_get_formats_authorized(self):
        """Test getting formats with authentication."""
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test getting formats
        response = client.get("/api/v1/music/formats",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert "total" in data
        assert isinstance(data["formats"], list)
    
    def test_music_health(self):
        """Test music service health endpoint."""
        response = client.get("/api/v1/music/health")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Music Generation Service"
        assert data["status"] == "healthy"

class TestUserEndpoints:
    """Test user management endpoints."""
    
    def test_get_user_info_unauthorized(self):
        """Test getting user info without authentication."""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401
    
    def test_get_user_info_authorized(self):
        """Test getting user info with authentication."""
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test getting user info
        response = client.get("/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert "email" in data
        assert "full_name" in data

class TestPublicEndpoints:
    """Test public endpoints that don't require authentication."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Music Generation LLM"
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Music Generation LLM API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data

class TestBatchGeneration:
    """Test batch music generation."""
    
    def test_batch_generate_unauthorized(self):
        """Test batch generation without authentication."""
        batch_requests = json.dumps([
            {"genre": "pop", "mood": "happy", "length": 30},
            {"genre": "rock", "mood": "energetic", "length": 45}
        ])
        
        response = client.post("/api/v1/music/batch-generate", data={
            "requests": batch_requests
        })
        assert response.status_code == 401
    
    @patch('app.services.music.MusicGenerationService.generate_music_description')
    def test_batch_generate_success(self, mock_generate):
        """Test successful batch generation."""
        # Mock the service response
        mock_generate.return_value = {
            "success": True,
            "data": {"title": "Test Song", "genre": "pop"}
        }
        
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test batch generation
        batch_requests = json.dumps([
            {"genre": "pop", "mood": "happy", "length": 30},
            {"genre": "rock", "mood": "energetic", "length": 45}
        ])
        
        response = client.post("/api/v1/music/batch-generate",
            data={"requests": batch_requests},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_requests"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2
    
    def test_batch_generate_invalid_json(self):
        """Test batch generation with invalid JSON."""
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test with invalid JSON
        response = client.post("/api/v1/music/batch-generate",
            data={"requests": "invalid json"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Invalid JSON format" in response.json()["detail"]
    
    def test_batch_generate_too_many_requests(self):
        """Test batch generation with too many requests."""
        # Login first
        login_response = client.post("/api/v1/auth/login", data={
            "username": "admin",
            "password": "password"
        })
        token = login_response.json()["access_token"]
        
        # Test with too many requests
        batch_requests = json.dumps([{"genre": "pop", "mood": "happy", "length": 30}] * 11)
        
        response = client.post("/api/v1/music/batch-generate",
            data={"requests": batch_requests},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Batch size cannot exceed 10" in response.json()["detail"]
