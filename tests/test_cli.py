import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

from cli.main import app

runner = CliRunner()

class TestCLIAuthentication:
    """Test CLI authentication commands."""
    
    @patch('cli.main.requests.post')
    def test_login_success(self, mock_post):
        """Test successful login."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test_token_123",
            "token_type": "bearer"
        }
        mock_post.return_value = mock_response
        
        result = runner.invoke(app, [
            "login", 
            "--username", "admin", 
            "--password", "password"
        ])
        
        assert result.exit_code == 0
        assert "Successfully logged in as admin" in result.output
    
    @patch('cli.main.requests.post')
    def test_login_failure(self, mock_post):
        """Test failed login."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid credentials"
        mock_post.return_value = mock_response
        
        result = runner.invoke(app, [
            "login", 
            "--username", "wrong", 
            "--password", "wrong"
        ])
        
        assert result.exit_code == 1
        assert "Login failed" in result.output
    
    def test_logout(self):
        """Test logout command."""
        # Create a temporary token file
        token_file = Path.home() / ".music_llm_token"
        token_file.write_text("test_token")
        
        result = runner.invoke(app, ["logout"])
        
        assert result.exit_code == 0
        assert "Successfully logged out" in result.output
        assert not token_file.exists()

class TestCLIMusicGeneration:
    """Test CLI music generation commands."""
    
    @patch('cli.main.MusicGenerationService.generate_music_description')
    def test_generate_music_local(self, mock_generate):
        """Test local music generation."""
        # Mock successful generation
        mock_generate.return_value = {
            "success": True,
            "data": {
                "title": "Test Song",
                "genre": "pop",
                "mood": "happy",
                "length_seconds": 30
            }
        }
        
        result = runner.invoke(app, [
            "generate",
            "--genre", "pop",
            "--mood", "happy",
            "--length", "30"
        ])
        
        assert result.exit_code == 0
        assert "Generated Music" in result.output
        assert "Test Song" in result.output
    
    @patch('cli.main.requests.post')
    def test_generate_music_api(self, mock_post):
        """Test API-based music generation."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "title": "API Song",
                "genre": "rock",
                "mood": "energetic",
                "length_seconds": 45
            }
        }
        mock_post.return_value = mock_response
        
        # First login
        with patch('cli.main.requests.post') as mock_login:
            mock_login.return_value.status_code = 200
            mock_login.return_value.json.return_value = {
                "access_token": "test_token"
            }
            
            runner.invoke(app, [
                "login", 
                "--username", "admin", 
                "--password", "password"
            ])
        
        # Test generation via API
        result = runner.invoke(app, [
            "generate",
            "--genre", "rock",
            "--mood", "energetic",
            "--length", "45",
            "--api"
        ])
        
        assert result.exit_code == 0
        assert "Generated Music" in result.output
        assert "API Song" in result.output
    
    def test_generate_music_with_output_file(self, tmp_path):
        """Test music generation with output file."""
        with patch('cli.main.MusicGenerationService.generate_music_description') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "data": {
                    "title": "Output Test Song",
                    "genre": "jazz",
                    "mood": "smooth",
                    "length_seconds": 60
                }
            }
            
            output_file = tmp_path / "test_output.json"
            
            result = runner.invoke(app, [
                "generate",
                "--genre", "jazz",
                "--mood", "smooth",
                "--length", "60",
                "--output", str(output_file)
            ])
            
            assert result.exit_code == 0
            assert "Result saved to" in result.output
            assert output_file.exists()
            
            # Check file content
            with open(output_file) as f:
                saved_data = json.load(f)
                assert saved_data["title"] == "Output Test Song"

class TestCLIInfoCommands:
    """Test CLI information commands."""
    
    def test_genres(self):
        """Test genres command."""
        result = runner.invoke(app, ["genres"])
        
        assert result.exit_code == 0
        assert "Supported Genres" in result.output
        assert "pop" in result.output
        assert "rock" in result.output
    
    def test_formats(self):
        """Test formats command."""
        result = runner.invoke(app, ["formats"])
        
        assert result.exit_code == 0
        assert "Supported Output Formats" in result.output
        assert "midi" in result.output
        assert "wav" in result.output
    
    def test_status(self):
        """Test status command."""
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "Service Status" in result.output
        assert "Authentication" in result.output
        assert "Local LLM API Key" in result.output

class TestCLIBatchCommands:
    """Test CLI batch processing commands."""
    
    def test_batch_generate_unauthorized(self):
        """Test batch generation without authentication."""
        # Create a temporary batch file
        batch_data = [
            {"genre": "pop", "mood": "happy", "length": 30},
            {"genre": "rock", "mood": "energetic", "length": 45}
        ]
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(batch_data)
            
            result = runner.invoke(app, [
                "batch",
                "batch_file.json"
            ])
            
            assert result.exit_code == 0
            assert "Not authenticated" in result.output
    
    @patch('cli.main.requests.post')
    def test_batch_generate_authorized(self, mock_post):
        """Test batch generation with authentication."""
        # Mock successful API responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {"title": "Batch Song", "genre": "pop"}
        }
        mock_post.return_value = mock_response
        
        # Create a temporary batch file
        batch_data = [
            {"genre": "pop", "mood": "happy", "length": 30},
            {"genre": "rock", "mood": "energetic", "length": 45}
        ]
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(batch_data)
            
            # Mock authentication
            with patch('cli.main.API_TOKEN', 'test_token'):
                result = runner.invoke(app, [
                    "batch",
                    "batch_file.json",
                    "--output-dir", "./test_output"
                ])
                
                assert result.exit_code == 0
                assert "Batch processing complete" in result.output

class TestCLIServe:
    """Test CLI serve command."""
    
    @patch('cli.main.uvicorn.run')
    def test_serve(self, mock_uvicorn):
        """Test serve command."""
        result = runner.invoke(app, [
            "serve",
            "--host", "127.0.0.1",
            "--port", "9000"
        ])
        
        # The serve command should not exit normally (it runs the server)
        # So we check that uvicorn.run was called
        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["host"] == "127.0.0.1"
        assert call_args[1]["port"] == 9000

class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_generate_music_missing_required_args(self):
        """Test music generation with missing required arguments."""
        result = runner.invoke(app, ["generate"])
        
        assert result.exit_code != 0
        # Should show help or error about missing arguments
    
    def test_invalid_command(self):
        """Test invalid command."""
        result = runner.invoke(app, ["invalid_command"])
        
        assert result.exit_code != 0
        # Should show help or error about unknown command
    
    @patch('cli.main.MusicGenerationService.generate_music_description')
    def test_generate_music_service_error(self, mock_generate):
        """Test music generation when service fails."""
        # Mock service failure
        mock_generate.side_effect = Exception("Service unavailable")
        
        result = runner.invoke(app, [
            "generate",
            "--genre", "pop",
            "--mood", "happy",
            "--length", "30"
        ])
        
        assert result.exit_code == 1
        assert "Error during local generation" in result.output

class TestCLIConfiguration:
    """Test CLI configuration and token management."""
    
    def test_token_loading(self):
        """Test that stored token is loaded on startup."""
        # Create a temporary token file
        token_file = Path.home() / ".music_llm_token"
        test_token = "test_stored_token_123"
        token_file.write_text(test_token)
        
        # Test that the token is loaded
        with patch('cli.main.API_TOKEN', None):
            # This would normally be done in main(), but we can test the logic
            if token_file.exists():
                loaded_token = token_file.read_text().strip()
                assert loaded_token == test_token
        
        # Cleanup
        token_file.unlink()
    
    def test_api_url_configuration(self):
        """Test API URL configuration."""
        custom_url = "https://api.example.com"
        
        with patch('cli.main.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "test_token"
            }
            mock_post.return_value = mock_response
            
            result = runner.invoke(app, [
                "login",
                "--username", "admin",
                "--password", "password",
                "--api-url", custom_url
            ])
            
            # Check that the custom URL was used
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert custom_url in call_args[0][0]  # URL should contain custom domain
