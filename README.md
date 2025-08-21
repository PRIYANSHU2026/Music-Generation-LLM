# Music Generation LLM 🎵🤖

A microservice for AI-powered music generation using Large Language Models. This project provides both a FastAPI-based REST API and a comprehensive CLI tool for generating music compositions based on genre, mood, and other parameters.

## ✨ Features

- **🎼 AI-Powered Music Generation**: Uses LLM models to create detailed music compositions
- **🚀 FastAPI Microservice**: High-performance REST API with automatic documentation
- **🖥️ Rich CLI Interface**: Beautiful command-line interface built with Typer and Rich
- **🔐 JWT Authentication**: Secure API access with token-based authentication
- **📊 Batch Processing**: Generate multiple music pieces in parallel
- **🎵 Multiple Output Formats**: Support for MIDI, WAV, and MP3 formats
- **🌐 CORS Support**: Ready for web application integration
- **📈 Health Monitoring**: Built-in health checks and monitoring endpoints
- **🐳 Docker Ready**: Containerized deployment with Docker

## 🏗️ Architecture

```
music_generation_llm/
├── app/                    # FastAPI application
│   ├── api/               # API routes and endpoints
│   ├── core/              # Configuration and security
│   ├── models/            # Data models and schemas
│   ├── services/          # Business logic services
│   └── db/                # Database models and connections
├── cli/                   # Command-line interface
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
├── Dockerfile            # Docker container
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM integration)
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/music-generation-llm.git
   cd music-generation-llm
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

### Running the Service

#### Option 1: FastAPI Server
```bash
# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or use the CLI
music-llm serve --host 0.0.0.0 --port 8000 --reload
```

#### Option 2: Docker
```bash
# Build the Docker image
docker build -t music-generation-llm .

# Run the container
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here music-generation-llm
```

## 🎯 Usage Examples

### CLI Usage

#### Authentication
```bash
# Login to the service
music-llm login --username admin --password password

# Check status
music-llm status

# Logout
music-llm logout
```

#### Music Generation
```bash
# Generate music locally
music-llm generate --genre pop --mood happy --length 30

# Generate via API (requires authentication)
music-llm generate --genre rock --mood energetic --length 45 --api

# Save output to file
music-llm generate --genre jazz --mood smooth --length 60 --output my_song.json

# Add custom notes
music-llm generate --genre classical --mood dramatic --length 120 --notes "Include a violin solo"
```

#### Batch Processing
```bash
# Create a batch file (batch_requests.json)
[
  {"genre": "pop", "mood": "happy", "length": 30},
  {"genre": "rock", "mood": "energetic", "length": 45},
  {"genre": "jazz", "mood": "smooth", "length": 60}
]

# Process batch requests
music-llm batch batch_requests.json --output-dir ./output
```

#### Information Commands
```bash
# List supported genres
music-llm genres

# List supported output formats
music-llm formats

# Show service status
music-llm status
```

### API Usage

#### Authentication
```bash
# Login to get access token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=password"

# Use the returned token for authenticated requests
export TOKEN="your_access_token_here"
```

#### Generate Music
```bash
# Generate music description
curl -X POST "http://localhost:8000/api/v1/music/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "genre=pop&mood=happy&length=30&additional_notes=Upbeat summer song"
```

#### Get Information
```bash
# Get supported genres
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/music/genres"

# Get supported formats
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/music/formats"

# Check service health
curl "http://localhost:8000/api/v1/music/health"
```

#### Batch Generation
```bash
# Batch generate multiple pieces
curl -X POST "http://localhost:8000/api/v1/music/batch-generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "requests=[{\"genre\":\"pop\",\"mood\":\"happy\",\"length\":30}]"
```

## 🔌 API Endpoints

### Public Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/register` - User registration

### Protected Endpoints (require authentication)
- `POST /api/v1/music/generate` - Generate music description
- `POST /api/v1/music/generate-midi` - Generate MIDI file
- `GET /api/v1/music/genres` - List supported genres
- `GET /api/v1/music/formats` - List supported formats
- `GET /api/v1/music/health` - Music service health
- `POST /api/v1/music/batch-generate` - Batch music generation
- `GET /api/v1/users/me` - Get current user info
- `PUT /api/v1/users/me` - Update current user info

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=cli

# Run specific test files
pytest tests/test_api.py
pytest tests/test_cli.py
```

### Test Coverage
```bash
# Generate coverage report
pytest --cov=app --cov=cli --cov-report=html

# Open coverage report
open htmlcov/index.html
```

## 🐳 Docker Deployment

### Development
```bash
docker build -t music-generation-llm:dev .
docker run -p 8000:8000 -e DEBUG=true music-generation-llm:dev
```

### Production
```bash
# Build production image
docker build -t music-generation-llm:prod .

# Run with production settings
docker run -d \
  -p 8000:8000 \
  -e DEBUG=false \
  -e SECRET_KEY=your_secret_key \
  -e OPENAI_API_KEY=your_openai_key \
  --name music-llm-service \
  music-generation-llm:prod
```

### Docker Compose
```yaml
version: '3.8'
services:
  music-llm:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - SECRET_KEY=your_secret_key
      - OPENAI_API_KEY=your_openai_key
    volumes:
      - ./output:/app/output
    restart: unless-stopped
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `false` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `SECRET_KEY` | JWT secret key | `your-secret-key-change-in-production` |
| `OPENAI_API_KEY` | OpenAI API key | `` |
| `LLM_MODEL_NAME` | LLM model to use | `gpt-3.5-turbo` |
| `MAX_MUSIC_LENGTH` | Maximum music length (seconds) | `300` |
| `DATABASE_URL` | Database connection string | `sqlite:///./music_llm.db` |

### Configuration File
Create a `.env` file in the project root:
```env
DEBUG=true
SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key-here
LLM_MODEL_NAME=gpt-4
MAX_MUSIC_LENGTH=600
```

## 🔧 Development

### Code Quality
```bash
# Format code with Black
black app/ cli/ tests/

# Sort imports with isort
isort app/ cli/ tests/

# Type checking with mypy
mypy app/ cli/

# Linting with flake8
flake8 app/ cli/ tests/
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 📚 API Documentation

Once the service is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc
- **OpenAPI schema**: http://localhost:8000/openapi.json

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the LLM API
- FastAPI for the excellent web framework
- Typer for the CLI framework
- Rich for beautiful terminal output
- The open-source community for various libraries and tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/music-generation-llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/music-generation-llm/discussions)
- **Email**: your.email@example.com

## 🔮 Roadmap

- [ ] Real-time music generation
- [ ] Multiple LLM provider support
- [ ] Advanced music theory integration
- [ ] Web-based music editor
- [ ] Mobile app support
- [ ] Collaborative music creation
- [ ] Music style transfer
- [ ] Integration with DAWs

---

**Made with ❤️ for the music and AI community**



