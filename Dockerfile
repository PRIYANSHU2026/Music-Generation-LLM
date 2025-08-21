FROM python:3.10-slim

# Install system dependencies required for audio and soundfont rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    fluidsynth \
    libsndfile1 \
    ffmpeg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir typer[all] rich

# Copy the rest of the application
COPY . .

# Pre-create necessary directories
RUN mkdir -p /app/static /app/output /app/temp_audio /app/soundfonts

# Optionally prefetch core soundfont for offline runs (best-effort)
RUN wget -q -O /app/soundfonts/Trumpet.sf2 https://github.com/FluidSynth/fluidsynth/raw/master/sf2/Trumpet.sf2 || true

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

# Default command runs the CLI help. Override with: docker run <img> python cli.py generate ...
CMD ["python", "cli.py", "info"]