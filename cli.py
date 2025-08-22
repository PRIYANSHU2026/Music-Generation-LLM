#!/usr/bin/env python

"""Adaptive Music Generator CLI

This is the main entry point for the Adaptive Music Generator CLI.
It imports and runs the CLI from the adaptive_music_generator package.
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_music_generator.cli import app

if __name__ == "__main__":
    app()