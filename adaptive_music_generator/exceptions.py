"""Custom exceptions for the Adaptive Music Generator.

This module defines custom exceptions used throughout the application.
"""


class MusicGenerationError(Exception):
    """Base exception for music generation errors."""
    pass


class InvalidParameterError(MusicGenerationError):
    """Exception raised when an invalid parameter is provided."""
    pass


class AudioConversionError(MusicGenerationError):
    """Exception raised when audio conversion fails."""
    pass


class VisualizationError(MusicGenerationError):
    """Exception raised when visualization generation fails."""
    pass


class LLMQueryError(MusicGenerationError):
    """Exception raised when querying the LLM API fails."""
    pass