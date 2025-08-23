#!/usr/bin/env python

"""
Music Exercise Generator
=====================
Core functionality for generating music exercises using LLM.
"""

import json
import random
import re
import os
import requests
from typing import Optional, List, Tuple, Dict, Any

# Default API settings
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "bPj0wARXs5dk2L1ipFOdoqHMmQnXuMNv")


def scale_json_durations(json_data: List[List[Any]], target_units: int) -> List[List[Any]]:
    """
    Scales durations so that their sum is exactly target_units (8th notes).
    
    Args:
        json_data: List of [note, duration] pairs
        target_units: Target total duration in 8th note units
        
    Returns:
        Scaled list of [note, duration] pairs
    """
    durations = [int(d) for _, d in json_data]
    total = sum(durations)
    if total == 0:
        return json_data

    # Calculate proportional scaling with integer arithmetic
    scaled = []
    remainder = target_units
    for i, (note, d) in enumerate(json_data):
        if i < len(json_data) - 1:
            # Proportional allocation
            portion = max(1, round(d * target_units / total))
            scaled.append([note, portion])
            remainder -= portion
        else:
            # Last note gets all remaining units
            scaled.append([note, max(1, remainder)])

    return scaled


def safe_parse_json(text: str) -> Optional[List]:
    """
    Safely parse JSON from LLM outputs, handling common formatting issues.
    
    Args:
        text: JSON string to parse
        
    Returns:
        Parsed JSON data or None if parsing fails
    """
    try:
        text = text.replace("'", '"')
        match = re.search(r"\[(\s*\[.*?\]\s*,?)*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}\nRaw text: {text}")
        return None


def get_style_based_on_level(level: str) -> str:
    """
    Get a random musical style appropriate for the given difficulty level.
    
    Args:
        level: Difficulty level (Beginner, Intermediate, Advanced)
        
    Returns:
        A style description string
    """
    styles = {
        "Beginner": ["simple", "legato", "stepwise"],
        "Intermediate": ["jazzy", "bluesy", "march-like", "syncopated"],
        "Advanced": ["technical", "chromatic", "fast arpeggios", "wide intervals"],
    }
    return random.choice(styles.get(level, ["technical"]))


def get_technique_based_on_level(level: str) -> str:
    """
    Get a random technique appropriate for the given difficulty level.
    
    Args:
        level: Difficulty level (Beginner, Intermediate, Advanced)
        
    Returns:
        A technique description string
    """
    techniques = {
        "Beginner": ["with long tones", "with simple rhythms", "focusing on tone"],
        "Intermediate": ["with slurs", "with accents", "using triplets"],
        "Advanced": ["with double tonguing", "with extreme registers", "complex rhythms"],
    }
    return random.choice(techniques.get(level, ["with slurs"]))


def get_fallback_exercise(instrument: str, level: str, key: str,
                          time_sig: str, measures: int) -> str:
    """
    Generate a fallback exercise when LLM generation fails.
    
    Args:
        instrument: Target instrument
        level: Difficulty level
        key: Musical key
        time_sig: Time signature (e.g., "4/4")
        measures: Number of measures
        
    Returns:
        JSON string with fallback exercise
    """
    instrument_patterns = {
        "Trumpet": ["C4", "D4", "E4", "G4", "E4", "C4"],
        "Piano": ["C4", "E4", "G4", "C5", "G4", "E4"],
        "Violin": ["G4", "A4", "B4", "D5", "B4", "G4"],
        "Clarinet": ["E4", "F4", "G4", "Bb4", "G4", "E4"],
        "Flute": ["A4", "B4", "C5", "E5", "C5", "A4"],
    }
    pattern = instrument_patterns.get(instrument, instrument_patterns["Trumpet"])
    numerator, denominator = map(int, time_sig.split('/'))

    # Calculate units based on 8th notes
    units_per_measure = numerator * (8 // denominator)  # 8th notes per measure
    target_units = measures * units_per_measure
    notes, durs = [], []
    i = 0

    # Use quarter notes (2 units) as base duration
    while sum(durs) < target_units:
        notes.append(pattern[i % len(pattern)])
        # Use quarter notes (2 units) by default
        durs.append(2)
        i += 1

    # Adjust last duration to match total exactly
    total_units = sum(durs)
    if total_units > target_units:
        durs[-1] = durs[-1] - (total_units - target_units)
    elif total_units < target_units:
        durs[-1] = durs[-1] + (target_units - total_units)

    return json.dumps([[n, d] for n, d in zip(notes, durs)])


def query_mistral(prompt: str, instrument: str, level: str, key: str,
                  time_sig: str, measures: int, api_key: Optional[str] = None) -> str:
    """
    Query Mistral API to generate a music exercise.
    
    Args:
        prompt: Custom prompt or empty string for default
        instrument: Target instrument
        level: Difficulty level
        key: Musical key
        time_sig: Time signature (e.g., "4/4")
        measures: Number of measures
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        JSON string with generated exercise
        
    Raises:
        Exception: If API call fails and fallback is used
    """
    api_key = api_key or MISTRAL_API_KEY
    if not api_key:
        raise ValueError("No Mistral API key provided. Set MISTRAL_API_KEY environment variable.")
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    numerator, denominator = map(int, time_sig.split('/'))

    # Calculate total required 8th notes
    units_per_measure = numerator * (8 // denominator)
    required_total = measures * units_per_measure

    # Duration explanation in prompt
    duration_constraint = (
        f"Sum of all durations MUST BE EXACTLY {required_total} units (8th notes). "
        f"Each integer duration represents an 8th note (1=8th, 2=quarter, 4=half, 8=whole). "
        f"If it doesn't match, the exercise is invalid."
    )
    system_prompt = (
        f"You are an expert music teacher specializing in {instrument.lower()}. "
        "Create customized exercises using INTEGER durations representing 8th notes."
    )

    if prompt.strip():
        user_prompt = (
            f"{prompt} {duration_constraint} Output ONLY a JSON array of [note, duration] pairs.\n\n"
            "The response must follow this JSON schema:\n"
            "{ \"type\": \"array\", \"items\": { \"type\": \"array\", \"items\": [ "
            "{\"type\": \"string\", \"description\": \"Note name (e.g., C4, F#5)\"}, "
            "{\"type\": \"integer\", \"description\": \"Duration in 8th note units\"} ], "
            "\"minItems\": 2, \"maxItems\": 2 } }"
        )
    else:
        style = get_style_based_on_level(level)
        technique = get_technique_based_on_level(level)
        user_prompt = (
            f"Create a {style} {instrument.lower()} exercise in {key} with {time_sig} time signature "
            f"{technique} for a {level.lower()} player. {duration_constraint} "
            "Output ONLY a JSON array of [note, duration] pairs following these rules: "
            "Use standard note names (e.g., \"Bb4\", \"F#5\"). Monophonic only. "
            "Durations: 1=8th, 2=quarter, 4=half, 8=whole. "
            "Sum must be exactly as specified. ONLY output the JSON array. No prose.\n\n"
            "The response must follow this JSON schema:\n"
            "{ \"type\": \"array\", \"items\": { \"type\": \"array\", \"items\": [ "
            "{\"type\": \"string\", \"description\": \"Note name (e.g., C4, F#5)\"}, "
            "{\"type\": \"integer\", \"description\": \"Duration in 8th note units\"} ], "
            "\"minItems\": 2, \"maxItems\": 2 } }"
        )

    payload = {
        "model": "mistral-medium",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7 if level == "Advanced" else 0.5,
        "max_tokens": 1000,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return content.replace("```json", "").replace("```", "").strip()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Rate limit exceeded for Mistral API. Using fallback exercise.")
        else:
            print(f"Error querying Mistral API: {e}")
        return get_fallback_exercise(instrument, level, key, time_sig, measures)
    except requests.exceptions.RequestException as e:
        print(f"Network error querying Mistral API: {e}")
        return get_fallback_exercise(instrument, level, key, time_sig, measures)
    except (KeyError, IndexError) as e:
        print(f"Error parsing Mistral API response: {e}")
        return get_fallback_exercise(instrument, level, key, time_sig, measures)


def generate_exercise(instrument: str, level: str, key: str, time_signature: str,
                      measures: int, custom_prompt: str = "", api_key: Optional[str] = None) -> List[List[Any]]:
    """
    Generate a music exercise with proper error handling.
    
    Args:
        instrument: Target instrument
        level: Difficulty level
        key: Musical key
        time_signature: Time signature (e.g., "4/4")
        measures: Number of measures
        custom_prompt: Optional custom prompt
        api_key: Optional API key
        
    Returns:
        List of [note, duration] pairs
        
    Raises:
        ValueError: If parameters are invalid
    """
    try:
        # Query LLM for exercise
        output = query_mistral(custom_prompt, instrument, level, key, time_signature, measures, api_key)
        parsed = safe_parse_json(output)
        
        if not parsed:
            print("Primary parsing failed, using fallback")
            fallback_str = get_fallback_exercise(instrument, level, key, time_signature, measures)
            parsed = safe_parse_json(fallback_str)
            if not parsed:
                print("Fallback parsing failed, using ultimate fallback")
                # Ultimate fallback: simple scale based on selected key
                key_notes = {
                    "C Major": ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"],
                    "G Major": ["G3", "A3", "B3", "C4", "D4", "E4", "F#4", "G4"],
                    "D Major": ["D4", "E4", "F#4", "G4", "A4", "B4", "C#5", "D5"],
                    "F Major": ["F3", "G3", "A3", "Bb3", "C4", "D4", "E4", "F4"],
                    "Bb Major": ["Bb3", "C4", "D4", "Eb4", "F4", "G4", "A4", "Bb4"],
                    "A Minor": ["A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4"],
                    "E Minor": ["E3", "F#3", "G3", "A3", "B3", "C4", "D4", "E4"],
                }
                notes = key_notes.get(key, key_notes["C Major"])
                numerator, denominator = map(int, time_signature.split('/'))
                units_per_measure = numerator * (8 // denominator)
                target_units = measures * units_per_measure
                note_duration = max(1, target_units / len(notes))
                parsed = [[n, note_duration] for n in notes]
                # Adjust last note to match total duration
                total = sum(d for _, d in parsed)
                if total < target_units:
                    parsed[-1][1] += target_units - total
                elif total > target_units:
                    parsed[-1][1] -= total - target_units

        # Clean note strings to remove ornamentation
        from .theory import clean_note_string
        for i, (note, dur) in enumerate(parsed):
            parsed[i][0] = clean_note_string(note)

        # Calculate total required 8th notes
        numerator, denominator = map(int, time_signature.split('/'))
        units_per_measure = numerator * (8 // denominator)
        total_units = measures * units_per_measure

        # Strict scaling
        parsed_scaled = scale_json_durations(parsed, total_units)
        
        return parsed_scaled
    except Exception as e:
        print(f"Error generating exercise: {e}")
        raise