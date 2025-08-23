#!/usr/bin/env python

"""
Music Visualization
=================
Functions for creating visual representations of music exercises.
"""

import os
import uuid
import json
from typing import Optional, List, Any

from lib.music_generation.theory import note_name_to_midi, midi_to_note_name


def create_visualization(json_data: str, time_sig: str) -> Optional[str]:
    """
    Create a piano roll visualization of the exercise.
    
    Args:
        json_data: JSON string containing [note, duration] pairs
        time_sig: Time signature (e.g., "4/4")
        
    Returns:
        Path to the generated image file or None if visualization fails
    """
    try:
        if not json_data or "Error" in json_data:
            return None

        parsed = json.loads(json_data)
        if not isinstance(parsed, list) or len(parsed) == 0:
            return None

        # Extract notes and durations
        notes = []
        durations = []
        for note, dur in parsed:
            try:
                midi_note = note_name_to_midi(note)
                notes.append(midi_note)
                durations.append(dur)
            except ValueError:
                notes.append(60)  # Default to middle C if parsing fails
                durations.append(dur)

        # Create piano roll visualization
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate time positions
        time_positions = [0]
        for dur in durations[:-1]:
            time_positions.append(time_positions[-1] + dur)

        # Plot notes as rectangles
        for i, (note, dur, pos) in enumerate(zip(notes, durations, time_positions)):
            rect = plt.Rectangle((pos, note - 0.4), dur, 0.8, color='blue', alpha=0.7)
            ax.add_patch(rect)
            # Add note name
            ax.text(pos + dur / 2, note + 0.5, midi_to_note_name(note),
                    ha='center', va='bottom', fontsize=8)

        # Add measure lines
        numerator, denominator = map(int, time_sig.split('/'))
        units_per_measure = numerator * (8 // denominator)
        max_time = time_positions[-1] + durations[-1]
        for measure in range(1, int(max_time / units_per_measure) + 1):
            measure_pos = measure * units_per_measure
            if measure_pos <= max_time:
                ax.axvline(x=measure_pos, color='gray', linestyle='--', alpha=0.5)

        # Set axis limits and labels
        if notes:
            ax.set_ylim(min(notes) - 5, max(notes) + 5)
        else:
            ax.set_ylim(55, 75)
        ax.set_xlim(0, max_time)
        ax.set_ylabel('MIDI Note')
        ax.set_xlabel('Time (8th note units)')
        ax.set_title('Exercise Visualization')

        # Add piano keyboard on y-axis
        ax.set_yticks([60, 62, 64, 65, 67, 69, 71, 72])  # C4 to C5
        ax.set_yticklabels(['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'])
        ax.grid(True, axis='y', alpha=0.3)

        # Create static directory if it doesn't exist
        os.makedirs('static', exist_ok=True)
        
        # Save figure to temporary file
        temp_img_path = os.path.join('static', f'visualization_{uuid.uuid4().hex}.png')
        plt.tight_layout()
        plt.savefig(temp_img_path)
        plt.close()

        return temp_img_path
    except ImportError as e:
        print(f"Visualization requires matplotlib: {e}")
        return None
    except ValueError as e:
        print(f"Error in visualization data: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating visualization: {e}")
        return None