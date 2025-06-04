import gradio as gr
import pickle
import torch
from transformers import T5Tokenizer
from model.transformer_model import Transformer
from huggingface_hub import hf_hub_download
import time

# 🎯 Step 1: Load model + tokenizer just once
repo_id = "amaai-lab/text2midi"
model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"🔥 Using device: {device}")

# 🎵 Load REMI tokenizer
with open(tokenizer_path, "rb") as f:
    r_tokenizer = pickle.load(f)

vocab_size = len(r_tokenizer)

# 🎼 Load Transformer model
model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 🔡 Load text tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")


# 🎶 Function to generate MIDI exercises
def generate_trumpet_exercise(difficulty, key, scale, exercise_type, tempo):
    # Pre-defined exercise templates for fast generation
    exercise_templates = {
        "intervals": {
            "beginner": "Simple ascending and descending intervals within one octave",
            "intermediate": "Interval patterns with varied articulation",
            "advanced": "Complex interval leaps across registers"
        },
        "scales": {
            "beginner": "Major scale in one octave",
            "intermediate": "Scale sequences with varied rhythm",
            "advanced": "Full range scales in thirds and arpeggiated patterns"
        },
        "arpeggios": {
            "beginner": "Basic tonic arpeggio",
            "intermediate": "Arpeggio sequences in triplets",
            "advanced": "Diminished and augmented arpeggios in all keys"
        },
        "tonguing": {
            "beginner": "Single tongue quarter notes",
            "intermediate": "Double tongue patterns",
            "advanced": "Triple tongue combinations at fast tempo"
        },
        "flexibility": {
            "beginner": "Simple lip slurs between partials",
            "intermediate": "Lip flexibility exercises spanning large intervals",
            "advanced": "Extreme range flexibility exercises"
        }
    }

    # Get template based on parameters
    description = exercise_templates[exercise_type][difficulty]

    # Construct optimized prompt
    prompt = f"Generate trumpet exercise: {description}. Key: {key} {scale}. Tempo: {tempo} BPM. Difficulty: {difficulty}."
    print(f"📝 Generating exercise: {prompt}")

    # Start timer
    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # 🚀 Optimized generation parameters for exercises
    output = model.generate(
        input_ids,
        attention_mask,
        max_len=300,  # Shorter sequences for exercises
        temperature=0.9,  # More focused output
        top_k=30,  # Limit choices for faster generation
        num_beams=2  # Faster than default beam search
    )

    # Decode and save MIDI
    output_list = output[0].tolist()
    generated_midi = r_tokenizer.decode(output_list)
    output_path = f"trumpet_exercise_{difficulty}_{key}_{scale}.mid"
    generated_midi.dump_midi(output_path)

    # Calculate generation time
    gen_time = time.time() - start_time
    print(f"⏱️ Generated in {gen_time:.2f} seconds")

    return output_path


# 🧑‍🏫 Gradio UI for Exercise Generator
with gr.Blocks(title="Trumpet Exercise Generator") as demo:
    gr.Markdown("# 🎺 AI Trumpet Exercise Generator")
    gr.Markdown("Generate customized trumpet exercises instantly")

    with gr.Row():
        with gr.Column():
            difficulty = gr.Dropdown(
                label="Difficulty Level",
                choices=["beginner", "intermediate", "advanced"],
                value="intermediate"
            )

            key = gr.Dropdown(
                label="Key",
                choices=["C", "F", "Bb", "G", "D", "A", "E"],
                value="Bb"
            )

            scale = gr.Dropdown(
                label="Scale",
                choices=["major", "minor"],
                value="major"
            )

            exercise_type = gr.Dropdown(
                label="Exercise Type",
                choices=["intervals", "scales", "arpeggios", "tonguing", "flexibility"],
                value="scales"
            )

            tempo = gr.Slider(
                label="Tempo (BPM)",
                minimum=60,
                maximum=180,
                value=120,
                step=5
            )

            generate_btn = gr.Button("🎵 Generate Exercise", variant="primary")

        with gr.Column():
            output_midi = gr.Audio(label="🎧 Generated Exercise", type="filepath")
            gr.Markdown("### 🎼 Exercise Tips")
            gr.Markdown("- Use a metronome during practice")
            gr.Markdown("- Focus on clean articulation")
            gr.Markdown("- Maintain consistent airflow")

    generate_btn.click(
        fn=generate_trumpet_exercise,
        inputs=[difficulty, key, scale, exercise_type, tempo],
        outputs=output_midi
    )

    # 📚 Example exercises
    gr.Examples(
        examples=[
            ["beginner", "C", "major", "scales", 100],
            ["intermediate", "Bb", "major", "arpeggios", 120],
            ["advanced", "G", "minor", "flexibility", 80]
        ],
        inputs=[difficulty, key, scale, exercise_type, tempo],
        outputs=output_midi,
        fn=generate_trumpet_exercise,
        cache_examples=True
    )

# 🚀 Run
if __name__ == "__main__":
    demo.launch()