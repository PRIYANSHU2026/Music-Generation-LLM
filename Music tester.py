import gradio as gr
import pickle
import torch
import torch.nn as nn
from transformers import T5Tokenizer
from model.transformer_model import Transformer
from huggingface_hub import hf_hub_download

# Initialize model and tokenizers (same as your original code)
repo_id = "amaai-lab/text2midi"
model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

with open(tokenizer_path, "rb") as f:
    r_tokenizer = pickle.load(f)

vocab_size = len(r_tokenizer)
model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")


def generate_midi(prompt, key, scale, tempo, instruments):
    # Construct the full prompt with musical parameters
    full_prompt = f"{prompt}. Instrumentation: {instruments}, trumpet. Key: {key} {scale}. Tempo: {tempo} BPM."

    print('Generating for prompt: ' + full_prompt)

    # Tokenize and generate
    inputs = tokenizer(full_prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
    input_ids = input_ids.to(device)
    attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)
    attention_mask = attention_mask.to(device)

    output = model.generate(input_ids, attention_mask, max_len=2000, temperature=1.0)
    output_list = output[0].tolist()
    generated_midi = r_tokenizer.decode(output_list)

    # Save and return the MIDI file
    output_path = "output.mid"
    generated_midi.dump_midi(output_path)
    return output_path


# Define Gradio interface
with gr.Blocks(title="Text-to-MIDI Generator") as demo:
    gr.Markdown("# 🎼 Text-to-MIDI Generator (Trumpet Focus)")
    gr.Markdown("Generate MIDI files with trumpet in your selected key and scale.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Musical Description",
                placeholder="Describe the music you want to generate...",
                lines=3,
                value="A melodic jazz piece with trumpet as the lead instrument"
            )
            instruments = gr.Textbox(
                label="Additional Instruments",
                value="piano, bass, drums",
                placeholder="List other instruments to include (comma separated)"
            )

            with gr.Row():
                key = gr.Dropdown(
                    label="Key",
                    choices=["A", "B", "C", "D", "E", "F"],
                    value="C"
                )
                scale = gr.Dropdown(
                    label="Scale",
                    choices=["major", "minor"],
                    value="major"
                )

            tempo = gr.Slider(
                label="Tempo (BPM)",
                minimum=40,
                maximum=200,
                value=120,
                step=5
            )

            generate_btn = gr.Button("Generate MIDI", variant="primary")

        with gr.Column():
            output_midi = gr.Audio(label="Generated MIDI", type="filepath")

    generate_btn.click(
        fn=generate_midi,
        inputs=[prompt, key, scale, tempo, instruments],
        outputs=output_midi
    )

    gr.Examples(
        examples=[
            ["A bright jazz trumpet solo with walking bass", "B", "major", 120, "piano, bass, drums"],
            ["A melancholic trumpet ballad", "F", "minor", 70, "piano, string quartet"],
            ["A lively mariachi trumpet line", "A", "major", 140, "guitar, violin, accordion"]
        ],
        inputs=[prompt, key, scale, tempo, instruments],
        outputs=output_midi,
        fn=generate_midi,
        cache_examples=True
    )

if __name__ == "__main__":
    demo.launch()