"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os
import modules.util.config
from datetime import datetime
from diffusers.utils import export_to_video
from diffusers import LTXImageToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer
from modules.util.utilities import clear_previous_model_memory

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/i2v/ltxvideo091"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization):
    print("----ltxvideo image2video 091 mode: ", memory_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.config.global_pipe is not None and 
        type(modules.util.config.global_pipe).__name__ == "LTXImageToVideoPipeline" and
        modules.util.config.global_memory_mode == memory_optimization):
        print(">>>>Reusing ltxvideo091 image2video pipe<<<<")
        return modules.util.config.global_pipe
    else:
        clear_previous_model_memory()
    
    repo_id = "a-r-r-o-w/LTX-Video-0.9.1-diffusers"
    single_file_url = "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors"
    text_encoder = T5EncoderModel.from_pretrained(
      repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    tokenizer = T5Tokenizer.from_pretrained(
      repo_id, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    modules.util.config.global_pipe = LTXImageToVideoPipeline.from_single_file(
        single_file_url, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        torch_dtype=torch.bfloat16
    )

    if memory_optimization == "Low VRAM":
        modules.util.config.global_pipe.enable_model_cpu_offload()

    modules.util.config.global_memory_mode = memory_optimization
    
    return modules.util.config.global_pipe

def generate_video(
    seed, input_image, prompt, negative_prompt, width, height, fps,
    num_inference_steps, num_frames, memory_optimization,
):
    if modules.util.config.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.config.global_inference_in_progress = True
    # Get pipeline (either cached or newly loaded)
    pipe = get_pipeline(memory_optimization)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    progress_bar = gr.Progress(track_tqdm=True)

    def callback_on_step_end(pipe, i, t, callback_kwargs):
        progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
        return callback_kwargs
    # Prepare inference parameters
    inference_params = {
        "image": input_image,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "generator": generator,
        "callback_on_step_end": callback_on_step_end,
    }

    # Generate video
    video = pipe(**inference_params).frames[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_filename = "ltxvideo091.mp4"
    
    gallery_items = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{base_filename}"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Save the video
    export_to_video(video, output_path, fps=fps)
    print(f"Video generated: {output_path}")
    modules.util.config.global_inference_in_progress = False
    
    return output_path

def create_ltximage2video091_tab():
    with gr.Row():
        ltxvideo091_memory_optimization = gr.Radio(
            choices=["No optimization", "Low VRAM"],
            label="Memory Optimization",
            value="Low VRAM",
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            ltxvideo091_input_image = gr.Image(label="Input Image", type="pil")
        with gr.Column():
            ltxvideo091_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            ltxvideo091_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                ltxvideo091_width_input = gr.Number(
                    label="Width", 
                    value=704, 
                    interactive=True
                )
                ltxvideo091_height_input = gr.Number(
                    label="Height", 
                    value=480, 
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                ltxvideo091_fps_input = gr.Number(
                    label="FPS", 
                    value=24,
                    interactive=True
                )
                ltxvideo091_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=50,
                    interactive=True
                )
                ltxvideo091_num_frames_input = gr.Number(
                    label="Number of frames", 
                    value=61,
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate video")
    output_video = gr.Video(label="Generated Video", show_label=True)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])

    generate_button.click(
        fn=generate_video,
        inputs=[
            seed_input, ltxvideo091_input_image, ltxvideo091_prompt_input, ltxvideo091_negative_prompt_input, ltxvideo091_width_input, 
            ltxvideo091_height_input, ltxvideo091_fps_input, ltxvideo091_num_inference_steps_input, 
            ltxvideo091_num_frames_input, ltxvideo091_memory_optimization,
        ],
        outputs=[output_video]
    )