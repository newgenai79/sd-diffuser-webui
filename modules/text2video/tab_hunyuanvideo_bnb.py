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
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers import BitsAndBytesConfig
from modules.util.utilities import clear_previous_model_memory

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2v/hunyuanvideo"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, quantization, vaeslicing, vaetiling):
    print("----hunyuanvideo mode: ", memory_optimization, quantization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.config.global_pipe is not None and 
        type(modules.util.config.global_pipe).__name__ == "HunyuanVideoPipeline" and
        modules.util.config.global_quantization == quantization and
        modules.util.config.global_memory_mode == memory_optimization):
        print(">>>>Reusing hunyuanvideo pipe<<<<")
        return modules.util.config.global_pipe
    else:
        clear_previous_model_memory()
    repo_id = "hunyuanvideo-community/HunyuanVideo"
    if (quantization == "int4"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif(quantization == "int8"):
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    transformer_quant = HunyuanVideoTransformer3DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    modules.util.config.global_pipe = HunyuanVideoPipeline.from_pretrained(
        repo_id, 
        transformer=transformer_quant,
        torch_dtype=torch.float16
    )
    if memory_optimization == "Low VRAM":
        modules.util.config.global_pipe.enable_model_cpu_offload()

    if vaeslicing:
        modules.util.config.global_pipe.vae.enable_slicing()
    else:
        modules.util.config.global_pipe.vae.disable_slicing()
    if vaetiling:
        modules.util.config.global_pipe.vae.enable_tiling()
    else:
        modules.util.config.global_pipe.vae.disable_tiling()

    modules.util.config.global_memory_mode = memory_optimization
    modules.util.config.global_quantization = quantization
    return modules.util.config.global_pipe

def generate_video(
    seed, prompt, width, height, fps,
    num_inference_steps, num_frames, memory_optimization, vaeslicing, vaetiling, quantization
):
    if modules.util.config.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.config.global_inference_in_progress = True
    # Get pipeline (either cached or newly loaded)
    pipe = get_pipeline(memory_optimization, quantization, vaeslicing, vaetiling)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    progress_bar = gr.Progress(track_tqdm=True)

    def callback_on_step_end(pipe, i, t, callback_kwargs):
        progress_bar(i / num_inference_steps, desc=f"Generating video (Step {i}/{num_inference_steps})")
        return callback_kwargs
    # Prepare inference parameters
    inference_params = {
        "prompt": prompt,
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
    
    base_filename = "hunyuanvideo_bnb.mp4"
    
    gallery_items = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{base_filename}"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Save the video
    export_to_video(video, output_path, fps=fps)
    print(f"Video generated: {output_path}")
    modules.util.config.global_inference_in_progress = False
    
    return output_path

def create_hunyuanvideo_bnb_tab():
    with gr.Row():
        with gr.Column():
            hunyuanvideo_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value="Low VRAM",
                interactive=True
            )
        with gr.Column():
            hunyuanvideo_vaeslicing = gr.Checkbox(label="VAE slicing", value=True, interactive=True)
            hunyuanvideo_vaetiling = gr.Checkbox(label="VAE Tiling", value=True, interactive=True)
        with gr.Column():
            hunyuanvideo_quantization = gr.Radio(
                choices=["int4", "int8"],
                label="BitsnBytes quantization",
                value="int4",
                interactive=True
            )
    with gr.Row():
        with gr.Column():
            hunyuanvideo_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                hunyuanvideo_width_input = gr.Number(
                    label="Width", 
                    value=512, 
                    interactive=True
                )
                hunyuanvideo_height_input = gr.Number(
                    label="Height", 
                    value=320, 
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                hunyuanvideo_fps_input = gr.Number(
                    label="FPS", 
                    value=24,
                    interactive=True
                )
                hunyuanvideo_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=50,
                    interactive=True
                )
                hunyuanvideo_num_frames_input = gr.Number(
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
            seed_input, hunyuanvideo_prompt_input, hunyuanvideo_width_input, 
            hunyuanvideo_height_input, hunyuanvideo_fps_input, hunyuanvideo_num_inference_steps_input, 
            hunyuanvideo_num_frames_input, hunyuanvideo_memory_optimization, hunyuanvideo_vaeslicing,
            hunyuanvideo_vaetiling, hunyuanvideo_quantization
        ],
        outputs=[output_video]
    )