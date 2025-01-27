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
from diffusers import CogVideoXPipeline
from modules.util.utilities import clear_previous_model_memory

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2v/cogvideox155b"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, vaeslicing, vaetiling):
    print("----cogvideox155b mode: ", memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.config.global_pipe is not None and 
        type(modules.util.config.global_pipe).__name__ == "CogVideoXPipeline" and
        modules.util.config.global_memory_mode == memory_optimization):
        print(">>>>Reusing cogvideox155b pipe<<<<")
        return modules.util.config.global_pipe
    else:
        clear_previous_model_memory()
    
    modules.util.config.global_pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX1.5-5B",
        torch_dtype=torch.bfloat16
    )

    if memory_optimization == "Low VRAM":
        modules.util.config.global_pipe.enable_model_cpu_offload()
    elif memory_optimization == "Extremely Low VRAM":
        modules.util.config.global_pipe.enable_sequential_cpu_offload()

    if vaeslicing:
        modules.util.config.global_pipe.vae.enable_slicing()
    else:
        modules.util.config.global_pipe.vae.disable_slicing()
    if vaetiling:
        modules.util.config.global_pipe.vae.enable_tiling()
    else:
        modules.util.config.global_pipe.vae.disable_tiling()

    modules.util.config.global_memory_mode = memory_optimization
    
    return modules.util.config.global_pipe

def generate_video(
    guidance_scale, seed, prompt, negative_prompt, width, height, fps,
    num_inference_steps, num_frames, use_dynamic_cfg, memory_optimization, vaeslicing, vaetiling
):
    if modules.util.config.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.config.global_inference_in_progress = True
    # Get pipeline (either cached or newly loaded)
    pipe = get_pipeline(memory_optimization, vaeslicing, vaetiling)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    progress_bar = gr.Progress(track_tqdm=True)

    def callback_on_step_end(pipe, i, t, callback_kwargs):
        progress_bar(i / num_inference_steps, desc=f"Generating video (Step {i}/{num_inference_steps})")
        return callback_kwargs
    # Prepare inference parameters
    inference_params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "num_frames": num_frames,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "use_dynamic_cfg": use_dynamic_cfg,
        "callback_on_step_end": callback_on_step_end,
    }

    # Generate video
    video = pipe(**inference_params).frames[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    base_filename = "cogvideox155b.mp4"
    
    gallery_items = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{base_filename}"
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Save the video
    export_to_video(video, output_path, fps=fps)
    print(f"Video generated: {output_path}")
    modules.util.config.global_inference_in_progress = False
    
    return output_path

def create_cogvideox155b_t2v_tab():
    with gr.Row():
        with gr.Column():
            cogvideox155b_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                label="Memory Optimization",
                value="Extremely Low VRAM",
                interactive=True
            )
        with gr.Column():
            cogvideox155b_vaeslicing = gr.Checkbox(label="VAE slicing", value=True, interactive=True)
            cogvideox155b_vaetiling = gr.Checkbox(label="VAE Tiling", value=True, interactive=True)
    with gr.Row():
        with gr.Column():
            cogvideox155b_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=5,
                interactive=True
            )
            cogvideox155b_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                cogvideox155b_width_input = gr.Number(
                    label="Width", 
                    value=512, 
                    interactive=True
                )
                cogvideox155b_height_input = gr.Number(
                    label="Height", 
                    value=320, 
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                cogvideox155b_fps_input = gr.Number(
                    label="FPS", 
                    value=15,
                    interactive=True
                )
                cogvideox155b_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=50,
                    interactive=True
                )
                cogvideox155b_num_frames_input = gr.Number(
                    label="Number of frames", 
                    value=61,
                    interactive=True
                )
                cogvideox155b_use_dynamic_cfg = gr.Checkbox(label="Use Dynamic CFG", value=True)
                cogvideox155b_guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, step=0.1, value=6)
    with gr.Row():
        generate_button = gr.Button("Generate video")
    output_video = gr.Video(label="Generated Video", show_label=True)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])

    generate_button.click(
        fn=generate_video,
        inputs=[
            cogvideox155b_guidance_scale,
            seed_input, cogvideox155b_prompt_input, cogvideox155b_negative_prompt_input, 
            cogvideox155b_width_input, cogvideox155b_height_input, cogvideox155b_fps_input, 
            cogvideox155b_num_inference_steps_input, cogvideox155b_num_frames_input, 
            cogvideox155b_use_dynamic_cfg, cogvideox155b_memory_optimization, 
            cogvideox155b_vaeslicing, cogvideox155b_vaetiling
        ],
        outputs=[output_video]
    )