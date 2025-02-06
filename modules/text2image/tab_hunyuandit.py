"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os
import modules.util.appstate
from datetime import datetime
from diffusers import HunyuanDiTPipeline
from modules.util.utilities import clear_previous_model_memory

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/hunyuandit"

RESOLUTIONS_hunyuandit = [
    "1024x1024",
    "1280x1280",
    "1024x768",
    "1152x864",
    "1280x960",
    "2048x2048",
    "768x1024",
    "864x1152",
    "960x1280",
    "1280x768",
    "768x1280"
]

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, vaeslicing, vaetiling):
    print("----hunyuandit mode: ", memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "HunyuanDiTPipeline" and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing hunyuandit pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    modules.util.appstate.global_pipe = HunyuanDiTPipeline.from_pretrained(
        "Tencent-Hunyuan/HunyuanDiT-Diffusers",
        torch_dtype=torch.float16,
    )
    modules.util.appstate.global_pipe.to("cuda")

    if memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()

    if vaeslicing:
        modules.util.appstate.global_pipe.vae.enable_slicing()
    else:
        modules.util.appstate.global_pipe.vae.disable_slicing()
    if vaetiling:
        modules.util.appstate.global_pipe.vae.enable_tiling()
    else:
        modules.util.appstate.global_pipe.vae.disable_tiling()

    modules.util.appstate.global_memory_mode = memory_optimization
    return modules.util.appstate.global_pipe

def get_dimensions(resolution):
    width, height = map(int, resolution.split('x'))
    return width, height
def generate_images(
    seed, prompt, negative_prompt, resolution, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling, 
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, vaeslicing, vaetiling,)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        width, height = get_dimensions(resolution)

        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs

        # Prepare inference parameters
        inference_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }

        # Generate images
        image = pipe(**inference_params).images[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "hunyuandit.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        image.save(output_path)
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "hunyuandit"))
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_hunyuandit_tab():
    with gr.Row():
        with gr.Column():
            hunyuandit_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value="Low VRAM",
                interactive=True
            )
        with gr.Column():
            hunyuandit_vaeslicing = gr.Checkbox(label="VAE slicing", value=True, interactive=True)
            hunyuandit_vaetiling = gr.Checkbox(label="VAE Tiling", value=True, interactive=True)
    with gr.Row():
        with gr.Column():
            hunyuandit_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            hunyuandit_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                hunyuandit_resolution_dropdown = gr.Dropdown(
                    choices=RESOLUTIONS_hunyuandit,
                    value="1024x1024",
                    label="Resolution"
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                hunyuandit_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=5.0, 
                    step=0.1,
                    interactive=True
                )
                hunyuandit_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=50,
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate image")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, hunyuandit_prompt_input, hunyuandit_negative_prompt_input, 
            hunyuandit_resolution_dropdown, hunyuandit_guidance_scale_slider, 
            hunyuandit_num_inference_steps_input, hunyuandit_memory_optimization, 
            hunyuandit_vaeslicing, hunyuandit_vaetiling,
        ],
        outputs=[output_gallery]
    )