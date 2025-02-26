"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os
import re
import modules.util.appstate
from datetime import datetime
from diffusers import Lumina2Text2ImgPipeline
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Lumina"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()
def get_dimensions(resolution):
    width, height = map(int, resolution.split('x'))
    return width, height
def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            if ((wp * patch_size)//32) % 2 == 0 and  ((hp * patch_size)//32) % 2 == 0:
                crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

def get_pipeline(inference_type, memory_optimization, vaeslicing, vaetiling):
    print("----Lumina2 mode: ", inference_type, memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
    type(modules.util.appstate.global_pipe).__name__ == "Lumina2Text2ImgPipeline" and
    modules.util.appstate.global_inference_type == inference_type and 
    modules.util.appstate.global_memory_mode == memory_optimization):
            print(">>>>Reusing Lumina2 Default pipe<<<<")
            return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    bfl_repo = "Alpha-VLLM/Lumina-Image-2.0"
    dtype = torch.bfloat16
    
    modules.util.appstate.global_pipe = Lumina2Text2ImgPipeline.from_pretrained(
        bfl_repo,
        torch_dtype=dtype,
    )
    if memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()
    elif memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
    if vaeslicing:
        modules.util.appstate.global_pipe.vae.enable_slicing()
    else:
        modules.util.appstate.global_pipe.vae.disable_slicing()
    if vaetiling:
        modules.util.appstate.global_pipe.vae.enable_tiling()
    else:
        modules.util.appstate.global_pipe.vae.disable_tiling()
        
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = inference_type
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, negative_prompt, resolution, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        width, height = get_dimensions(resolution)
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline("lumina2", memory_optimization, vaeslicing, vaetiling)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs

        # Prepare inference parameters
        inference_params = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }
        start_time = datetime.now()
        # Generate images
        image = pipe(**inference_params).images[0]
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "lumina2.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        metadata = {
            "model": "Lumina-Image-2.0",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "resolution": resolution,
            "memory_optimization": memory_optimization,
            "vae_slicing": vaeslicing,
            "vae_tiling": vaetiling,
            "timestamp": timestamp,
            "generation_time": generation_time
        }
        # Save the image
        image.save(output_path)
        modules.util.utilities.save_metadata_to_file(output_path, metadata)
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "Lumina 2"))
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False
def create_lumina2_tab():
    gr.HTML("<style>.small-button { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; align-self: end; line-height: 1em; border-radius: 0.5em; }</style>", visible=False)
    initial_state = state_manager.get_state("lumina2") or {}
    with gr.Row():
        with gr.Column():
                lumina2_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    value=initial_state.get("memory_optimization", "Low VRAM"),
                    interactive=True
                )
        with gr.Column():
                lumina2_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", False), interactive=True)
                lumina2_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", False), interactive=True)
    with gr.Row():
        with gr.Column():
            lumina2_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=6,
                interactive=True
            )
            lumina2_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                lumina2_resolution_dropdown = gr.Dropdown(
                    choices=[f"{w}x{h}" for w, h in generate_crop_size_list((1024 // 64) ** 2, 64)],
                    value=initial_state.get("resolution", "1024x1024"),
                    label="Resolution"
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("♻️ Randomize seed")
            with gr.Row():
                lumina2_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 4.0),
                    step=0.1,
                    interactive=True
                )
                lumina2_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 40),
                    interactive=True
                )
            with gr.Row():
                save_state_button = gr.Button("💾 Save State")
    with gr.Row():
        generate_button = gr.Button("🎨 Generate image")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    def save_current_state(memory_optimization, vaeslicing, vaetiling, resolution, guidance_scale, inference_steps):
        state_dict = {
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "resolution": resolution,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        state_manager.save_state("lumina2", state_dict)
        return (inference_type, gguf, quantization, memory_optimization, vaeslicing, vaetiling, resolution, guidance_scale, inference_steps)
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            lumina2_memory_optimization, 
            lumina2_vaeslicing, 
            lumina2_vaetiling, 
            lumina2_resolution_dropdown, 
            lumina2_guidance_scale_slider, 
            lumina2_num_inference_steps_input
        ],
        outputs=[
            lumina2_memory_optimization,
            lumina2_vaeslicing,
            lumina2_vaetiling,
            lumina2_resolution_dropdown,
            lumina2_guidance_scale_slider,
            lumina2_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, lumina2_prompt_input, lumina2_negative_prompt_input, 
            lumina2_resolution_dropdown, lumina2_guidance_scale_slider, 
            lumina2_num_inference_steps_input, lumina2_memory_optimization, 
            lumina2_vaeslicing, lumina2_vaetiling
        ],
        outputs=[output_gallery]
    )