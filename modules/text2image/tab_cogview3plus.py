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
from diffusers import CogView3PlusPipeline
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/cogView3Plus"
RESOLUTIONS_cogView3Plus = [
    "512x512",
    "720x480",
    "1024x1024",
    "1280x720",
    "2048x2048"
]

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, vaeslicing, vaetiling):
    print("----cogView3Plus mode: ",memory_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "CogView3PlusPipeline" and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing cogView3Plus pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
    
    modules.util.appstate.global_pipe = CogView3PlusPipeline.from_pretrained(
        "THUDM/CogView3-Plus-3B",
        torch_dtype=torch.bfloat16,
    )
    modules.util.appstate.global_pipe.text_encoder = modules.util.appstate.global_pipe.text_encoder.to("cpu")
    modules.util.appstate.global_pipe.vae = modules.util.appstate.global_pipe.vae.to("cuda")
    modules.util.appstate.global_pipe.transformer = modules.util.appstate.global_pipe.transformer.to("cuda")

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
        pipe = get_pipeline(memory_optimization, vaeslicing, vaetiling)
        generator = torch.Generator(device="cpu").manual_seed(seed)
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
        
        base_filename = "cogView3Plus.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        image.save(output_path)
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "cogView3Plus"))
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_cogView3Plus_tab():
    initial_state = state_manager.get_state("cogview3plus") or {}
    with gr.Row():
        with gr.Column():
            cogView3Plus_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            cogView3Plus_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            cogView3Plus_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            cogView3Plus_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            cogView3Plus_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                cogView3Plus_resolution_dropdown = gr.Dropdown(
                    choices=RESOLUTIONS_cogView3Plus,
                    value=initial_state.get("resolution", "512x512"),
                    label="Resolution"
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
                save_state_button = gr.Button("Save State")
            with gr.Row():
                cogView3Plus_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 7.0),
                    step=0.1,
                    interactive=True
                )
                cogView3Plus_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
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

    def save_current_state(memory_optimization, vaeslicing, vaetiling, resolution, guidance_scale, inference_steps):
        state_dict = {
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "resolution": resolution,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("cogview3plus") or {}
        state_manager.save_state("cogview3plus", state_dict)
        return memory_optimization, vaeslicing, vaetiling, resolution, guidance_scale, inference_steps

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            cogView3Plus_memory_optimization, 
            cogView3Plus_vaeslicing, 
            cogView3Plus_vaetiling, 
            cogView3Plus_resolution_dropdown, 
            cogView3Plus_guidance_scale_slider, 
            cogView3Plus_num_inference_steps_input
        ],
        outputs=[
            cogView3Plus_memory_optimization, 
            cogView3Plus_vaeslicing, 
            cogView3Plus_vaetiling, 
            cogView3Plus_resolution_dropdown, 
            cogView3Plus_guidance_scale_slider, 
            cogView3Plus_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, cogView3Plus_prompt_input, cogView3Plus_negative_prompt_input, 
            cogView3Plus_resolution_dropdown, cogView3Plus_guidance_scale_slider, 
            cogView3Plus_num_inference_steps_input, cogView3Plus_memory_optimization, 
            cogView3Plus_vaeslicing, cogView3Plus_vaetiling,
        ],
        outputs=[output_gallery]
    )