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
from diffusers import AuraFlowPipeline, AuraFlowTransformer2DModel
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/auraflow"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, vaeslicing, vaetiling, inference_type):
    print("----auraflow mode: ", memory_optimization, vaeslicing, vaetiling, inference_type)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "AuraFlowPipeline" and
        modules.util.appstate.global_inference_type == inference_type and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing auraflow pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    modules.util.appstate.global_pipe = AuraFlowPipeline.from_pretrained(
        "fal/AuraFlow-v0.3",
        torch_dtype=torch.bfloat16,
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

    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = inference_type
    return modules.util.appstate.global_pipe
def get_gguf(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        pipe = get_pipeline(memory_optimization, vaeslicing, vaetiling, "auraflow")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        """
        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs
        """
        # Prepare inference parameters
        inference_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        start_time = datetime.now()
        # Generate images
        image = pipe(**inference_params).images[0]
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "auraflow.png"

        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        metadata = {
            "model": "AuraFlow",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
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
        gallery_items.append((output_path, "AuraFlow"))
    
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_auraflow_tab():
    initial_state = state_manager.get_state("auraflow") or {}
    with gr.Row():
        with gr.Column():
            auraflow_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            auraflow_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            auraflow_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            auraflow_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            auraflow_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                auraflow_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                auraflow_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                auraflow_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 3.5),
                    step=0.1,
                    interactive=True
                )
                auraflow_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
                    interactive=True
                )
            with gr.Row():
                save_state_button = gr.Button("Save State")
    with gr.Row():
        generate_button = gr.Button("Generate image")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    def save_current_state(memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps):
        state_dict = {
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": int(width),
            "height": int(height),
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("auraflow") or {}
        state_manager.save_state("auraflow", state_dict)
        return memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            auraflow_memory_optimization, 
            auraflow_vaeslicing, 
            auraflow_vaetiling, 
            auraflow_width_input, 
            auraflow_height_input, 
            auraflow_guidance_scale_slider, 
            auraflow_num_inference_steps_input
        ],
        outputs=[
            auraflow_memory_optimization, 
            auraflow_vaeslicing, 
            auraflow_vaetiling, 
            auraflow_width_input, 
            auraflow_height_input, 
            auraflow_guidance_scale_slider, 
            auraflow_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, auraflow_prompt_input, auraflow_negative_prompt_input, auraflow_width_input, 
            auraflow_height_input, auraflow_guidance_scale_slider, auraflow_num_inference_steps_input, 
            auraflow_memory_optimization, auraflow_vaeslicing, auraflow_vaetiling
        ],
        outputs=[output_gallery]
    )
