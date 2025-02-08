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
from diffusers import Lumina2Text2ImgPipeline
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager
from diffusers import BitsAndBytesConfig

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Lumina"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(quantization, memory_optimization, vaeslicing, vaetiling):
    print("----Lumina mode: ", quantization, memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "Lumina2Text2ImgPipeline" and
        modules.util.appstate.global_quantization == quantization and 
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Lumina2 pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    bfl_repo = "Alpha-VLLM/Lumina-Image-2.0"
    dtype = torch.bfloat16
    if (quantization == "None"):
        modules.util.appstate.global_pipe = Lumina2Text2ImgPipeline.from_pretrained(
            bfl_repo,
            torch_dtype=dtype,
        )
    elif (quantization == "int8"):
        from diffusers import Lumina2Transformer2DModel
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        transformer_8bit = Lumina2Transformer2DModel.from_pretrained(
            bfl_repo,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        modules.util.appstate.global_pipe = Lumina2Text2ImgPipeline.from_pretrained(
            bfl_repo,
            transformer=transformer_8bit,
            torch_dtype=dtype
        )
    elif (quantization == "int4"):
        from diffusers import Lumina2Transformer2DModel
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        transformer_4bit = Lumina2Transformer2DModel.from_pretrained(
            bfl_repo,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
        modules.util.appstate.global_pipe = Lumina2Text2ImgPipeline.from_pretrained(
            bfl_repo,
            transformer=transformer_4bit,
            torch_dtype=torch.bfloat16
        )
        
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
        
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_quantization = quantization
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling, quantization
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(quantization, memory_optimization, vaeslicing, vaetiling)
        generator = torch.Generator(device="cuda").manual_seed(seed)

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

        # Generate images
        image = pipe(**inference_params).images[0]

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "lumina2.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        image.save(output_path)
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

def create_lumina2_BnB_tab():
    initial_state = state_manager.get_state("lumina2") or {}
    with gr.Row():
        with gr.Column():
            lumina2_quantization = gr.Radio(
                choices=["None", "int4", "int8"],
                label="BitsnBytes quantization",
                value=initial_state.get("quantization", "None"),
                interactive=True
            )
        with gr.Column():
            lumina2_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            lumina2_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            lumina2_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            lumina2_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            lumina2_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                lumina2_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                lumina2_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
                save_state_button = gr.Button("Save State")
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
                    value=initial_state.get("inference_steps", 30),
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

    def save_current_state(quantization, memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps):
        state_dict = {
            "quantization": quantization,
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": int(width),
            "height": int(height),
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("lumina2") or {}
        return state_manager.save_state("lumina2", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            lumina2_quantization,
            lumina2_memory_optimization, 
            lumina2_vaeslicing, 
            lumina2_vaetiling, 
            lumina2_width_input, 
            lumina2_height_input, 
            lumina2_guidance_scale_slider, 
            lumina2_num_inference_steps_input
        ],
        outputs=[gr.Textbox(visible=False)]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, lumina2_prompt_input, lumina2_negative_prompt_input, lumina2_width_input, 
            lumina2_height_input, lumina2_guidance_scale_slider, lumina2_num_inference_steps_input, 
            lumina2_memory_optimization, lumina2_vaeslicing, lumina2_vaetiling, lumina2_quantization
        ],
        outputs=[output_gallery]
    )