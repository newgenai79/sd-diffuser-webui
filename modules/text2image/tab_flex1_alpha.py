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
from diffusers import FluxPipeline, FluxTransformer2DModel
from modules.util.utilities import clear_previous_model_memory

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flex.1_alpha"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, quantization, vaeslicing, vaetiling):
    model_id = "ostris/Flex.1-alpha"
    dtype = torch.bfloat16
    print("----Flex.1_alpha mode: ",memory_optimization, quantization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.config.global_pipe is not None and 
        type(modules.util.config.global_pipe).__name__ == "FluxPipeline" and
        modules.util.config.global_quantization == quantization and
        modules.util.config.global_memory_mode == memory_optimization):
        print(">>>>Reusing Flex.1_alpha pipe<<<<")
        return modules.util.config.global_pipe
    else:
        clear_previous_model_memory()
        
    if quantization == "int8wo":
        from diffusers import TorchAoConfig
        quantization_config = TorchAoConfig("int8wo")
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            torch_dtype=dtype,
        )
    modules.util.config.global_pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=dtype,
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
        
    # Update global variables
    modules.util.config.global_memory_mode = memory_optimization
    modules.util.config.global_quantization = quantization
    return modules.util.config.global_pipe

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, quantization, vaeslicing, vaetiling
):

    if modules.util.config.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.config.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, quantization, vaeslicing, vaetiling)
        generator = torch.Generator(device="cuda").manual_seed(seed)

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
            "max_sequence_length":512,
            "callback_on_step_end": callback_on_step_end,
        }

        # Generate images
        image = pipe(**inference_params).images[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "flex1_alpha.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        image.save(output_path)
        print(f"Image generated: {output_path}")
        modules.util.config.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "Flex.1-alpha"))
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.config.global_inference_in_progress = False

def create_flex1_alpha_tab():
    with gr.Row():
        with gr.Column(scale=0):
            flex1_alpha_quantization = gr.Radio(
                choices=["No quantization", "int8wo"],
                label="TorchAO quantization",
                value="int8wo",
                interactive=True
            )
        with gr.Column(scale=2):
            flex1_alpha_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                label="Memory Optimization",
                value="Low VRAM",
                interactive=True
            )
        with gr.Column(scale=0):
            flex1_alpha_vaeslicing = gr.Checkbox(label="VAE slicing", value=True, interactive=True)
            flex1_alpha_vaetiling = gr.Checkbox(label="VAE Tiling", value=True, interactive=True)
    with gr.Row():
        with gr.Column():
            flex1_alpha_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            flex1_alpha_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                flex1_alpha_width_input = gr.Number(
                    label="Width", 
                    value=512, 
                    interactive=True
                )
                flex1_alpha_height_input = gr.Number(
                    label="Height", 
                    value=512, 
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                flex1_alpha_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=1.0, 
                    step=0.1,
                    interactive=True
                )
                flex1_alpha_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=20,
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
            seed_input, flex1_alpha_prompt_input, flex1_alpha_negative_prompt_input, 
            flex1_alpha_width_input, flex1_alpha_height_input, flex1_alpha_guidance_scale_slider, 
            flex1_alpha_num_inference_steps_input, flex1_alpha_memory_optimization, 
            flex1_alpha_quantization, flex1_alpha_vaeslicing, flex1_alpha_vaetiling            
        ],
        outputs=[output_gallery]
    )