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
from diffusers import SanaPipeline
from modules.util.utilities import clear_previous_model_memory

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Sana"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(inference_type, memory_optimization, vaeslicing, vaetiling):
    print("----Sana mode: ",inference_type, memory_optimization, vaeslicing, vaetiling)
    if (modules.util.config.global_pipe is not None and 
        type(modules.util.config.global_pipe).__name__ == "SanaPipeline" and
        modules.util.config.global_inference_type == inference_type and 
        modules.util.config.global_memory_mode == memory_optimization):
        print(">>>>Reusing Sana pipe<<<<")
        return modules.util.config.global_pipe
    else:
        clear_previous_model_memory()
    
    # Determine model path based on inference type
    if inference_type == "Sana 4K":
        model_path = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"
    else:
        model_path = "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers"

    # Initialize pipeline
    modules.util.config.global_pipe = SanaPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,
        variant="bf16",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    modules.util.config.global_pipe.to("cuda")
    modules.util.config.global_pipe.vae.to(torch.bfloat16)
    modules.util.config.global_pipe.text_encoder.to(torch.bfloat16)
    
    if memory_optimization == "Low VRAM":
        modules.util.config.global_pipe.enable_model_cpu_offload()
    if vaeslicing:
        modules.util.config.global_pipe.enable_vae_slicing()
    else:
        modules.util.config.global_pipe.disable_vae_slicing()
    if vaetiling:
        if inference_type == "Sana 4K":
            modules.util.config.global_pipe.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024, tile_sample_stride_height=896, tile_sample_stride_width=896,)
        else:
            modules.util.config.global_pipe.enable_vae_tiling()
    else:
        modules.util.config.global_pipe.disable_vae_tiling()
    if inference_type == "Sana 4K":
        if modules.util.config.global_pipe.transformer.config.sample_size == 128:
            from patch_conv import convert_model
            modules.util.config.global_pipe.vae = convert_model(modules.util.config.global_pipe.vae, splits=32)
    # Update global variables
    modules.util.config.global_memory_mode = memory_optimization
    modules.util.config.global_inference_type = inference_type
    return modules.util.config.global_pipe

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, inference_type, 
    num_images_per_prompt, vaeslicing, vaetiling, 
):

    if modules.util.config.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.config.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(inference_type, memory_optimization, vaeslicing, vaetiling)
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
            "num_images_per_prompt": num_images_per_prompt,
            "callback_on_step_end": callback_on_step_end,
        }
        
        
        # Generate images
        images = pipe(**inference_params).images
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Get base filename based on inference type
        if inference_type == "Sana 2K":
            base_filename = "sana_2K.png"
        elif inference_type == "Sana 4K":
            base_filename = "sana_4K.png"
        
        # Save each image with unique timestamp and collect paths for gallery
        gallery_items = []
        for idx, image in enumerate(images):
            # Generate unique timestamp for each image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{idx+1}_{base_filename}"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # Save the image
            image.save(output_path)
            print(f"Image {idx+1} generated: {output_path}")
            
            # Add to gallery items
            gallery_items.append((output_path, f"{inference_type}"))
        modules.util.config.global_inference_in_progress = False
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.config.global_inference_in_progress = False

def create_sana_tab():
    with gr.Row():
        with gr.Column():
            sana_inference_type = gr.Radio(
                choices=["Sana 2K", "Sana 4K"],
                label="Inference type",
                value="Sana 2K",
                interactive=True
            )
        with gr.Column():
            sana_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value="Low VRAM",
                interactive=True
            )
        with gr.Column():
            sana_vaeslicing = gr.Checkbox(label="VAE slicing", value=True, interactive=True)
            sana_vaetiling = gr.Checkbox(label="VAE Tiling", value=True, interactive=True)
    with gr.Row():
        with gr.Column():
            sana_prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="", 
                lines=3,
                interactive=True
            )
            sana_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                placeholder="",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                sana_width_input = gr.Number(
                    label="Width", 
                    value=2048, 
                    minimum=512, 
                    maximum=4096, 
                    step=64,
                    interactive=True
                )
                sana_height_input = gr.Number(
                    label="Height", 
                    value=2048, 
                    minimum=512, 
                    maximum=4096, 
                    step=64,
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                sana_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=7.0, 
                    step=0.1,
                    interactive=True
                )
                sana_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=30,
                    interactive=True
                )
                sana_num_images_per_prompt_input = gr.Number(
                    label="Number of images/prompt", 
                    value=1,
                    minimum=1,
                    step=1,
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate image(s)")
    output_gallery = gr.Gallery(
        label="Generated Images",
        columns=3,
        rows=None,  # Allow dynamic rows based on number of images
        height="auto"
    )

    def update_dimensions(selected_type):
        return (4096, 4096) if selected_type == "Sana 4K" else (2048, 2048)
    sana_inference_type.change(update_dimensions, [sana_inference_type], [sana_width_input, sana_height_input])
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, sana_prompt_input, sana_negative_prompt_input, sana_width_input, 
            sana_height_input, sana_guidance_scale_slider, sana_num_inference_steps_input, 
            sana_memory_optimization, sana_inference_type, sana_num_images_per_prompt_input,
            sana_vaeslicing, sana_vaetiling, 
        ],
        outputs=[output_gallery]
    )