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
from diffusers import SanaPipeline, SanaSprintPipeline
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Sana"
style_list = [
    {
        "name": "(No style)",
        "prompt": "{prompt}",
        "negative_prompt": "",
    },
    {
        "name": "Cinematic",
        "prompt": "cinematic still {prompt} . emotional, harmonious, vignette, highly detailed, high budget, bokeh, "
        "cinemascope, moody, epic, gorgeous, film grain, grainy",
        "negative_prompt": "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    },
    {
        "name": "Photographic",
        "prompt": "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed",
        "negative_prompt": "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly",
    },
    {
        "name": "Anime",
        "prompt": "anime artwork {prompt} . anime style, key visual, vibrant, studio anime,  highly detailed",
        "negative_prompt": "photo, deformed, black and white, realism, disfigured, low contrast",
    },
    {
        "name": "Manga",
        "prompt": "manga style {prompt} . vibrant, high-energy, detailed, iconic, Japanese comic style",
        "negative_prompt": "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, Western comic style",
    },
    {
        "name": "Digital Art",
        "prompt": "concept art {prompt} . digital artwork, illustrative, painterly, matte painting, highly detailed",
        "negative_prompt": "photo, photorealistic, realism, ugly",
    },
    {
        "name": "Pixel art",
        "prompt": "pixel-art {prompt} . low-res, blocky, pixel art style, 8-bit graphics",
        "negative_prompt": "sloppy, messy, blurry, noisy, highly detailed, ultra textured, photo, realistic",
    },
    {
        "name": "Fantasy art",
        "prompt": "ethereal fantasy concept art of  {prompt} . magnificent, celestial, ethereal, painterly, epic, "
        "majestic, magical, fantasy art, cover art, dreamy",
        "negative_prompt": "photographic, realistic, realism, 35mm film, dslr, cropped, frame, text, deformed, "
        "glitch, noise, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, "
        "disfigured, sloppy, duplicate, mutated, black and white",
    },
    {
        "name": "Neonpunk",
        "prompt": "neonpunk style {prompt} . cyberpunk, vaporwave, neon, vibes, vibrant, stunningly beautiful, crisp, "
        "detailed, sleek, ultramodern, magenta highlights, dark purple shadows, high contrast, cinematic, "
        "ultra detailed, intricate, professional",
        "negative_prompt": "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
    },
    {
        "name": "3D Model",
        "prompt": "professional 3d model {prompt} . octane render, highly detailed, volumetric, dramatic lighting",
        "negative_prompt": "ugly, deformed, noisy, low poly, blurry, painting",
    },
]
def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(inference_type, memory_optimization, vaeslicing, vaetiling):
    print("----Sana mode: ",inference_type, memory_optimization, vaeslicing, vaetiling)
    if (modules.util.appstate.global_pipe is not None and 
        (type(modules.util.appstate.global_pipe).__name__ == "SanaPipeline" or type(modules.util.appstate.global_pipe).__name__ == "SanaSprintPipeline") and
        modules.util.appstate.global_inference_type == inference_type and 
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Sana pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
        torch.cuda.synchronize()
    
    # Determine model path based on inference type
    if inference_type == "Sana_1600M_512px_MultiLing":
        model_path = "Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers"
        dtype = torch.float16
        variant="fp16"
    elif inference_type == "Sana 1K":
        model_path = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
        dtype = torch.bfloat16
        variant="bf16"
    elif inference_type == "Sana 2K":
        model_path = "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers"
        dtype = torch.bfloat16
        variant="bf16"
    elif inference_type == "Sana 4K":
        model_path = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"
        dtype = torch.bfloat16
        variant="bf16"
    elif inference_type == "Twig-v0-alpha":
        model_path = "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
        dtype = torch.bfloat16
        variant="bf16"
    elif inference_type == "Sana Sprint 1.6B":
        model_path = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
        dtype = torch.bfloat16
        variant=None
    elif inference_type == "Sana v1.5 1.6B 1K":
        model_path = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
        dtype = torch.bfloat16
        variant=None
    elif inference_type == "Sana v1.5 4.8B 1K":
        model_path = "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"
        dtype = torch.bfloat16
        variant=None
    # Initialize pipeline
    if inference_type == "Twig-v0-alpha":
        from diffusers import SanaTransformer2DModel
        transformer = SanaTransformer2DModel.from_single_file (
            "Swarmeta-AI/Twig-v0-alpha/Twig-v0-alpha-1.6B-2048x-fp16.pth",
            torch_dtype=dtype,
        )
        modules.util.appstate.global_pipe = SanaPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            transformer=transformer,
            torch_dtype=dtype,
            use_safetensors=True,
        )
    else:
        if inference_type == "Sana_1600M_512px_MultiLing":
            modules.util.appstate.global_pipe = SanaPipeline.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        elif inference_type == "Sana Sprint 1.6B":
            modules.util.appstate.global_pipe = SanaSprintPipeline.from_pretrained(
                pretrained_model_name_or_path=model_path,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        else:
            modules.util.appstate.global_pipe = SanaPipeline.from_pretrained(
                pretrained_model_name_or_path=model_path,
                variant=variant,
                torch_dtype=dtype,
                use_safetensors=True,
            )
    
    if memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()
    elif memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
    if vaeslicing:
        modules.util.appstate.global_pipe.enable_vae_slicing()
    else:
        modules.util.appstate.global_pipe.disable_vae_slicing()
    if vaetiling:
        if inference_type == "Sana 4K":
            modules.util.appstate.global_pipe.vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024, tile_sample_stride_height=896, tile_sample_stride_width=896,)
        else:
            modules.util.appstate.global_pipe.enable_vae_tiling()
    else:
        modules.util.appstate.global_pipe.disable_vae_tiling()
    if inference_type == "Sana 4K":
        if modules.util.appstate.global_pipe.transformer.config.sample_size == 128:
            from patch_conv import convert_model
            modules.util.appstate.global_pipe.vae = convert_model(modules.util.appstate.global_pipe.vae, splits=32)
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = inference_type
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, inference_type, 
    vaeslicing, vaetiling, sana_dropdown, no_of_images, randomize_seed
):

    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Apply style template if a style is selected
        if sana_dropdown != "(No style)":
            selected_style = next((style for style in style_list if style["name"] == sana_dropdown), None)
            if selected_style:
                prompt = selected_style["prompt"].replace("{prompt}", prompt)
                if not negative_prompt:  # Only override if no custom negative prompt
                    negative_prompt = selected_style["negative_prompt"]

        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(inference_type, memory_optimization, vaeslicing, vaetiling)
        # generator = torch.Generator(device="cpu").manual_seed(seed)
        
        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs
        # Get base filename based on inference type
        if inference_type == "Sana_1600M_512px_MultiLing":
            base_filename = "sana_512.png"
        elif inference_type == "Sana 1K":
            base_filename = "sana_1K.png"
        elif inference_type == "Sana 2K":
            base_filename = "sana_2K.png"
        elif inference_type == "Sana 4K":
            base_filename = "sana_4K.png"
        elif inference_type == "Twig-v0-alpha":
            base_filename = "sana_Twig-v0-alpha.png"
        elif inference_type == "Sana Sprint 1.6B":
            base_filename = "sana_sprint.png"
        elif inference_type == "Sana v1.5 1.6B 1K":
            base_filename = "sana_v1.5_1.6B_1K.png"            
        elif inference_type == "Sana v1.5 4.8B 1K":
            base_filename = "sana_v1.5_4.8B_1K.png"
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        gallery_items = []
        for img_idx in range(no_of_images):
            current_seed = random_seed() if randomize_seed else seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            inference_params = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if inference_type == "Sana Sprint 1.6B":
                if num_inference_steps != 2:
                    inference_params["intermediate_timesteps"] = None
            else:
                inference_params["negative_prompt"] = negative_prompt
            # Generate images
            start_time = datetime.now()
            image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            # Generate unique timestamp for each image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{img_idx+1}_{base_filename}"
            output_path = os.path.join(OUTPUT_DIR, filename)
            metadata = {
                "inference_type": inference_type,
                "style": sana_dropdown,
                "prompt": prompt,
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
            if not inference_type == "Sana Sprint 1.6B":
                metadata["negative_prompt"] = negative_prompt
            # Save the image
            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image {img_idx+1}/{no_of_images} generated: {output_path}")
            
            # Add to gallery items
            gallery_items.append((output_path, f"{inference_type}"))
        modules.util.appstate.global_inference_in_progress = False
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def update_style_prompts(style_name):
    """Update the prompt template and negative prompt based on selected style"""
    if not style_name or style_name == "(No style)":
        return "", ""  # Return empty strings for both template and negative prompt
    
    selected_style = next((style for style in style_list if style["name"] == style_name), None)
    if selected_style:
        return selected_style["prompt"], selected_style["negative_prompt"]
    return "", ""

def create_sana_tab():
    initial_state = state_manager.get_state("sana") or {}
    with gr.Row():
        sana_inference_type = gr.Radio(
            choices=["Sana Sprint 1.6B", "Sana v1.5 1.6B 1K", "Sana v1.5 4.8B 1K", "Sana 1K", "Sana 2K", "Sana 4K"], # "Sana_1600M_512px_MultiLing", "Sana_1600M_1024px_MultiLing",  "Twig-v0-alpha"
            label="Inference type",
            value=initial_state.get("inference_type", "Sana Sprint 1.6B"),
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            sana_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            sana_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            sana_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                sana_prompt_input = gr.Textbox(
                    label="Prompt", 
                    placeholder="", 
                    lines=3,
                    interactive=True
                )
            with gr.Row():
                sana_negative_prompt_input = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="",
                    lines=3,
                    interactive=False
                )
            with gr.Row():
                sana_style_dropdown = gr.Dropdown(
                    choices=[style["name"] for style in style_list],
                    value="(No style)",
                    label="Style"
                )
                sana_prompt_style_template_input = gr.Textbox(
                    label="Prompt Style Template",
                    interactive=False
                )
        with gr.Column():
            with gr.Row():
                sana_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    minimum=512, 
                    maximum=4096, 
                    step=64,
                    interactive=True
                )
                sana_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
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
                    value=initial_state.get("guidance_scale", 7.0),
                    step=0.1,
                    interactive=True
                )
                sana_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 2),
                    interactive=True
                )
            with gr.Row():
                sana_no_of_images_input = gr.Number(
                    label="Number of Images", 
                    value=1,
                    interactive=True
                )
                sana_randomize_seed = gr.Checkbox(label="Randomize seed", value=False, interactive=True)

    with gr.Row():
        generate_button = gr.Button("Generate image(s)")
        save_state_button = gr.Button("Save State")
    output_gallery = gr.Gallery(
        label="Generated Images",
        columns=3,
        rows=None,
        height="auto"
    )

    def update_dimensions(selected_type):
        if selected_type == "Sana 1K":
            return (1024, 1024, 20, gr.update(interactive=True))
        elif selected_type == "Sana 2K":
            return (2048, 2048, 20, gr.update(interactive=True))
        elif selected_type == "Sana 4K":
            return (4096, 4096, 20, gr.update(interactive=True))
        elif selected_type == "Sana Sprint 1.6B":
            return (1024, 1024, 2, gr.update(interactive=False))
        else:
            return (1024, 1024, 20, gr.update(interactive=True))

    sana_style_dropdown.change(
        fn=update_style_prompts,
        inputs=[sana_style_dropdown],
        outputs=[sana_prompt_style_template_input, sana_negative_prompt_input]
    )
    sana_inference_type.change(
        fn=update_dimensions,
        inputs=[sana_inference_type],
        outputs=[sana_width_input, sana_height_input, sana_num_inference_steps_input, sana_negative_prompt_input]
    )

    def save_current_state(inference_type, memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps):
        state_dict = {
            "inference_type": inference_type,
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": int(width),
            "height": int(height),
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("sana") or {}
        state_manager.save_state("sana", state_dict)
        return inference_type, memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            sana_inference_type,
            sana_memory_optimization, 
            sana_vaeslicing, 
            sana_vaetiling, 
            sana_width_input, 
            sana_height_input, 
            sana_guidance_scale_slider, 
            sana_num_inference_steps_input
        ],
        outputs=[
            sana_inference_type,
            sana_memory_optimization, 
            sana_vaeslicing, 
            sana_vaetiling, 
            sana_width_input, 
            sana_height_input, 
            sana_guidance_scale_slider, 
            sana_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, sana_prompt_input, sana_negative_prompt_input, sana_width_input, 
            sana_height_input, sana_guidance_scale_slider, sana_num_inference_steps_input, 
            sana_memory_optimization, sana_inference_type, sana_vaeslicing, sana_vaetiling, 
            sana_style_dropdown, sana_no_of_images_input, sana_randomize_seed
        ],
        outputs=[output_gallery]
    )