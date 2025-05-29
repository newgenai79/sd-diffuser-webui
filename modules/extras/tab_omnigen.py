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
from diffusers import OmniGenPipeline
from diffusers.utils import load_image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/extras/omnigen"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, vaeslicing, vaetiling):
    print("----OmniGen mode: ", memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "OmniGenPipeline" and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing OmniGen pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
    
    repo_id = "Shitao/OmniGen-v1-diffusers"

    modules.util.appstate.global_pipe = OmniGenPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        cache_dir=f"models/{repo_id}",
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
    
    return modules.util.appstate.global_pipe

def generate_image(
    memory_optimization, vaeslicing, vaetiling, input_image1, input_image2, 
    input_image3, prompt, width, height, guidance_scale, img_guidance_scale, 
    num_inference_steps, use_input_image_size_as_output, seed, max_input_image_size
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, vaeslicing, vaetiling)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            print(i, num_inference_steps)
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs
        input_images = [input_image1, input_image2, input_image3]
        input_images = [img for img in input_images if img is not None]
        if len(input_images) == 0:
            input_images = None
        
        """
        # Prepare inference parameters
        inference_params = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }
        input_images = []
        if input_image1 is not None:
            input_images.append(input_image1)
            if input_image2 is not None:
                input_images.append(input_image2)
            if input_image3 is not None:
                input_images.append(input_image3)

        # Add input_images to params only if there are any images selected
        if len(input_images) > 0:
            inference_params["input_images"] = input_images
            inference_params["img_guidance_scale"] = img_guidance_scale
            inference_params["use_input_image_size_as_output"] = use_input_image_size_as_output
        if not use_input_image_size_as_output or len(input_images) == 0:
            inference_params["width"] = width
            inference_params["height"] = height
        print(inference_params)
        """
        inference_params = {
            "prompt": prompt,
            "input_images": input_images,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "img_guidance_scale": img_guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "max_input_image_size": max_input_image_size,
        }
        if input_images is not None and len(input_images) > 0:
            inference_params["use_input_image_size_as_output"] = use_input_image_size_as_output
        print(inference_params)
        image = pipe(**inference_params).images[0]
        """
        # Generate image
        image = pipe(
            prompt=prompt,
            input_images=input_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=num_inference_steps,
            use_input_image_size_as_output=use_input_image_size_as_output,
            generator=generator,
            max_input_image_size=max_input_image_size,
        ).images[0]
        """
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "omnigen.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        image.save(output_path)
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "OmniGen"))
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_omnigen_tab():
    initial_state = state_manager.get_state("omnigen") or {}
    with gr.Row():
        with gr.Column():
            omnigen_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Extremely Low VRAM"),
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                omnigen_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", False), interactive=True)
                omnigen_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            omnigen_input_image1 = gr.Image(label="<img><|image_1|></img>", type="pil")
        with gr.Column():
            omnigen_input_image2 = gr.Image(label="<img><|image_2|></img>", type="pil")
        with gr.Column():
            omnigen_input_image3 = gr.Image(label="<img><|image_3|></img>", type="pil")
    with gr.Row():
        with gr.Column():
            omnigen_prompt = gr.Textbox(
                label="Prompt", 
                lines=15,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                omnigen_width = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                omnigen_height = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
            with gr.Row():
                omnigen_guidance_scale = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=5.0, 
                    value=initial_state.get("guidance_scale", 2.5),
                    step=0.1,
                    interactive=True
                )
                omnigen_img_guidance_scale = gr.Slider(
                    label="Image Guidance Scale", 
                    minimum=1.0, 
                    maximum=2.0, 
                    value=initial_state.get("img_guidance_scale", 1.6),
                    step=0.1,
                    interactive=True
                )
                omnigen_max_input_image_size = gr.Slider(
                    label="Max input image size", 
                    minimum=128, 
                    maximum=2048, 
                    value=initial_state.get("max_input_image_size", 1024),
                    step=16,
                    interactive=True
                )
                omnigen_num_inference_steps = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
                    interactive=True
                )
                omnigen_use_input_image_size_as_output = gr.Checkbox(label="use_input_image_size_as_output", value=initial_state.get("use_input_image_size_as_output", True), interactive=True)
            with gr.Row():
                omnigen_seed = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
                save_state_button = gr.Button("Save State")
    with gr.Row():
        generate_button = gr.Button("Generate image")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    def save_current_state(memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, img_guidance_scale, num_inference_steps, use_input_image_size_as_output, max_input_image_size):
        state_dict = {
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "img_guidance_scale": img_guidance_scale,
            "num_inference_steps": num_inference_steps,
            "use_input_image_size_as_output": use_input_image_size_as_output,
            "max_input_image_size": max_input_image_size
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("omnigen") or {}
        return state_manager.save_state("omnigen", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[omnigen_seed])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            omnigen_memory_optimization, 
            omnigen_vaeslicing, 
            omnigen_vaetiling, 
            omnigen_width, 
            omnigen_height, 
            omnigen_guidance_scale, 
            omnigen_img_guidance_scale, 
            omnigen_num_inference_steps, 
            omnigen_use_input_image_size_as_output,
            omnigen_max_input_image_size
        ],
        outputs=[gr.Textbox(visible=False)]
    )

    generate_button.click(
        fn=generate_image,
        inputs=[
            omnigen_memory_optimization, omnigen_vaeslicing, omnigen_vaetiling, 
            omnigen_input_image1, omnigen_input_image2, omnigen_input_image3, 
            omnigen_prompt, omnigen_width, omnigen_height, omnigen_guidance_scale, 
            omnigen_img_guidance_scale, omnigen_num_inference_steps, 
            omnigen_use_input_image_size_as_output, omnigen_seed, omnigen_max_input_image_size
        ],
        outputs=[output_gallery]
    )