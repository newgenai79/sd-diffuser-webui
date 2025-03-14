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
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(model_type, memory_optimization):
    print("----Flux mode: ", model_type, memory_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "FluxPipeline" and 
            modules.util.appstate.global_model_type == model_type and
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing flux Default pipe<<<<")
                return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
        torch.cuda.synchronize()
    if(model_type == "FLUX.1-dev"):
        transformer_repo = "mit-han-lab/svdq-int4-flux.1-dev"
        bfl_repo = "black-forest-labs/FLUX.1-dev"
    elif(model_type == "FLUX.1-schnell"):
        transformer_repo = "mit-han-lab/svdq-int4-flux.1-schnell"
        bfl_repo = "black-forest-labs/FLUX.1-schnell"

    dtype = torch.bfloat16

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        transformer_repo, 
        offload=True
    )
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        "mit-han-lab/svdq-flux.1-t5"
    )
    modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
        bfl_repo, 
        text_encoder_2=text_encoder_2, 
        transformer=transformer, 
        torch_dtype=torch.bfloat16
    )

    if memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
        
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_model_type = model_type
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, model_type,
    no_of_images, randomize_seed
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    
    gallery_items = []
    
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(model_type, memory_optimization)
        progress_bar = gr.Progress(track_tqdm=True)
        
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx+1}: (Step {i}/{num_inference_steps})")
            return callback_kwargs
        guidance_scale = float(guidance_scale)
        # Generate multiple images in a loop
        for img_idx in range(no_of_images):
            modules.util.appstate.global_inference_in_progress = True
            
            # If randomize_seed is True, generate a new random seed for each image
            current_seed = random_seed() if randomize_seed else seed
            
            # Create generator with the current seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
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
            
            print(f"Generating image {img_idx+1}/{no_of_images} with seed: {current_seed}")
            
            start_time = datetime.now()
            # Generate image
            image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Create filename with index
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_flux_schnell_{img_idx+1}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            metadata = {
                "model": model_type,
                "prompt": prompt,
                "seed": current_seed,
                "guidance_scale": f"{float(guidance_scale):.2f}",
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "memory_optimization": memory_optimization,
                "timestamp": timestamp,
                "generation_time": generation_time,
            }
            
            # Save the image
            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image {img_idx+1}/{no_of_images} generated: {output_path}")
            
            # Add to gallery items
            gallery_items.append((output_path, model_type))
            
            modules.util.appstate.global_inference_in_progress = False
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_flux_schnell_tab():
    initial_state = state_manager.get_state("flux-schnell") or {}
    with gr.Row():
        with gr.Column():
            flux_model_type = gr.Dropdown(
                choices=["FLUX.1-schnell"],
                value=initial_state.get("model_type", "FLUX.1-schnell"),
                label="Select model",
                interactive=True,
            )
        with gr.Column():
            flux_memory_optimization = gr.Radio(
                choices=["No optimization", "Extremely Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Extremely Low VRAM"),
                interactive=True
            )
    with gr.Row():
        flux_prompt_input = gr.Textbox(
            label="Prompt (support 77 tokens)", 
            lines=6,
            interactive=True
        )
        with gr.Column():
            with gr.Row():
                flux_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                flux_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                dummy = gr.Textbox(
                    label="", 
                    visible=False
                )
                random_button = gr.Button("♻️ Randomize seed")
            with gr.Row():
                flux_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 3.5),
                    step=0.1,
                    interactive=True
                )
                flux_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
                    interactive=True
                )
            with gr.Row():
                flux_no_of_images_input = gr.Number(
                    label="Number of Images", 
                    value=1,
                    interactive=True
                )
                flux_randomize_seed = gr.Checkbox(label="Randomize seed", value=False, interactive=True)
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

    def save_current_state(model_type, memory_optimization, width, height, guidance_scale, inference_steps):
        state_dict = {
            "model_type": model_type,
            "memory_optimization": memory_optimization,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("flux-schnell") or {}
        state_manager.save_state("flux-schnell", state_dict)
        return model_type, memory_optimization, width, height, guidance_scale, inference_steps
    
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flux_model_type,
            flux_memory_optimization, 
            flux_width_input, 
            flux_height_input, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input
        ],
        outputs=[
            flux_model_type,
            flux_memory_optimization, 
            flux_width_input, 
            flux_height_input, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, flux_prompt_input,  
            flux_width_input, flux_height_input, flux_guidance_scale_slider, 
            flux_num_inference_steps_input, flux_memory_optimization, flux_model_type,
            flux_no_of_images_input, flux_randomize_seed
        ],
        outputs=[output_gallery]
    )