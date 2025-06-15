"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os
import gc
import modules.util.appstate
from datetime import datetime
from nunchaku import NunchakuFluxTransformer2dModel
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager
from nunchaku.utils import get_precision

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, inference_type, performance_optimization):
    print("----FluxPipeline mode: ", memory_optimization, inference_type, performance_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "FluxPipeline" and 
            modules.util.appstate.global_inference_type == inference_type and 
            modules.util.appstate.global_performance_optimization == performance_optimization and
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing FluxPipeline pipe<<<<")
                return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
        torch.cuda.synchronize()

    dtype = torch.bfloat16

    precision = get_precision()
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )
    if performance_optimization == "nunchaku-fp16":
        transformer.set_attention_impl("nunchaku-fp16")
    modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        torch_dtype=torch.bfloat16
    )
    if performance_optimization == "apply_cache_on_pipe":
        from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
        apply_cache_on_pipe(
            modules.util.appstate.global_pipe, residual_diff_threshold=0.12
        )
    if memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
        
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = inference_type
    modules.util.appstate.global_performance_optimization = performance_optimization
    return modules.util.appstate.global_pipe

def generate_images(
    seed, guidance_scale, num_inference_steps, 
    memory_optimization, no_of_images, randomize_seed, input_image,
    performance_optimization
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    
    gallery_items = []

    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, "flux.1-dev-redux", performance_optimization)
        progress_bar = gr.Progress(track_tqdm=True)
        
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx+1}: (Step {i}/{num_inference_steps})")
            return callback_kwargs
        guidance_scale = float(guidance_scale)
        modules.util.appstate.global_inference_in_progress = True
        input_image = load_image(input_image)
        pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
        ).to("cuda")
        pipe_prior_output = pipe_prior_redux(input_image)
        # Generate multiple images in a loop
        for img_idx in range(no_of_images):
           
            # If randomize_seed is True, generate a new random seed for each image
            current_seed = random_seed() if randomize_seed else seed
            
            # Create generator with the current seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            start_time = datetime.now()
            # Generate image
            image = pipe(
                guidance_scale=guidance_scale, 
                num_inference_steps=num_inference_steps, 
                **pipe_prior_output
            ).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Create filename with index
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_flux_redux_{img_idx+1}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            metadata = {
                "model": "FLUX.1-dev-redux",
                "seed": current_seed,
                "guidance_scale": f"{float(guidance_scale):.2f}",
                "num_inference_steps": num_inference_steps,
                "memory_optimization": memory_optimization,
                "performance_optimization": performance_optimization,
                "timestamp": timestamp,
                "generation_time": generation_time,
            }
            
            # Save the image
            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image {img_idx+1}/{no_of_images} generated: {output_path}")
            
            # Add to gallery items
            gallery_items.append((output_path, "FLUX.1-dev-redux"))
        modules.util.appstate.global_inference_in_progress = False
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        del pipe_prior_redux
        del pipe_prior_output
        gc.collect()
        modules.util.appstate.global_inference_in_progress = False

def create_flux_redux_tab():
    initial_state = state_manager.get_state("flux-redux") or {}
    with gr.Row():
        flux_memory_optimization = gr.Radio(
            choices=["No optimization", "Extremely Low VRAM"],
            label="Memory Optimization",
            value=initial_state.get("memory_optimization", "Extremely Low VRAM"),
            interactive=True
        )
        flux_performance_optimization = gr.Radio(
            choices=["No optimization", "nunchaku-fp16", "apply_cache_on_pipe"],
            label="Performance Optimization",
            value=initial_state.get("performance_optimization", "apply_cache_on_pipe"),
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                flux_input_image = gr.Image(label="Input Image", type="pil")
        with gr.Column():
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("♻️ Randomize seed")
            with gr.Row():
                flux_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=50.0, 
                    value=initial_state.get("guidance_scale", 10),
                    step=0.1,
                    interactive=True
                )
                flux_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 30),
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
        generate_button = gr.Button("🎨 Generate image")
        save_state_button = gr.Button("💾 Save State")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    def save_current_state(memory_optimization, guidance_scale, inference_steps, performance_optimization):
        state_dict = {
            "memory_optimization": memory_optimization,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "performance_optimization": performance_optimization
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("flux-redux") or {}
        state_manager.save_state("flux-redux", state_dict)
        return memory_optimization, guidance_scale, inference_steps, performance_optimization
    
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flux_memory_optimization, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input,
            flux_performance_optimization
        ],
        outputs=[
            flux_memory_optimization, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input,
            flux_performance_optimization
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, flux_guidance_scale_slider, 
            flux_num_inference_steps_input, flux_memory_optimization, 
            flux_no_of_images_input, flux_randomize_seed, flux_input_image,
            flux_performance_optimization
        ],
        outputs=[output_gallery]
    )