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
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import GGUFQuantizationConfig
from diffusers import BitsAndBytesConfig
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"

flux_dev_gguf_list = [
    "flux1-dev-Q2_K.gguf - 4.03 GB", 
    "flux1-dev-Q3_K_S.gguf - 5.23 GB", 
    "flux1-dev-Q4_0.gguf - 6.79 GB", 
    "flux1-dev-Q4_1.gguf - 7.53 GB", 
    "flux1-dev-Q4_K_S.gguf - 6.81 GB", 
    "flux1-dev-Q5_0.gguf - 8.27 GB", 
    "flux1-dev-Q5_1.gguf - 9.01 GB", 
    "flux1-dev-Q5_K_S.gguf - 8.29 GB", 
    "flux1-dev-Q6_K.gguf - 9.86 GB", 
    "flux1-dev-Q8_0.gguf - 12.7 GB"
]
flux_schnell_gguf_list = [
    "flux1-schnell-Q2_K.gguf - 4.01 GB", 
    "flux1-schnell-Q3_K_S.gguf - 5.21 GB", 
    "flux1-schnell-Q4_0.gguf - 6.77 GB", 
    "flux1-schnell-Q4_1.gguf - 7.51 GB", 
    "flux1-schnell-Q4_K_S.gguf - 6.78 GB", 
    "flux1-schnell-Q5_0.gguf - 8.25 GB", 
    "flux1-schnell-Q5_1.gguf - 8.99 GB", 
    "flux1-schnell-Q5_K_S.gguf - 8.26 GB", 
    "flux1-schnell-Q6_K.gguf - 9.83 GB", 
    "flux1-schnell-Q8_0.gguf - 12.7 GB"
]
def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def getGGUF(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size

def get_pipeline(model_type, inference_type, gguf_file, quantization, memory_optimization, vaeslicing, vaetiling):
    print("----Flux mode: ", model_type, inference_type, gguf_file, quantization, memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
    type(modules.util.appstate.global_pipe).__name__ == "FluxPipeline" and 
    modules.util.appstate.global_model_type == model_type):
        if (inference_type == "Default"):
            if(modules.util.appstate.global_inference_type == inference_type and 
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing flux Default pipe<<<<")
                return modules.util.appstate.global_pipe
            else:
                clear_previous_model_memory()
        elif (inference_type == "GGUF"):
            if(modules.util.appstate.global_inference_type == inference_type and 
                modules.util.appstate.global_selected_gguf == gguf_file and 
                modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing flux GGUF pipe<<<<")
                return modules.util.appstate.global_pipe
            else:
                clear_previous_model_memory()
        elif (inference_type == "BnB"):
            if(modules.util.appstate.global_inference_type == inference_type and 
            modules.util.appstate.global_quantization == quantization and 
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing flux BnB pipe<<<<")
                return modules.util.appstate.global_pipe
            else:
                clear_previous_model_memory()
        else:
            clear_previous_model_memory()
            return None
    else:
        clear_previous_model_memory()

    bfl_repo = model_type
    dtype = torch.bfloat16
    
    if (inference_type == "Default"):
        modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
            bfl_repo,
            torch_dtype=dtype,
        )
    elif (inference_type == "GGUF"):
        if (model_type == "black-forest-labs/FLUX.1-dev"):
            URL = "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/"
        elif (model_type == "black-forest-labs/FLUX.1-schnell"):
            URL = "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/"
        
        transformer_path = f"{URL}{gguf_file}"
        transformer = FluxTransformer2DModel.from_single_file(
            transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=dtype,
        )

        modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
            bfl_repo,
            transformer=transformer,
            torch_dtype=dtype,
        )
    elif (inference_type == "BnB"):
        if (quantization == "int8"):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            transformer_8bit = FluxTransformer2DModel.from_pretrained(
                bfl_repo,
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )
            modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
                bfl_repo,
                transformer=transformer_8bit,
                torch_dtype=dtype
            )
        elif (quantization == "int4"):
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            transformer_4bit = FluxTransformer2DModel.from_pretrained(
                bfl_repo,
                subfolder="transformer",
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )
            modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
                bfl_repo,
                transformer=transformer_4bit,
                torch_dtype=dtype
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
    modules.util.appstate.global_quantization = quantization
    modules.util.appstate.global_selected_gguf = gguf_file
    modules.util.appstate.global_model_type = model_type
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling, 
    inference_type, gguf, quantization, model_type
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        gguf_file, gguf_file_size = getGGUF(gguf)
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(model_type, inference_type, gguf_file, quantization, memory_optimization, vaeslicing, vaetiling)

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
        
        base_filename = "flux.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        metadata = {
            "model": model_type,
            "inference_type": inference_type,
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
        if (inference_type == "GGUF"):
            metadata["gguf_model"] = gguf_file
        if (inference_type == "BnB"):
            metadata["quantization"] = quantization
        # Save the image
        image.save(output_path)
        modules.util.utilities.save_metadata_to_file(output_path, metadata)
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "Flux"))
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_flux_tab():
    initial_state = state_manager.get_state("flux") or {}
    with gr.Row():
        flux_model_type = gr.Dropdown(
            choices=["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"],
            value=initial_state.get("model_type", "black-forest-labs/FLUX.1-dev"),
            label="Select model",
            interactive=True,
        )
    with gr.Row():
        with gr.Column():
            flux_inference_type = gr.Radio(
                choices=["Default", "GGUF", "BnB"],
                label="Inference type",
                value=initial_state.get("inference_type", "GGUF"),
                interactive=True
            )
            flux_gguf = gr.Dropdown(
                choices=flux_dev_gguf_list,
                value=initial_state.get("gguf", "flux1-dev-Q2_K.gguf - 4.03 GB"),
                label="Select GGUF",
                interactive=True,
                visible=initial_state.get("inference_type", "GGUF") == "GGUF"
            )
            flux_quantization = gr.Radio(
                choices=["int4", "int8"],
                label="BitsnBytes quantization",
                value=initial_state.get("quantization", "int4"),
                interactive=True,
                visible=initial_state.get("inference_type", "GGUF") == "BnB"
            )
        with gr.Column():
            with gr.Row():
                flux_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    value=initial_state.get("memory_optimization", "Low VRAM"),
                    interactive=True
                )
            with gr.Row():
                flux_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
                flux_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            flux_prompt_input = gr.Textbox(
                label="Prompt", 
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
                    value=initial_state.get("guidance_scale", 1.0),
                    step=0.1,
                    interactive=True
                )
                flux_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 20),
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

    def save_current_state(model_type, inference_type, gguf, quantization, memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps):
        state_dict = {
            "model_type": model_type,
            "inference_type": inference_type,
            "gguf": gguf,
            "quantization": quantization,
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("flux") or {}
        state_manager.save_state("flux", state_dict)
        return model_type, inference_type, gguf, quantization, memory_optimization, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps
    def update_visibility(inference_type):
        return (
            gr.update(visible=(inference_type == "GGUF")),
            gr.update(visible=(inference_type == "BnB"))
        )
    flux_inference_type.change(
        fn=update_visibility,
        inputs=[flux_inference_type],
        outputs=[flux_gguf, flux_quantization]
    )
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flux_model_type,
            flux_inference_type,
            flux_gguf,
            flux_quantization,
            flux_memory_optimization, 
            flux_vaeslicing, 
            flux_vaetiling, 
            flux_width_input, 
            flux_height_input, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input
        ],
        outputs=[
            flux_model_type,
            flux_inference_type,
            flux_gguf,
            flux_quantization,
            flux_memory_optimization, 
            flux_vaeslicing, 
            flux_vaetiling, 
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
            flux_num_inference_steps_input, flux_memory_optimization, 
            flux_vaeslicing, flux_vaetiling, flux_inference_type, 
            flux_gguf, flux_quantization, flux_model_type
        ],
        outputs=[output_gallery]
    )

    def update_gguf_choices(model_type):
        if model_type == "black-forest-labs/FLUX.1-dev":
            return gr.update(choices=flux_dev_gguf_list, value=flux_dev_gguf_list[0])
        elif model_type == "black-forest-labs/FLUX.1-schnell":
            return gr.update(choices=flux_schnell_gguf_list, value=flux_schnell_gguf_list[0])

    # Update the event handler - no changes needed here, but included for context
    flux_model_type.change(
        fn=update_gguf_choices,
        inputs=[flux_model_type],
        outputs=[flux_gguf]
    )