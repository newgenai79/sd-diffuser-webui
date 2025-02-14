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
from diffusers import Lumina2Text2ImgPipeline, Lumina2Transformer2DModel
from diffusers import GGUFQuantizationConfig
from diffusers import BitsAndBytesConfig
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Lumina"
gguf_list = [
    "lumina2-q2_k.gguf - 857 MB",
    "lumina2-q3_k_m.gguf - 1.12 GB",
    "lumina2-q4_0.gguf - 1.47 GB",
    "lumina2-q4_1.gguf - 1.63 GB",
    "lumina2-q4_k_m.gguf - 1.47 GB",
    "lumina2-q5_0.gguf - 1.79 GB",
    "lumina2-q5_1.gguf - 1.96 GB",
    "lumina2-q5_k_m.gguf - 1.79 GB",
    "lumina2-q6_k.gguf - 2.14 GB",
    "lumina2-q8_0.gguf - 2.77 GB"
]
def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()
def get_dimensions(resolution):
    width, height = map(int, resolution.split('x'))
    return width, height
def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            if ((wp * patch_size)//32) % 2 == 0 and  ((hp * patch_size)//32) % 2 == 0:
                crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list
def getGGUF(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size

def get_pipeline(inference_type, gguf_file, quantization, memory_optimization, vaeslicing, vaetiling):
    print("----Lumina mode: ", inference_type, gguf_file, quantization, memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and type(modules.util.appstate.global_pipe).__name__ == "Lumina2Text2ImgPipeline"):
        if (inference_type == "Default"):
            if(modules.util.appstate.global_inference_type == inference_type and 
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing Lumina2 Default pipe<<<<")
                return modules.util.appstate.global_pipe
            else:
                clear_previous_model_memory()
        elif (inference_type == "GGUF"):
            if(modules.util.appstate.global_inference_type == inference_type and 
                modules.util.appstate.global_selected_gguf == gguf_file and 
                modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing Lumina2 GGUF pipe<<<<")
                return modules.util.appstate.global_pipe
            else:
                clear_previous_model_memory()
        elif (inference_type == "BnB"):
            if(modules.util.appstate.global_inference_type == inference_type and 
            modules.util.appstate.global_quantization == quantization and 
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing Lumina2 BnB pipe<<<<")
                return modules.util.appstate.global_pipe
            else:
                clear_previous_model_memory()
        else:
            clear_previous_model_memory()
            return None
    else:
        clear_previous_model_memory()

    bfl_repo = "Alpha-VLLM/Lumina-Image-2.0"
    dtype = torch.bfloat16
    
    if (inference_type == "Default"):
        modules.util.appstate.global_pipe = Lumina2Text2ImgPipeline.from_pretrained(
            bfl_repo,
            torch_dtype=dtype,
        )
    elif (inference_type == "GGUF"):
        transformer_path = f"https://huggingface.co/calcuis/lumina-gguf/blob/main/{gguf_file}"
        transformer = Lumina2Transformer2DModel.from_single_file(
            transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=dtype,
        )

        modules.util.appstate.global_pipe = Lumina2Text2ImgPipeline.from_pretrained(
            bfl_repo,
            transformer=transformer,
            torch_dtype=dtype,
        )
    elif (inference_type == "BnB"):
        if (quantization == "int8"):
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
                torch_dtype=dtype
            )

    if memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()
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
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, negative_prompt, resolution, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling, 
    inference_type, gguf, quantization
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        width, height = get_dimensions(resolution)
        gguf_file, gguf_file_size = getGGUF(gguf)
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(inference_type, gguf_file, quantization, memory_optimization, vaeslicing, vaetiling)

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
        
        base_filename = "lumina2.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        metadata = {
            "model": "Lumina-Image-2.0",
            "inference_type": inference_type,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "resolution": resolution,
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
        gallery_items.append((output_path, "Lumina 2"))
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_lumina2_tab():
    initial_state = state_manager.get_state("lumina2") or {}
    with gr.Row():
        with gr.Column():
            lumina2_inference_type = gr.Radio(
                choices=["Default", "GGUF", "BnB"],
                label="Inference type",
                value=initial_state.get("inference_type", "GGUF"),
                interactive=True
            )
            lumina2_gguf = gr.Dropdown(
                choices=gguf_list,
                value=initial_state.get("gguf", "lumina2-q8_0.gguf - 2.77 GB"),
                label="Select GGUF",
                interactive=True,
                visible=initial_state.get("inference_type", "GGUF") == "GGUF"
            )
            lumina2_quantization = gr.Radio(
                choices=["int4", "int8"],
                label="BitsnBytes quantization",
                value=initial_state.get("quantization", "int4"),
                interactive=True,
                visible=initial_state.get("inference_type", "GGUF") == "BnB"
            )
        with gr.Column():
            with gr.Row():
                lumina2_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM"],
                    label="Memory Optimization",
                    value=initial_state.get("memory_optimization", "Low VRAM"),
                    interactive=True
                )
            with gr.Row():
                lumina2_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", False), interactive=True)
                lumina2_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", False), interactive=True)
    with gr.Row():
        with gr.Column():
            lumina2_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=6,
                interactive=True
            )
            lumina2_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                lumina2_resolution_dropdown = gr.Dropdown(
                    choices=[f"{w}x{h}" for w, h in generate_crop_size_list((1024 // 64) ** 2, 64)],
                    value=initial_state.get("resolution", "1024x1024"),
                    label="Resolution"
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                dummy = gr.Textbox(
                    label="", 
                    visible=False
                )
                random_button = gr.Button("♻️ Randomize seed")
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
                    value=initial_state.get("inference_steps", 40),
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

    def save_current_state(inference_type, gguf, quantization, memory_optimization, vaeslicing, vaetiling, resolution, guidance_scale, inference_steps):
        state_dict = {
            "inference_type": inference_type,
            "gguf": gguf,
            "quantization": quantization,
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "resolution": resolution,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        state_manager.save_state("lumina2", state_dict)
        return (inference_type, gguf, quantization, memory_optimization, vaeslicing, vaetiling, resolution, guidance_scale, inference_steps)
    def update_visibility(inference_type):
        return (
            gr.update(visible=(inference_type == "GGUF")),
            gr.update(visible=(inference_type == "BnB"))
        )
    lumina2_inference_type.change(
        fn=update_visibility,
        inputs=[lumina2_inference_type],
        outputs=[lumina2_gguf, lumina2_quantization]
    )
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            lumina2_inference_type,
            lumina2_gguf,
            lumina2_quantization,
            lumina2_memory_optimization, 
            lumina2_vaeslicing, 
            lumina2_vaetiling, 
            lumina2_resolution_dropdown, 
            lumina2_guidance_scale_slider, 
            lumina2_num_inference_steps_input
        ],
        outputs=[
            lumina2_inference_type,
            lumina2_gguf,
            lumina2_quantization,
            lumina2_memory_optimization,
            lumina2_vaeslicing,
            lumina2_vaetiling,
            lumina2_resolution_dropdown,
            lumina2_guidance_scale_slider,
            lumina2_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, lumina2_prompt_input, lumina2_negative_prompt_input, 
            lumina2_resolution_dropdown, lumina2_guidance_scale_slider, 
            lumina2_num_inference_steps_input, lumina2_memory_optimization, 
            lumina2_vaeslicing, lumina2_vaetiling, lumina2_inference_type, 
            lumina2_gguf, lumina2_quantization
        ],
        outputs=[output_gallery]
    )