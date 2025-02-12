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
from diffusers import Lumina2Text2ImgPipeline, Lumina2Transformer2DModel, GGUFQuantizationConfig
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
def get_gguf(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size

def get_pipeline(gguf_file, memory_optimization, vaeslicing, vaetiling):
    print("----Lumina mode: ", gguf_file, memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "Lumina2Text2ImgPipeline" and
        modules.util.appstate.global_selected_gguf == gguf_file and 
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Lumina2 pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    bfl_repo = "Alpha-VLLM/Lumina-Image-2.0"
    dtype = torch.bfloat16
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
    modules.util.appstate.global_selected_gguf = gguf_file
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, negative_prompt, resolution, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling, gguf_file
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        width, height = get_dimensions(resolution)
        gguf_file, gguf_file_size = get_gguf(gguf_file)
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(gguf_file, memory_optimization, vaeslicing, vaetiling)
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

        # Generate images
        image = pipe(**inference_params).images[0]

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "lumina2_gguf.png"
        
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

def create_lumina2_gguf_tab():
    initial_state = state_manager.get_state("lumina2_gguf") or {}
    with gr.Row():
        with gr.Column():
            lumina2_gguf_dropdown = gr.Dropdown(
                choices=gguf_list,
                value=initial_state.get("gguf", "lumina2-q8_0.gguf - 2.77 GB"),
                label="Select GGUF"
            )
        with gr.Column():
            lumina2_gguf_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            lumina2_gguf_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            lumina2_gguf_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            lumina2_gguf_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=6,
                interactive=True
            )
            lumina2_gguf_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                lumina2_gguf_resolution_dropdown = gr.Dropdown(
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
                lumina2_gguf_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 4.0),
                    step=0.1,
                    interactive=True
                )
                lumina2_gguf_num_inference_steps_input = gr.Number(
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

    def save_current_state(gguf, memory_optimization, vaeslicing, vaetiling, resolution, guidance_scale, inference_steps):
        state_dict = {
            "gguf": gguf,
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "resolution": resolution,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("lumina2_gguf") or {}
        return state_manager.save_state("lumina2_gguf", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            lumina2_gguf_dropdown,
            lumina2_gguf_memory_optimization, 
            lumina2_gguf_vaeslicing, 
            lumina2_gguf_vaetiling, 
            lumina2_gguf_resolution_dropdown, 
            lumina2_gguf_guidance_scale_slider, 
            lumina2_gguf_num_inference_steps_input
        ],
        outputs=[gr.Textbox(visible=False)]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, lumina2_gguf_prompt_input, lumina2_gguf_negative_prompt_input, 
            lumina2_gguf_resolution_dropdown, lumina2_gguf_guidance_scale_slider, 
            lumina2_gguf_num_inference_steps_input, lumina2_gguf_memory_optimization, 
            lumina2_gguf_vaeslicing, lumina2_gguf_vaetiling, lumina2_gguf_dropdown
        ],
        outputs=[output_gallery]
    )