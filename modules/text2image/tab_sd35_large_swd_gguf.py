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
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from diffusers import GGUFQuantizationConfig
from peft import PeftModel
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/swd"
gguf_list = [
    "sd3.5_large-F16.gguf - 16.3 GB",
    "sd3.5_large-Q4_0.gguf - 4.77 GB",
    "sd3.5_large-Q4_1.gguf - 5.27 GB",
    "sd3.5_large-Q5_0.gguf - 5.77 GB",
    "sd3.5_large-Q5_1.gguf - 6.27 GB",
    "sd3.5_large-Q8_0.gguf - 8.78 GB",
]
"""
    "SwD-2B_4steps": {
        "repo": "yresearch/swd-medium-4-steps",
        "scales": [256, 512, 768, 1024],
        "scales_inf": [32, 64, 96, 128],
        "sigmas": [1.0000, 0.9454, 0.7904, 0.6022, 0.0000],
        "inference_steps": 4
    },
    "SwD-2B_6steps": {
        "repo": "yresearch/swd-medium-6-steps",
        "scales": [256, 384, 512, 640, 768, 1024],
        "scales_inf": [32, 48, 64, 80, 96, 128],
        "sigmas": [1.0000, 0.9454, 0.8959, 0.7904, 0.7371, 0.6022, 0.0000],
        "inference_steps": 6
    },
"""

models = {
    "SwD-8B_4steps": {
        "repo": "yresearch/swd-large-4-steps",
        "scales": [512, 640, 768, 1024],
        "scales_inf": [64, 80, 96, 128],
        "sigmas": [1.0000, 0.8959, 0.7371, 0.6022, 0.0000],
        "inference_steps": 4
    },
    "SwD-8B_6steps": {
        "repo": "yresearch/swd-large-6-steps",
        "scales": [256, 384, 512, 640, 768, 1024],
        "scales_inf": [32, 48, 64, 80, 96, 128],
        "sigmas": [1.0000, 0.9454, 0.8959, 0.7904, 0.7371, 0.6022, 0.0000],
        "inference_steps": 6
    }
}
default_model = "SwD-8B_4steps"
def get_gguf(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size
def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, gguf_file, vaeslicing, vaetiling, inference_type, selected_lora):
    print("----SWD mode: ", memory_optimization, gguf_file, vaeslicing, vaetiling, inference_type, selected_lora)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "StableDiffusion3Pipeline" and
        modules.util.appstate.global_selected_gguf == gguf_file and
        modules.util.appstate.global_inference_type == inference_type and 
        modules.util.appstate.global_selected_lora == selected_lora and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing SWD pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
    base_model_path = "models/stable-diffusion-3.5-large"
    os.makedirs(base_model_path, exist_ok=True)
    transformer_path = f"https://huggingface.co/city96/stable-diffusion-3.5-large-gguf/blob/main/{gguf_file}"
    transformer_gguf = SD3Transformer2DModel.from_single_file(
        transformer_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        torch_dtype=torch.float16,
        cache_dir=base_model_path
    )

    modules.util.appstate.global_pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        transformer=transformer_gguf,
        torch_dtype=torch.float16,
        custom_pipeline='quickjkee/swd_pipeline',
        cache_dir=base_model_path
    )
    lora_path = models[selected_lora]["repo"]
    modules.util.appstate.global_pipe.transformer = PeftModel.from_pretrained(
        modules.util.appstate.global_pipe.transformer,
        lora_path,
        cache_dir=base_model_path
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

    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_selected_gguf = gguf_file
    modules.util.appstate.global_inference_type = inference_type
    modules.util.appstate.global_selected_lora = selected_lora
    return modules.util.appstate.global_pipe

def get_gguf(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size

def generate_images(
    seed, prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling, gguf_file,
    selected_model, no_of_images, randomize_seed
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        gguf_file, gguf_file_size = get_gguf(gguf_file)
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, gguf_file, vaeslicing, vaetiling, "sd3.5_large_swd_gguf", selected_model)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        progress_bar = gr.Progress(track_tqdm=True)
        sigmas = models[selected_model]["sigmas"]
        base_filename = "sd35_large_swd_gguf.png"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        gallery_items = []
        filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        scales_inf = models[selected_model]["scales_inf"]
        for img_idx in range(no_of_images):
            current_seed = random_seed() if randomize_seed else seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            # Update progress description to show current image
            def callback_on_step_end(pipe, i, t, callback_kwargs):
                progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx + 1} of {no_of_images}: (Step {i}/{num_inference_steps})")
                return callback_kwargs
            
            inference_params = {
                "prompt": prompt,
                "sigmas": torch.tensor(sigmas).to('cuda'),
                "timesteps": torch.tensor(sigmas[:-1]).to('cuda') * 1000,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "scales": scales_inf,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end
            }
            start_time = datetime.now()
            image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Include image index in filename
            filename = f"{filename_timestamp}_img{img_idx + 1}_{base_filename}"
            output_path = os.path.join(OUTPUT_DIR, filename)
            metadata = {
                "base-model": "stabilityai/stable-diffusion-3.5-large",
                "model": selected_model,
                "gguf_file": gguf_file,
                "scales": scales_inf,
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
                "generation_time": generation_time,
            }

            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image generated: {output_path}")
            gallery_items.append((output_path, f"SD3.5-Large/{selected_model}"))
        modules.util.appstate.global_inference_in_progress = False
    
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_sd35_large_swd_gguf_tab():
    initial_state = state_manager.get_state("swd_gguf") or {}
    with gr.Row():
        with gr.Column():
            swd_gguf_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            swd_gguf_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", False), interactive=True)
            swd_gguf_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", False), interactive=True)
    with gr.Row():
        with gr.Column():
            swd_gguf_dropdown = gr.Dropdown(
                choices=gguf_list,
                value=initial_state.get("gguf", "sd3.5_large-Q5_1.gguf - 6.27 GB"),
                label="Select GGUF"
            )
        with gr.Column():
            swd_model_dropdown = gr.Dropdown(
                choices=list(models.keys()),
                value=initial_state.get("model", default_model),
                label="Select SWD model"
            )
    with gr.Row():
        with gr.Column():
            swd_gguf_prompt_input = gr.Textbox(
                label="Prompt",
                lines=4,
                interactive=True
            )
            swd_gguf_guidance_scale_slider = gr.Slider(
                label="Guidance Scale", 
                minimum=0.0, 
                maximum=20.0, 
                value=initial_state.get("guidance_scale", 0.0),
                step=0.1,
                interactive=True
            )
            swd_gguf_num_inference_steps_input = gr.Number(
                label="Number of Inference Steps", 
                value=models[default_model]["inference_steps"],
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                swd_gguf_width_dropdown = gr.Dropdown(
                    label="Width", 
                    choices=models[default_model]["scales"],
                    value=models[default_model]["scales"][0],
                    interactive=True
                )
                swd_gguf_height_dropdown = gr.Dropdown(
                    label="Height", 
                    choices=models[default_model]["scales"],
                    value=models[default_model]["scales"][0],
                    interactive=True
                )
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                swd_no_of_images_input = gr.Number(
                    label="Number of Images", 
                    value=1,
                    interactive=True
                )
                swd_randomize_seed = gr.Checkbox(label="Randomize seed", value=False, interactive=True)
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

    def update_sliders(selected_model):
        scales = models[selected_model]["scales"]
        inference_steps = models[selected_model]["inference_steps"]
        return (
            gr.update(choices=scales, value=scales[0]),
            gr.update(choices=scales, value=scales[0]),
            gr.update(value=inference_steps)
        )
    swd_model_dropdown.change(
        fn=update_sliders,
        inputs=[swd_model_dropdown],
        outputs=[swd_gguf_width_dropdown, swd_gguf_height_dropdown, swd_gguf_num_inference_steps_input]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, swd_gguf_prompt_input, swd_gguf_width_dropdown, 
            swd_gguf_height_dropdown, swd_gguf_guidance_scale_slider, 
            swd_gguf_num_inference_steps_input, swd_gguf_memory_optimization, 
            swd_gguf_vaeslicing, swd_gguf_vaetiling, swd_gguf_dropdown, 
            swd_model_dropdown, swd_no_of_images_input, swd_randomize_seed
        ],
        outputs=[output_gallery]
    )
