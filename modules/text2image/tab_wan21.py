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
from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video import WanVideoPipeline
from modelscope import snapshot_download
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/wan211.3"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(inference_type, memory_optimization):
    print("----Wan 2.1 mode: ", inference_type, memory_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "WanVideoPipeline" and
        modules.util.appstate.global_inference_type == inference_type and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Wan 2.1 pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
    if(memory_optimization == "bfloat16"):
        dtype=torch.bfloat16
    else:
        dtype=torch.float8_e4m3fn

    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")

    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
            "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
            "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        ],
        torch_dtype=dtype
    )

    modules.util.appstate.global_pipe = WanVideoPipeline.from_model_manager(
        model_manager, 
        torch_dtype=torch.bfloat16, 
        device="cuda"
    )

    modules.util.appstate.global_pipe.enable_vram_management(
        num_persistent_param_in_dit=None
    )

    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = inference_type
    return modules.util.appstate.global_pipe

def generate_video(
    seed, prompt, negative_prompt, width, height, num_inference_steps, 
    memory_optimization, cfg_scale
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        gallery_items = []
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline("wan21t2v", memory_optimization)
        progress_bar = gr.Progress(track_tqdm=True)
        start_time = datetime.now()
        # Generate video
        video = pipe(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            num_frames=1,
            seed=seed, 
            tiled=True,
            progress_bar_cmd=lambda x: progress_bar.tqdm(x, desc="Processing")
        )
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "wan21.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        metadata = {
            "model": "Wan2.1-1.3B",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "memory_optimization": memory_optimization,
            "timestamp": timestamp,
            "generation_time": generation_time
        }
        # Save the video
        video[0].save(output_path)
        modules.util.utilities.save_metadata_to_file(output_path, metadata)
        gallery_items.append((output_path, "Wan 2.1 - 1.3B"))
        
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_wan21_t2i_tab():
    initial_state = state_manager.get_state("wan21_t2i") or {}
    with gr.Row():
        with gr.Column():
            wan21_memory_optimization = gr.Radio(
                choices=["bfloat16", "float8_e4m3fn"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "bfloat16"),
                interactive=True
            )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                wan21_prompt_input = gr.Textbox(
                    label="Prompt", 
                    lines=6,
                    interactive=True
                )
            with gr.Row():
                wan21_negative_prompt_input = gr.Textbox(
                    label="Negative prompt", 
                    lines=4,
                    value="",
                    interactive=True
                )
        with gr.Column():
            with gr.Row():
                wan21_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 832),
                    interactive=True
                )
                wan21_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 480),
                    interactive=True
                )
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():

                wan21_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
                    interactive=True
                )

                wan21_guidance_scale_slider = gr.Slider(
                    label="Cfg Scale", 
                    minimum=1.0, 
                    maximum=10.0, 
                    value=initial_state.get("guidance_scale", 5.0),
                    step=0.1,
                    interactive=True
                )
            with gr.Row():
                save_state_button = gr.Button("Save State")
    with gr.Row():
        generate_button = gr.Button("Generate image")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    def save_current_state(memory_optimization,  width, height, inference_steps, guidance_scale):
        state_dict = {
            "memory_optimization": memory_optimization,
            "width": int(width),
            "height": int(height),
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale
            
        }
        initial_state = state_manager.get_state("wan21_t2i") or {}
        return state_manager.save_state("wan21_t2i", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            wan21_memory_optimization, 
            wan21_width_input, 
            wan21_height_input, 
            wan21_num_inference_steps_input,
            wan21_guidance_scale_slider
        ],
        outputs=[gr.Textbox(visible=False)]
    )
    generate_button.click(
        fn=generate_video,
        inputs=[
            seed_input, wan21_prompt_input, wan21_negative_prompt_input, wan21_width_input, 
            wan21_height_input, wan21_num_inference_steps_input, wan21_memory_optimization, 
            wan21_guidance_scale_slider
        ],
        outputs=[output_gallery]
    )