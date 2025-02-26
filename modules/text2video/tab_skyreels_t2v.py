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
from diffusers.utils import export_to_video
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2v/skyreels"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(inference_type, memory_optimization, vaeslicing, vaetiling):
    print("----skyreels mode: ", inference_type, memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "HunyuanVideoPipeline" and
        modules.util.appstate.global_inference_type == inference_type and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Skyreel pipe<<<<")
        # return modules.util.appstate.global_pipe
        clear_previous_model_memory()
    else:
        clear_previous_model_memory()

    repo_id = "newgenai79/HunyuanVideo-int4"
    transformer_model_id = "newgenai79/SkyReels-V1-Hunyuan-T2V-int4"
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        transformer_model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    modules.util.appstate.global_pipe = HunyuanVideoPipeline.from_pretrained(
        repo_id, 
        transformer=transformer,
        torch_dtype=torch.bfloat16
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
    modules.util.appstate.global_inference_type = inference_type
    return modules.util.appstate.global_pipe

def generate_video(
    seed, prompt, width, height, fps, num_inference_steps, num_frames, 
    memory_optimization, vaeslicing, vaetiling, guidance_scale
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline("skyreelst2v", memory_optimization, vaeslicing, vaetiling)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating video (Step {i}/{num_inference_steps})")
            return callback_kwargs
        # Prepare inference parameters
        inference_params = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }

        # Generate video
        video = pipe(**inference_params).frames[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "skyreels.mp4"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the video
        export_to_video(video, output_path, fps=fps)
        print(f"Video generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        
        return output_path
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_skyreels_t2v_tab():
    initial_state = state_manager.get_state("skyreels_t2v") or {}
    with gr.Row():
        with gr.Column():
            skyreels_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            skyreels_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            skyreels_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
    with gr.Row():
        with gr.Column():
            skyreels_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=10,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                skyreels_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 512),
                    interactive=True
                )
                skyreels_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 512),
                    interactive=True
                )
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=42, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                skyreels_fps_input = gr.Number(
                    label="FPS", 
                    value=initial_state.get("fps", 24),
                    interactive=True
                )
                skyreels_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
                    interactive=True
                )
            with gr.Row():
                skyreels_num_frames_input = gr.Number(
                    label="Number of frames", 
                    value=initial_state.get("no_of_frames", 49),
                    interactive=True
                )
                skyreels_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=10.0, 
                    value=initial_state.get("guidance_scale", 3.0),
                    step=0.1,
                    interactive=True
                )
            with gr.Row():
                save_state_button = gr.Button("Save State")
    with gr.Row():
        generate_button = gr.Button("Generate video")
    output_video = gr.Video(label="Generated Video", show_label=True)

    def save_current_state(memory_optimization, vaeslicing, vaetiling, width, height, fps, inference_steps, no_of_frames, guidance_scale):
        state_dict = {
            "memory_optimization": memory_optimization,
            "guidance_scale": guidance_scale,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": int(width),
            "height": int(height),
            "fps": fps,
            "inference_steps": inference_steps,
            "no_of_frames": no_of_frames
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("skyreels_t2v") or {}
        return state_manager.save_state("skyreels_t2v", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            skyreels_memory_optimization, 
            skyreels_vaeslicing,
            skyreels_vaetiling,
            skyreels_width_input, 
            skyreels_height_input, 
            skyreels_fps_input, 
            skyreels_num_inference_steps_input,
            skyreels_num_frames_input,
            skyreels_guidance_scale_slider
        ],
        outputs=[gr.Textbox(visible=False)]
    )

    generate_button.click(
        fn=generate_video,
        inputs=[
            seed_input, skyreels_prompt_input, skyreels_width_input, 
            skyreels_height_input, skyreels_fps_input, skyreels_num_inference_steps_input, 
            skyreels_num_frames_input, skyreels_memory_optimization, skyreels_vaeslicing,
            skyreels_vaetiling, skyreels_guidance_scale_slider
        ],
        outputs=[output_video]
    )