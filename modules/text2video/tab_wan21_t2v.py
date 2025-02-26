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
from diffsynth.data.video import save_video, VideoData
from modelscope import snapshot_download
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2v/wan21"

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
    seed, prompt, negative_prompt, width, height, fps, num_inference_steps, 
    num_frames, memory_optimization, quality
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline("wan21t2v", memory_optimization)
        progress_bar = gr.Progress(track_tqdm=True)
        # Generate video
        video = pipe(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            seed=seed, 
            tiled=True,
            progress_bar_cmd=lambda x: progress_bar.tqdm(x, desc="Processing")
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "wan21.mp4"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the video
        save_video(video, output_path, fps=fps, quality=quality)
        print(f"Video generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        
        return output_path
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_wan21_t2v_tab():
    initial_state = state_manager.get_state("wan21_t2v") or {}
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
                    lines=3,
                    value="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                    interactive=True
                )
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
                with gr.Column():
                    wan21_fps_input = gr.Number(
                        label="FPS", 
                        value=initial_state.get("fps", 15),
                        interactive=True
                    )
                with gr.Column():
                    wan21_num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps", 
                        value=initial_state.get("inference_steps", 50),
                        interactive=True
                    )
                with gr.Column():
                    wan21_num_frames_input = gr.Number(
                        label="Number of frames", 
                        value=initial_state.get("no_of_frames", 81),
                        interactive=True
                    )
            with gr.Row():
                wan21_quality_slider = gr.Slider(
                    label="Video quality", 
                    minimum=1, 
                    maximum=10, 
                    value=initial_state.get("quality", 5),
                    step=1,
                    interactive=True
                )
            # with gr.Row():
                # save_state_button = gr.Button("Save State")
        with gr.Column():
            output_video = gr.Video(label="Generated Video", show_label=True)
    with gr.Row():
        generate_button = gr.Button("Generate video")
    

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
        initial_state = state_manager.get_state("wan21_t2v") or {}
        return state_manager.save_state("wan21_t2v", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    """
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            wan21_memory_optimization, 
            wan21_vaeslicing,
            wan21_vaetiling,
            wan21_width_input, 
            wan21_height_input, 
            wan21_fps_input, 
            wan21_num_inference_steps_input,
            wan21_num_frames_input,
            wan21_guidance_scale_slider
        ],
        outputs=[gr.Textbox(visible=False)]
    )
    """
    generate_button.click(
        fn=generate_video,
        inputs=[
            seed_input, wan21_prompt_input, wan21_negative_prompt_input, wan21_width_input, 
            wan21_height_input, wan21_fps_input, wan21_num_inference_steps_input, 
            wan21_num_frames_input, wan21_memory_optimization, wan21_quality_slider
        ],
        outputs=[output_video]
    )