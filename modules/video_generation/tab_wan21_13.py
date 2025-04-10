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
from huggingface_hub import snapshot_download
from PIL import Image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/video_generation/wan21"

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
    if inference_type == "Text to video":
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir="models/Wan-AI/Wan2.1-T2V-1.3B",
            local_dir_use_symlinks=False,
            allow_patterns=[
                "diffusion_pytorch_model.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "google/",
                "config.json"
            ]
        )
        snapshot_download(
            "newgenai79/Wan2.1-1.3b-speedcontrol-v1", 
            local_dir="models/newgenai79/Wan2.1-1.3b-speedcontrol-v1",
            local_dir_use_symlinks=False,
            allow_patterns=[
                "model.safetensors",
            ]
        )

        # Load models
        modules.util.appstate.global_model_manager = ModelManager(device="cpu")
        modules.util.appstate.global_model_manager.load_models(
            [ 
                "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                # "models/newgenai79/Wan2.1-1.3b-speedcontrol-v1/model.safetensors"
            ],
            torch_dtype=dtype
        )
    elif inference_type == "Image to video":
        snapshot_download(
            repo_id="alibaba-pai/Wan2.1-Fun-1.3B-InP",
            local_dir="models/alibaba-pai/Wan2.1-Fun-1.3B-InP",
            local_dir_use_symlinks=False,
            allow_patterns=[
                "diffusion_pytorch_model.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "google/",
                "xlm-roberta-large/"
            ]
        )
        # Load models
        modules.util.appstate.global_model_manager = ModelManager(device="cpu")
        modules.util.appstate.global_model_manager.load_models(
            [
                "models/alibaba-pai/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
                "models/alibaba-pai/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
                "models/alibaba-pai/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
                "models/alibaba-pai/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ],
            torch_dtype=dtype
        )
    elif inference_type == "Control":
        snapshot_download(
            repo_id="alibaba-pai/Wan2.1-Fun-1.3B-Control",
            local_dir="models/alibaba-pai/Wan2.1-Fun-1.3B-Control",
            local_dir_use_symlinks=False,
            allow_patterns=[
                "diffusion_pytorch_model.safetensors",
                "models_t5_umt5-xxl-enc-bf16.pth",
                "Wan2.1_VAE.pth",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                "google/",
                "xlm-roberta-large/"
            ]
        )
        modules.util.appstate.global_model_manager = ModelManager(device="cpu")
        modules.util.appstate.global_model_manager.load_models(
            [
                "models/alibaba-pai/Wan2.1-Fun-1.3B-Control/diffusion_pytorch_model.safetensors",
                "models/alibaba-pai/Wan2.1-Fun-1.3B-Control/models_t5_umt5-xxl-enc-bf16.pth",
                "models/alibaba-pai/Wan2.1-Fun-1.3B-Control/Wan2.1_VAE.pth",
                "models/alibaba-pai/Wan2.1-Fun-1.3B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ],
            torch_dtype=dtype
        )

    modules.util.appstate.global_pipe = WanVideoPipeline.from_model_manager(
        modules.util.appstate.global_model_manager, 
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
    num_frames, memory_optimization, quality, tea_cache_l1_thresh,
    inference_type, t2v_mode, image, video
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(inference_type, memory_optimization)
        progress_bar = gr.Progress(track_tqdm=True)
        inference_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "seed": seed,
            "tiled": True,
            "progress_bar_cmd": lambda x: progress_bar.tqdm(x, desc="Generating video")
        }
        if(tea_cache_l1_thresh > 0):
            inference_params["tea_cache_l1_thresh"] = tea_cache_l1_thresh
            inference_params["tea_cache_model_id"]="Wan2.1-T2V-1.3B"
        fname = ""
        if inference_type == "Text to video":
            fname = "t2v"
            if t2v_mode == "Fast":
                inference_params["motion_bucket_id"] = 100
            if t2v_mode == "Slow":
                inference_params["motion_bucket_id"] = 0
        elif inference_type == "Image to video":
            fname = "i2v"
            input_image = Image.open(image)
            inference_params["input_image"] = input_image
        elif inference_type == "Control":
            fname = "control"
            control_video = VideoData(video, height=height, width=width)
            inference_params["control_video"] = control_video
        # Generate video
        video = pipe(**inference_params)
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = f"wan2113B_{fname}.mp4"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the video
        save_video(video, output_path, fps=fps, quality=quality)

        print(f"Video generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        if inference_type == "Image to video":
            del input_image
        if inference_type == "Control":
            del control_video
        del video
        return output_path
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_wan21_tab():
    initial_state = state_manager.get_state("wan21_t2v") or {}
    with gr.Row():
        wan21_inference_type = gr.Radio(
            choices=["Text to video", "Image to video", "Control"],
            label="Inference type",
            value=initial_state.get("inference_type", "Text to video"),
            interactive=True
        )
        wan21_quality_slider = gr.Slider(
            label="Video quality (10-highest quality)", 
            minimum=1, 
            maximum=10, 
            value=5,
            step=1,
            interactive=True
        )
    with gr.Row():
        wan21_memory_optimization = gr.Radio(
            choices=["bfloat16", "float8_e4m3fn"],
            label="Memory Optimization",
            value=initial_state.get("memory_optimization", "bfloat16"),
            interactive=True
        )
        wan21_tea_cache_l1_thresh_slider = gr.Slider(
            label="Tea cache (larger value - faster inference but low quality) (Set the value to 0 to disable)", 
            minimum=0, 
            maximum=1, 
            value=0.05,
            step=0.01,
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            wan21_t2v_mode = gr.Radio(
                choices=["Normal", "Fast", "Slow"],
                label="Video speed (T2V)",
                value=initial_state.get("t2v_mode", "Normal"),
                interactive=True
            )
        with gr.Column():
            wan21_image = gr.Image(label="Image (I2V)", type="filepath", width=256, height=256, interactive=False)
        with gr.Column():
            wan21_video = gr.Video(label="Video (Control)", show_label=True, width=256, height=256, interactive=False)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                wan21_prompt_input = gr.Textbox(
                    label="Prompt", 
                    value="A documentary style picture shows a lively puppy running quickly on the green grass. The puppy has brown fur, two ears standing up, and a focused and cheerful expression. The sun shines on its body, making its fur look particularly soft and shiny. The background is an open grassland, occasionally dotted with wild flowers, and the blue sky and a few white clouds can be vaguely seen in the distance. The perspective is clear, capturing the dynamics of the puppy running and the vitality of the surrounding grass. The side view of the middle ground moves.",
                    lines=6,
                    interactive=True
                )
            with gr.Row():
                wan21_negative_prompt_input = gr.Textbox(
                    label="Negative prompt", 
                    lines=3,
                    value="bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
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
                wan21_fps_input = gr.Number(
                    label="FPS", 
                    value=15,
                    interactive=True
                )
                wan21_num_inference_steps_input = gr.Number(
                    label="Inference Steps", 
                    value=50,
                    interactive=True
                )
                wan21_num_frames_input = gr.Number(
                    label="Number of frames [(8*n)+1]", 
                    value=81,
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
    def toggle_interactive(inference_type):
        return gr.update(interactive=(inference_type == "Text to video")), gr.update(interactive=(inference_type == "Image to video")), gr.update(interactive=(inference_type == "Control"))

    wan21_inference_type.change(toggle_interactive, inputs=[wan21_inference_type], outputs=[wan21_t2v_mode, wan21_image, wan21_video])
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
            wan21_num_frames_input, wan21_memory_optimization, wan21_quality_slider,
            wan21_tea_cache_l1_thresh_slider,
            wan21_inference_type, wan21_t2v_mode, wan21_image, wan21_video
        ],
        outputs=[output_video]
    )