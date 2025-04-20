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
OUTPUT_DIR = "output/video_generation/vace"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(inference_type, memory_optimization):
    print("----Vace mode: ", inference_type, memory_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "WanVideoPipeline" and
        modules.util.appstate.global_inference_type == inference_type and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Vace pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
    if(memory_optimization == "bfloat16"):
        dtype=torch.bfloat16
    else:
        dtype=torch.float8_e4m3fn

    snapshot_download(
        repo_id="ali-vilab/VACE-Wan2.1-1.3B-Preview",
        local_dir="models/iic/VACE-Wan2.1-1.3B-Preview",
        local_dir_use_symlinks=False,
        allow_patterns=[
            "diffusion_pytorch_model.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
            "google/",
            "config.json"
        ]
    )

    # Load models
    modules.util.appstate.global_model_manager = ModelManager(device="cpu")
    modules.util.appstate.global_model_manager.load_models(
        [ 
            "models/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors",
            "models/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth",
            "models/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth",
        ],
        torch_dtype=dtype
    )

    modules.util.appstate.global_pipe = WanVideoPipeline.from_model_manager(
        modules.util.appstate.global_model_manager, 
        torch_dtype=torch.bfloat16, 
        device="cuda"
    )
    modules.util.appstate.global_pipe.enable_vram_management(
        # num_persistent_param_in_dit=None
        # num_persistent_param_in_dit=4000000000
        num_persistent_param_in_dit=0
    )

    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = inference_type
    return modules.util.appstate.global_pipe

def generate_video(
    seed, prompt, negative_prompt, width, height, fps, num_inference_steps, 
    num_frames, memory_optimization, quality, tea_cache_l1_thresh,
    inference_type, v2v_video, i2v_reference_image, dr2v_video, dr2v_reference_image
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline("vace", memory_optimization)
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
            # "progress_bar_cmd": lambda x: progress_bar.tqdm(x, desc="Generating video")
        }
        """
        if(tea_cache_l1_thresh > 0):
            inference_params["tea_cache_l1_thresh"] = tea_cache_l1_thresh
            # inference_params["tea_cache_model_id"]="Wan2.1-T2V-1.3B"
            inference_params["tea_cache_model_id"]="VACE-Wan2.1-1.3B-Preview"
        """
        fname = ""
        "Depth V2V", "Reference I2V", "Depth DR2V"
        if inference_type == "Depth V2V":
            fname = "depth"
            control_video = VideoData(v2v_video, height=height, width=width)
            inference_params["vace_video"] = control_video
        elif inference_type == "Reference I2V":
            fname = "reference"
            input_image = Image.open(i2v_reference_image).resize((832, 480))
            inference_params["vace_reference_image"] = input_image
        elif inference_type == "Depth DR2V":
            fname = "depth_reference"
            control_video = VideoData(dr2v_video, height=height, width=width)
            inference_params["vace_video"] = control_video
            input_image = Image.open(dr2v_reference_image).resize((832, 480))
            inference_params["vace_reference_image"] = input_image
        # Generate video
        import pprint
        pprint.pprint(inference_params)
        video = pipe(**inference_params)
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = f"vace_{fname}.mp4"
        
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
            if end_image:
                del end_input_image
        if inference_type == "Control":
            del control_video
        del video
        return output_path
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_vace_tab():
    initial_state = state_manager.get_state("vace") or {}
    with gr.Row():
        vace_inference_type = gr.Radio(
            choices=["Depth V2V", "Reference I2V", "Depth DR2V"],
            label="Inference type",
            value=initial_state.get("inference_type", "Depth V2V"),
            interactive=True
        )
        vace_quality_slider = gr.Slider(
            label="Video quality (10-highest quality)", 
            minimum=1, 
            maximum=10, 
            value=5,
            step=1,
            interactive=True
        )
    with gr.Row():
        vace_memory_optimization = gr.Radio(
            choices=["bfloat16", "float8_e4m3fn"],
            label="Memory Optimization",
            value=initial_state.get("memory_optimization", "float8_e4m3fn"),
            interactive=True
        )
        vace_tea_cache_l1_thresh_slider = gr.Slider(
            label="Tea cache (Not supported)", 
            minimum=0, 
            maximum=1, 
            value=0,
            step=0.01,
            interactive=False
        )
    with gr.Row():
        with gr.Column(scale=20):
            with gr.Accordion("Depth Video to Video", open=True) as accordion_v2v:
                vace_v2v_video = gr.Video(label="Video", show_label=True, width=256, height=256, interactive=True)
        with gr.Column(scale=20):
            with gr.Accordion("Reference Image to Video", open=False) as accordion_i2v:
                vace_i2v_reference_image = gr.Image(label="Image", type="filepath", width=256, height=256, interactive=False)
        with gr.Column(scale=60):
            with gr.Accordion("Depth video + Reference image to Video", open=False) as accordion_dr2v:
                with gr.Row():
                    vace_dr2v_video = gr.Video(label="Video", show_label=True, width=256, height=256, interactive=False)
                    vace_dr2v_reference_image = gr.Image(label="Reference image", type="filepath", width=256, height=256, interactive=False)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                vace_prompt_input = gr.Textbox(
                    label="Prompt", 
                    value="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
                    lines=6,
                    interactive=True
                )
            with gr.Row():
                vace_negative_prompt_input = gr.Textbox(
                    label="Negative prompt", 
                    lines=3,
                    value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    interactive=True
                )
            with gr.Row():
                vace_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 832),
                    interactive=False
                )
                vace_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 480),
                    interactive=False
                )
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                vace_fps_input = gr.Number(
                    label="FPS", 
                    value=15,
                    interactive=True
                )
                vace_num_inference_steps_input = gr.Number(
                    label="Inference Steps", 
                    value=50,
                    interactive=True
                )
                vace_num_frames_input = gr.Number(
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
        initial_state = state_manager.get_state("vace") or {}
        return state_manager.save_state("vace", state_dict)

    # Event handlers
    def toggle_interactive(inference_type):
        return gr.update(open=(inference_type == "Depth V2V")), gr.update(interactive=(inference_type == "Depth V2V")), gr.update(open=(inference_type == "Reference I2V")), gr.update(interactive=(inference_type == "Reference I2V")),  gr.update(open=(inference_type == "Depth DR2V")), gr.update(interactive=(inference_type == "Depth DR2V")),gr.update(interactive=(inference_type == "Depth DR2V"))

    vace_inference_type.change(toggle_interactive, inputs=[vace_inference_type], 
        outputs=[accordion_v2v, vace_v2v_video, accordion_i2v, vace_i2v_reference_image, accordion_dr2v, vace_dr2v_video, vace_dr2v_reference_image])
    random_button.click(fn=random_seed, outputs=[seed_input])
    """
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            vace_memory_optimization, 
            vace_vaeslicing,
            vace_vaetiling,
            vace_width_input, 
            vace_height_input, 
            vace_fps_input, 
            vace_num_inference_steps_input,
            vace_num_frames_input,
            vace_guidance_scale_slider
        ],
        outputs=[gr.Textbox(visible=False)]
    )
    """
    generate_button.click(
        fn=generate_video,
        inputs=[
            seed_input, vace_prompt_input, vace_negative_prompt_input, vace_width_input, 
            vace_height_input, vace_fps_input, vace_num_inference_steps_input, 
            vace_num_frames_input, vace_memory_optimization, vace_quality_slider,
            vace_tea_cache_l1_thresh_slider, vace_inference_type,
            vace_v2v_video, vace_i2v_reference_image, vace_dr2v_video, vace_dr2v_reference_image

        ],
        outputs=[output_video]
    )