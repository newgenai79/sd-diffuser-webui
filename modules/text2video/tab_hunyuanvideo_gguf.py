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
from diffusers import GGUFQuantizationConfig
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2v/hunyuanvideo"
gguf_list = [
    "hunyuan-video-t2v-720p-BF16.gguf - 25.6 GB",
    "hunyuan-video-t2v-720p-Q3_K_M.gguf - 6.24 GB",
    "hunyuan-video-t2v-720p-Q3_K_S.gguf - 6.09 GB",
    "hunyuan-video-t2v-720p-Q4_0.gguf - 7.74 GB",
    "hunyuan-video-t2v-720p-Q4_1.gguf - 8.52 GB",
    "hunyuan-video-t2v-720p-Q4_K_M.gguf - 7.88 GB",
    "hunyuan-video-t2v-720p-Q4_K_S.gguf - 7.74 GB",
    "hunyuan-video-t2v-720p-Q5_0.gguf - 9.3 GB",
    "hunyuan-video-t2v-720p-Q5_1.gguf - 10.1 GB",
    "hunyuan-video-t2v-720p-Q5_K_M.gguf - 9.45 GB",
    "hunyuan-video-t2v-720p-Q5_K_S.gguf - 9.3 GB",
    "hunyuan-video-t2v-720p-Q6_K.gguf - 11 GB",
    "hunyuan-video-t2v-720p-Q8_0.gguf - 14 GB"
]
def get_gguf(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size
def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, gguf_file, vaeslicing, vaetiling):
    print("----hunyuanvideo mode: ", memory_optimization, gguf_file, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "HunyuanVideoPipeline" and
        modules.util.appstate.global_selected_gguf == gguf_file and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing hunyuanvideo pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
    
    transformer_path = f"https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/{gguf_file}"
    transformer = HunyuanVideoTransformer3DModel.from_single_file(
        transformer_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
    modules.util.appstate.global_pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", 
        transformer=transformer,
        torch_dtype=torch.float16
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

    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_selected_gguf = gguf_file
    return modules.util.appstate.global_pipe

def generate_video(
    seed, prompt, width, height, fps,
    num_inference_steps, num_frames, memory_optimization, vaeslicing, vaetiling, gguf_file
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        gguf_file, gguf_file_size = get_gguf(gguf_file)
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, gguf_file, vaeslicing, vaetiling)
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
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }

        # Generate video
        video = pipe(**inference_params).frames[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "hunyuanvideo_gguf.mp4"
        
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

def create_hunyuanvideo_gguf_tab():
    initial_state = state_manager.get_state("hunyuanvideo_gguf") or {}
    with gr.Row():
        with gr.Column():
            hunyuanvideo_gguf_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            hunyuanvideo_gguf_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            hunyuanvideo_gguf_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
        with gr.Column():
            hunyuanvideo_gguf_dropdown = gr.Dropdown(
                choices=gguf_list,
                value=initial_state.get("gguf", "hunyuan-video-t2v-720p-Q3_K_S.gguf - 6.09 GB"),
                label="Select GGUF"
            )
    with gr.Row():
        with gr.Column():
            hunyuanvideo_gguf_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                hunyuanvideo_gguf_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 512),
                    interactive=True
                )
                hunyuanvideo_gguf_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 512),
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
                save_state_button = gr.Button("Save State")
            with gr.Row():
                hunyuanvideo_gguf_fps_input = gr.Number(
                    label="FPS", 
                    value=initial_state.get("fps", 24),
                    interactive=True
                )
                hunyuanvideo_gguf_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
                    interactive=True
                )
                hunyuanvideo_gguf_num_frames_input = gr.Number(
                    label="Number of frames", 
                    value=initial_state.get("no_of_frames", 61),
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate video")
    output_video = gr.Video(label="Generated Video", show_label=True)

    def save_current_state(gguf, memory_optimization, vaeslicing, vaetiling, width, height, fps, inference_steps, no_of_frames):
        state_dict = {
            "gguf": gguf,
            "memory_optimization": memory_optimization,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": int(width),
            "height": int(height),
            "fps": fps,
            "inference_steps": inference_steps,
            "no_of_frames": no_of_frames
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("hunyuanvideo_gguf") or {}
        return state_manager.save_state("hunyuanvideo_gguf", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            hunyuanvideo_gguf_dropdown,
            hunyuanvideo_gguf_memory_optimization, 
            hunyuanvideo_gguf_vaeslicing,
            hunyuanvideo_gguf_vaetiling,
            hunyuanvideo_gguf_width_input, 
            hunyuanvideo_gguf_height_input, 
            hunyuanvideo_gguf_fps_input, 
            hunyuanvideo_gguf_num_inference_steps_input,
            hunyuanvideo_gguf_num_frames_input
        ],
        outputs=[gr.Textbox(visible=False)]
    )

    generate_button.click(
        fn=generate_video,
        inputs=[
            seed_input, hunyuanvideo_gguf_prompt_input, hunyuanvideo_gguf_width_input, 
            hunyuanvideo_gguf_height_input, hunyuanvideo_gguf_fps_input, hunyuanvideo_gguf_num_inference_steps_input, 
            hunyuanvideo_gguf_num_frames_input, hunyuanvideo_gguf_memory_optimization, hunyuanvideo_gguf_vaeslicing,
            hunyuanvideo_gguf_vaetiling, hunyuanvideo_gguf_dropdown
        ],
        outputs=[output_video]
    )