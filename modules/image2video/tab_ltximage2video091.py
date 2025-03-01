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
from diffusers.utils import export_to_video, load_image
from diffusers import LTXImageToVideoPipeline
from transformers import T5EncoderModel, T5Tokenizer
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/i2v/ltxvideo091"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization):
    print("----ltxvideo image2video 091 mode: ", memory_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "LTXImageToVideoPipeline" and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing ltxvideo091 image2video pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
    
    repo_id = "newgenai79/LTX-Video-0.9.1-diffusers"
    modules.util.appstate.global_pipe = LTXImageToVideoPipeline.from_pretrained(
        repo_id, 
        torch_dtype=torch.bfloat16
    )

    if memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()
    elif memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")

    modules.util.appstate.global_memory_mode = memory_optimization
    
    return modules.util.appstate.global_pipe

def generate_video(
    seed, input_image, prompt, negative_prompt, width, height, fps,
    num_inference_steps, num_frames, memory_optimization,
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating video (Step {i}/{num_inference_steps})")
            return callback_kwargs
        image = load_image(
            input_image
        )
        # Prepare inference parameters
        inference_params = {
            "image": image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
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
        
        base_filename = "ltxvideo091.mp4"
        
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

def create_ltximage2video091_tab():
    initial_state = state_manager.get_state("ltximage2video091") or {}
    with gr.Row():
        ltximage2video091_memory_optimization = gr.Radio(
            choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
            label="Memory Optimization",
            value=initial_state.get("memory_optimization", "Low VRAM"),
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            ltximage2video091_input_image = gr.Image(label="Input Image", type="pil")
        with gr.Column():
            ltximage2video091_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=4,
                interactive=True
            )
            ltximage2video091_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                ltximage2video091_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 704),
                    interactive=True
                )
                ltximage2video091_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 480),
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
                save_state_button = gr.Button("Save State")
            with gr.Row():
                ltximage2video091_fps_input = gr.Number(
                    label="FPS", 
                    value=initial_state.get("fps", 24),
                    interactive=True
                )
                ltximage2video091_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
                    interactive=True
                )
                ltximage2video091_num_frames_input = gr.Number(
                    label="Number of frames", 
                    value=initial_state.get("no_of_frames", 61),
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate video")
    output_video = gr.Video(label="Generated Video", show_label=True)

    def save_current_state(memory_optimization, width, height, fps, inference_steps, no_of_frames):
        state_dict = {
            "memory_optimization": memory_optimization,
            "width": int(width),
            "height": int(height),
            "fps": fps,
            "inference_steps": inference_steps,
            "no_of_frames": no_of_frames
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("ltximage2video091") or {}
        return state_manager.save_state("ltximage2video091", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            ltximage2video091_memory_optimization, 
            ltximage2video091_width_input, 
            ltximage2video091_height_input, 
            ltximage2video091_fps_input, 
            ltximage2video091_num_inference_steps_input,
            ltximage2video091_num_frames_input
        ],
        outputs=[gr.Textbox(visible=False)]
    )

    generate_button.click(
        fn=generate_video,
        inputs=[
            seed_input, ltximage2video091_input_image, ltximage2video091_prompt_input, ltximage2video091_negative_prompt_input, ltximage2video091_width_input, 
            ltximage2video091_height_input, ltximage2video091_fps_input, ltximage2video091_num_inference_steps_input, 
            ltximage2video091_num_frames_input, ltximage2video091_memory_optimization,
        ],
        outputs=[output_video]
    )