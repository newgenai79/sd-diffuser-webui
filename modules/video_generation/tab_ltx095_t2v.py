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
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from diffusers.utils import export_to_video, load_video, load_image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/video_generation/ltx095"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, vaeslicing, vaetiling):
    print("----LTX Video 0.9.5 mode: ", memory_optimization, vaeslicing, vaetiling)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "LTXConditionPipeline" and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing LTX Video 0.9.5 pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    base_model_path = "models/Lightricks"
    modules.util.appstate.global_pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.5", 
        torch_dtype=torch.bfloat16,
        cache_dir=base_model_path
    )
    if memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    elif memory_optimization == "Low VRAM":
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

    return modules.util.appstate.global_pipe

def generate_video(
    inference_type, quality, memory_optimization, vaeslicing, vaetiling, 
    prompt, negative_prompt, width, height, seed, fps, 
    num_inference_steps, num_frames, image,
    conditioning1_image, conditioning1_video, frame_index1,
    conditioning2_image, conditioning2_video, frame_index2,
    conditioning3_image, conditioning3_video, frame_index3
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, vaeslicing, vaetiling)
        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        inference_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "num_frames": num_frames,
            "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }
        if inference_type == "Image to video":
            image_i2v = load_image( image )
            condition = LTXVideoCondition(
                image=image_i2v,
                frame_index=0,
            )
            inference_params["conditions"] = [condition]
        elif inference_type == "Conditioning":
            conditions = []
            if (conditioning1_image or conditioning1_video) and frame_index1 is not None:
                if (conditioning1_image):
                    image = load_image( conditioning1_image )
                    condition = LTXVideoCondition(
                        image=image,
                        frame_index=frame_index1,
                    )
                    conditions.append(condition)
                else:
                    video = load_video( conditioning1_video )
                    condition = LTXVideoCondition(
                        video=video,
                        frame_index=frame_index1,
                    )
                    conditions.append(condition)
            if (conditioning2_image or conditioning2_video) and frame_index2 is not None:
                if (conditioning2_image):
                    image = load_image( conditioning2_image )
                    condition = LTXVideoCondition(
                        image=image,
                        frame_index=frame_index2,
                    )
                    conditions.append(condition)
                else:
                    video = load_video( conditioning2_video )
                    condition = LTXVideoCondition(
                        video=video,
                        frame_index=frame_index2,
                    )
                    conditions.append(condition)
            if (conditioning3_image or conditioning3_video) and frame_index3 is not None:
                if (conditioning3_image):
                    image = load_image( conditioning3_image )
                    condition = LTXVideoCondition(
                        image=image,
                        frame_index=frame_index3,
                    )
                    conditions.append(condition)
                else:
                    video = load_video( conditioning3_video )
                    condition = LTXVideoCondition(
                        video=video,
                        frame_index=frame_index3,
                    )
                    conditions.append(condition)
            inference_params["conditions"] = conditions
            if not conditions:
                print("Input data")
                return False
        # Generate video
        video = pipe(**inference_params).frames[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "ltx095.mp4"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the video
        export_to_video(video, output_path, fps=fps, quality=float(quality))
        # export_to_video(video, output_path, fps=fps)
        print(f"Video generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        
        return output_path
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_ltx095_tab():
    initial_state = state_manager.get_state("ltx095_t2v") or {}
    with gr.Row():
        ltx095_inference_type = gr.Radio(
            choices=["Text to video", "Image to video", "Conditioning"],
            label="Inference type",
            value=initial_state.get("inference_type", "Text to video"),
            interactive=True
        )
        ltx095_quality = gr.Slider(
            label="Video quality (10-highest quality)", 
            minimum=1, 
            maximum=10, 
            value=5,
            step=1,
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            ltx095_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Extremely Low VRAM"),
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                ltx095_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", False), interactive=True)
                ltx095_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", False), interactive=True)
    with gr.Row():
        with gr.Accordion("Select images/videos for Conditioning inference", open=False) as accordion_conditoning:
            with gr.Group():
                with gr.Row():
                    ltx095_conditioning1_image = gr.Image(label="Image", type="filepath", width=256, height=256)
                    ltx095_conditioning1_video = gr.Video(label="Video", show_label=True, width=256, height=256)
                    ltx095_frame_index1 = gr.Number(label="Frame index", value=0, precision=0, interactive=True )
            gr.Markdown("---")
            with gr.Group():
                with gr.Row():
                    ltx095_conditioning2_image = gr.Image(label="Image", type="filepath", width=256, height=256)
                    ltx095_conditioning2_video = gr.Video(label="Video", show_label=True, width=256, height=256)
                    ltx095_frame_index2 = gr.Number(label="Frame index", value=0, precision=0, interactive=True )
            gr.Markdown("---")
            with gr.Group():
                with gr.Row():
                    ltx095_conditioning3_image = gr.Image(label="Image", type="filepath", width=256, height=256)
                    ltx095_conditioning3_video = gr.Video(label="Video", show_label=True, width=256, height=256)
                    ltx095_frame_index3 = gr.Number(label="Frame index", value=0, precision=0, interactive=True )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Accordion("Select images for Image to Video inference", open=False) as accordion_i2v:
                    ltx095_image = gr.Image(label="Image (Mandatory for I2V)", type="filepath", width=384, height=384)

            with gr.Row():
                ltx095_prompt = gr.Textbox(
                    label="Prompt", 
                    lines=6,
                    interactive=True
                )
            with gr.Row():
                ltx095_negative_prompt = gr.Textbox(
                    label="Negative prompt", 
                    lines=3,
                    value="worst quality, inconsistent motion, blurry, jittery, distorted",
                    interactive=True
                )
            with gr.Row():
                ltx095_width = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 768),
                    interactive=True
                )
                ltx095_height = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 512),
                    interactive=True
                )
            with gr.Row():
                ltx095_seed = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                ltx095_fps= gr.Number(
                    label="FPS", 
                    value=24,
                    interactive=True
                )
                ltx095_num_inference_steps = gr.Number(
                    label="Inference Steps", 
                    value=40,
                    interactive=True
                )
                ltx095_num_frames = gr.Number(
                    label="Number of frames [(8*n)+1]", 
                    value=161,
                    interactive=True
                )

            # with gr.Row():
                # save_state_button = gr.Button("Save State")
        with gr.Column():
            output_video = gr.Video(label="Generated Video", show_label=True)
    with gr.Row():
        generate_button = gr.Button("Generate video")
    
    def toggle_accordion(inference_type):
        return gr.update(open=(inference_type == "Image to video")), gr.update(open=(inference_type == "Conditioning"))
    ltx095_inference_type.change(toggle_accordion, inputs=[ltx095_inference_type], outputs=[accordion_i2v, accordion_conditoning])

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
        initial_state = state_manager.get_state("ltx095_t2v") or {}
        return state_manager.save_state("ltx095_t2v", state_dict)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[ltx095_seed])
    """
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            ltx095_memory_optimization, 
            ltx095_vaeslicing,
            ltx095_vaetiling,
            ltx095_width_input, 
            ltx095_height_input, 
            ltx095_fps_input, 
            ltx095_num_inference_steps_input,
            ltx095_num_frames_input,
            ltx095_guidance_scale_slider
        ],
        outputs=[gr.Textbox(visible=False)]
    )
    """
    generate_button.click(
        fn=generate_video,
        inputs=[
            ltx095_inference_type, ltx095_quality, ltx095_memory_optimization, 
            ltx095_vaeslicing, ltx095_vaetiling, ltx095_prompt, 
            ltx095_negative_prompt, ltx095_width, ltx095_height, ltx095_seed, 
            ltx095_fps, ltx095_num_inference_steps, ltx095_num_frames, ltx095_image,
            ltx095_conditioning1_image, ltx095_conditioning1_video, ltx095_frame_index1,
            ltx095_conditioning2_image, ltx095_conditioning2_video, ltx095_frame_index2,
            ltx095_conditioning3_image, ltx095_conditioning3_video, ltx095_frame_index3
        ],
        outputs=[output_video]
    )