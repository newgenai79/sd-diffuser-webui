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
from diffusers import AutoPipelineForText2Image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Kandinsky3"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization):
    print("----kandinsky3 mode: ", memory_optimization)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "Kandinsky3Pipeline" and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing kandinsky3 pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    modules.util.appstate.global_pipe = AutoPipelineForText2Image.from_pretrained(
        "kandinsky-community/kandinsky-3",
        variant="fp16", 
        torch_dtype=torch.float16,
    )

    if memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()
    elif memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
    modules.util.appstate.global_memory_mode = memory_optimization
    
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization,
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs

        # Prepare inference parameters
        inference_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
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
        
        base_filename = "kandinsky3.png"
        
        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Save the image
        image.save(output_path)
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "Kandinsky-3"))
        
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_kandinsky3_tab():
    initial_state = state_manager.get_state("kandinsky3") or {}
    with gr.Row():
        kandinsky3_memory_optimization = gr.Radio(
            choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
            label="Memory Optimization",
            value=initial_state.get("memory_optimization", "Low VRAM"),
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            kandinsky3_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            kandinsky3_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                kandinsky3_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                kandinsky3_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
                save_state_button = gr.Button("Save State")
            with gr.Row():
                kandinsky3_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 3.0),
                    step=0.1,
                    interactive=True
                )
                kandinsky3_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 25),
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate image")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    def save_current_state(memory_optimization, width, height, guidance_scale, inference_steps):
        state_dict = {
            "memory_optimization": memory_optimization,
            "width": int(width),
            "height": int(height),
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("kandinsky3") or {}
        state_manager.save_state("kandinsky3", state_dict)
        return memory_optimization, width, height, guidance_scale, inference_steps

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            kandinsky3_memory_optimization, 
            kandinsky3_width_input, 
            kandinsky3_height_input, 
            kandinsky3_guidance_scale_slider, 
            kandinsky3_num_inference_steps_input
        ],
        outputs=[
            kandinsky3_memory_optimization, 
            kandinsky3_width_input, 
            kandinsky3_height_input, 
            kandinsky3_guidance_scale_slider, 
            kandinsky3_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, kandinsky3_prompt_input, kandinsky3_negative_prompt_input, kandinsky3_width_input, 
            kandinsky3_height_input, kandinsky3_guidance_scale_slider, kandinsky3_num_inference_steps_input, 
            kandinsky3_memory_optimization,
        ],
        outputs=[output_gallery]
    )