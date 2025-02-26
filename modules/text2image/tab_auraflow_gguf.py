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
from diffusers import AuraFlowPipeline, AuraFlowTransformer2DModel
from diffusers import GGUFQuantizationConfig
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/auraflow"
gguf_list = [
    "aura_flow_0.3-F16.gguf - 13.7 GB",
    "aura_flow_0.3-Q2_K.gguf - 2.4 GB",
    "aura_flow_0.3-Q3_K_M.gguf - 3.13 GB",
    "aura_flow_0.3-Q3_K_S.gguf - 3.05 GB",
    "aura_flow_0.3-Q4_0.gguf - 3.95 GB",
    "aura_flow_0.3-Q4_1.gguf - 4.37 GB",
    "aura_flow_0.3-Q4_K_M.gguf - 4.05 GB",
    "aura_flow_0.3-Q4_K_S.gguf - 3.95 GB",
    "aura_flow_0.3-Q5_0.gguf - 4.8 GB",
    "aura_flow_0.3-Q5_1.gguf - 5.22 GB",
    "aura_flow_0.3-Q5_K_M.gguf - 4.85 GB",
    "aura_flow_0.3-Q5_K_S.gguf - 4.8 GB",
    "aura_flow_0.3-Q6_K.gguf - 5.7 GB",
    "aura_flow_0.3-Q8_0.gguf - 7.35 GB"
]
def get_gguf(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size
def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, gguf_file, vaeslicing, vaetiling, inference_type):
    print("----auraflow mode: ", memory_optimization, gguf_file, vaeslicing, vaetiling, inference_type)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "AuraFlowPipeline" and
        modules.util.appstate.global_selected_gguf == gguf_file and
        modules.util.appstate.global_inference_type == inference_type and 
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing auraflow pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    transformer_path = f"https://huggingface.co/city96/AuraFlow-v0.3-gguf/blob/main/{gguf_file}"
    transformer = AuraFlowTransformer2DModel.from_single_file(
        transformer_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    modules.util.appstate.global_pipe = AuraFlowPipeline.from_pretrained(
        "fal/AuraFlow-v0.3",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
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
    
    return modules.util.appstate.global_pipe
def get_gguf(gguf_user_selection):
    gguf_file, gguf_file_size_str = gguf_user_selection.split(' - ')
    gguf_file_size = float(gguf_file_size_str.replace(' GB', ''))
    return gguf_file, gguf_file_size

def generate_images(
    seed, prompt, negative_prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, vaeslicing, vaetiling, gguf_file
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        gguf_file, gguf_file_size = get_gguf(gguf_file)
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, gguf_file, vaeslicing, vaetiling, "auraflow_gguf")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        """
        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs
        """
        # Prepare inference parameters
        inference_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        start_time = datetime.now()
        # Generate images
        image = pipe(**inference_params).images[0]
        end_time = datetime.now()
        generation_time = (end_time - start_time).total_seconds()
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        base_filename = "auraflow_gguf.png"

        gallery_items = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{base_filename}"
        output_path = os.path.join(OUTPUT_DIR, filename)
        metadata = {
            "model": "AuraFlow",
            "gguf_file": gguf_file,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "memory_optimization": memory_optimization,
            "vae_slicing": vaeslicing,
            "vae_tiling": vaetiling,
            "timestamp": timestamp,
            "generation_time": generation_time
        }
        # Save the image
        image.save(output_path)
        modules.util.utilities.save_metadata_to_file(output_path, metadata)
        print(f"Image generated: {output_path}")
        modules.util.appstate.global_inference_in_progress = False
        # Add to gallery items
        gallery_items.append((output_path, "AuraFlow"))
    
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_auraflow_gguf_tab():
    initial_state = state_manager.get_state("auraflow_gguf") or {}
    with gr.Row():
        with gr.Column():
            auraflow_gguf_memory_optimization = gr.Radio(
                choices=["No optimization", "Low VRAM"],
                label="Memory Optimization",
                value=initial_state.get("memory_optimization", "Low VRAM"),
                interactive=True
            )
        with gr.Column():
            auraflow_gguf_vaeslicing = gr.Checkbox(label="VAE Slicing", value=initial_state.get("vaeslicing", True), interactive=True)
            auraflow_gguf_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
        with gr.Column():
            auraflow_gguf_dropdown = gr.Dropdown(
                choices=gguf_list,
                value=initial_state.get("gguf", "aura_flow_0.3-Q6_K.gguf - 5.7 GB"),
                label="Select GGUF"
            )
    with gr.Row():
        with gr.Column():
            auraflow_gguf_prompt_input = gr.Textbox(
                label="Prompt", 
                lines=3,
                interactive=True
            )
            auraflow_gguf_negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                lines=3,
                interactive=True
            )
        with gr.Column():
            with gr.Row():
                auraflow_gguf_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                auraflow_gguf_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("Randomize Seed")
            with gr.Row():
                auraflow_gguf_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 3.5),
                    step=0.1,
                    interactive=True
                )
                auraflow_gguf_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
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

    def save_current_state(memory_optimization, gguf, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps):
        state_dict = {
            "memory_optimization": memory_optimization,
            "gguf": gguf,
            "vaeslicing": vaeslicing,
            "vaetiling": vaetiling,
            "width": int(width),
            "height": int(height),
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("auraflow_gguf") or {}
        state_manager.save_state("auraflow_gguf", state_dict)
        return memory_optimization, gguf, vaeslicing, vaetiling, width, height, guidance_scale, inference_steps

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            auraflow_gguf_memory_optimization, 
            auraflow_gguf_dropdown, 
            auraflow_gguf_vaeslicing, 
            auraflow_gguf_vaetiling, 
            auraflow_gguf_width_input, 
            auraflow_gguf_height_input, 
            auraflow_gguf_guidance_scale_slider, 
            auraflow_gguf_num_inference_steps_input
        ],
        outputs=[
            auraflow_gguf_memory_optimization, 
            auraflow_gguf_dropdown, 
            auraflow_gguf_vaeslicing, 
            auraflow_gguf_vaetiling, 
            auraflow_gguf_width_input, 
            auraflow_gguf_height_input, 
            auraflow_gguf_guidance_scale_slider, 
            auraflow_gguf_num_inference_steps_input
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, auraflow_gguf_prompt_input, auraflow_gguf_negative_prompt_input, auraflow_gguf_width_input, 
            auraflow_gguf_height_input, auraflow_gguf_guidance_scale_slider, auraflow_gguf_num_inference_steps_input, 
            auraflow_gguf_memory_optimization, auraflow_gguf_vaeslicing, auraflow_gguf_vaetiling, auraflow_gguf_dropdown
        ],
        outputs=[output_gallery]
    )
