"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os
import gc
import modules.util.appstate
from PIL import Image
from datetime import datetime
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from huggingface_hub import snapshot_download
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager
from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
from transformers import T5EncoderModel
from controlnet_aux import (
    HEDdetector, 
    LineartDetector, 
    OpenposeDetector,
    DWposeDetector, 
    MidasDetector
)

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flex.2-preview"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, vaetiling, model_type):
    print("----Flex.2-preview mode: ",memory_optimization, vaetiling, model_type)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "Flex2Pipeline" and
        modules.util.appstate.global_inference_type == model_type and 
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Flex.2-preview pipe<<<<")
        return modules.util.appstate.global_pipe, modules.util.appstate.global_text_encoder_2
    else:
        clear_previous_model_memory()

    if model_type=="No quantization":
        model_name_or_path = "newgenai79/Flex2-preview"
    elif model_type=="int4":
        model_name_or_path = "newgenai79/Flex.2-preview-int4"
    elif model_type=="int8":
        model_name_or_path = "newgenai79/Flex.2-preview-int8"
    
    cache_path="models"
    dtype = torch.bfloat16
    snapshot_download(
        repo_id=model_name_or_path,
        local_dir=f"{cache_path}/{model_name_or_path}",
        local_dir_use_symlinks=False
    )
    modules.util.appstate.global_text_encoder_2 = AutoPipelineForText2Image.from_pretrained(
        model_name_or_path,
        transformer=None,
        vae=None,
        torch_dtype=dtype,
        cache_dir=f"{cache_path}/{model_name_or_path}",
    )
    modules.util.appstate.global_pipe = AutoPipelineForText2Image.from_pretrained(
        model_name_or_path,
        custom_pipeline=f"{cache_path}/{model_name_or_path}",
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        torch_dtype=dtype,
        cache_dir=f"{cache_path}/{model_name_or_path}",
    )

    if memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()
    elif memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")

    if vaetiling:
        modules.util.appstate.global_pipe.vae.enable_tiling()
    else:
        modules.util.appstate.global_pipe.vae.disable_tiling()
        
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = model_type
    # modules.util.appstate.global_text_encoder_2 = text_encoder_2
    return modules.util.appstate.global_pipe, modules.util.appstate.global_text_encoder_2

def generate_preview_control_image(source_image_path, control_type):
    if not source_image_path:
        return None
    
    try:
        # Load source image
        source_image = load_image(source_image_path)
        
        # Specify custom cache directory for model files
        cache_dir = os.path.join("models", "controlnet")
        os.makedirs(cache_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize appropriate detector based on control type
        if control_type == "Line":
            detector = LineartDetector.from_pretrained("lllyasviel/Annotators", cache_dir=cache_dir)
            detector.to(device)
        elif control_type == "Depth":
            detector = MidasDetector.from_pretrained("lllyasviel/Annotators", cache_dir=cache_dir)
            detector.to(device)
        elif control_type == "OpenPose":
            detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators", cache_dir=cache_dir)
            detector.to(device)
        elif control_type == "DWPose":
            detector = DWposeDetector()
            detector.to(device)
        else:
            print(f"Unsupported control type: {control_type}")
            return None
        
        # Generate control image
        with torch.no_grad():
            if control_type == "OpenPose":
                control_image = detector(source_image, hand_and_face=True)
            else:
                control_image = detector(source_image)
        
        # Save the control image
        control_dir = os.path.join("output", "control_images")
        os.makedirs(control_dir, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        control_image_path = os.path.join(control_dir, f"control_{control_type}_{timestamp}.png")
        control_image.save(control_image_path)
        del detector
        return control_image_path
    except Exception as e:
        print(f"Error generating control image: {str(e)}")
        return None

def preview_control_image(source_image, control_type):
    if source_image is None:
        return None
    control_image_path = generate_preview_control_image(source_image, control_type)
    return control_image_path
def preview_inpaint_control_image(source_image):
    if source_image is None:
        return None
    # Hardcoded to use Depth as requested
    control_image_path = generate_preview_control_image(source_image, "Depth")
    return control_image_path

def generate_images(
    inference_type, control_input, source_image, 
    control_image, inpaint_source_image, 
    inpaint_control_image, inpaint_mask_image, 
    prompt, width, height, 
    seed, model_type, memory_optimization, vaetiling, guidance_scale, 
    num_inference_steps, control_strength, control_stop,
    no_of_images, randomize_seed
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        # Get pipeline (either cached or newly loaded)
        pipe, text_pipeline = get_pipeline(memory_optimization, vaetiling, model_type)
        # generator = torch.Generator(device="cpu").manual_seed(seed)
        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image (Step {i}/{num_inference_steps})")
            return callback_kwargs
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_filename = "flex2_preview.png"
        gallery_items = []
        text_pipeline.to('cuda')
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
                pipe=text_pipeline,
                prompt=prompt,
            )
        text_pipeline.to('cpu')
        # Prepare inference parameters
        inference_params = {
            # "prompt": prompt,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            # "generator": generator,
            "callback_on_step_end": callback_on_step_end,
        }
        if inference_type=="Control":
            inference_params["control_strength"] = control_strength
            inference_params["control_stop"] = control_stop
            inference_params["control_image"] = load_image(control_image)
        elif inference_type=="Inpaint":
            inference_params["control_strength"] = 0.5
            inference_params["control_stop"] = 0.33
            inference_params["control_image"] = load_image(inpaint_control_image)
            inference_params["inpaint_image"] = load_image(inpaint_source_image)
            inference_params["inpaint_mask"] = load_image(inpaint_mask_image)

        for img_idx in range(no_of_images):
            current_seed = random_seed() if randomize_seed else seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            inference_params["generator"] = generator
            # Generate images
            start_time = datetime.now()
            image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
           
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{base_filename}"
            output_path = os.path.join(OUTPUT_DIR, filename)
            metadata = {
                "model": "Flex.2-preview",
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "memory_optimization": memory_optimization,
                "vae_tiling": vaetiling,
                "timestamp": timestamp,
                "generation_time": generation_time
            }
            # Save the image
            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image {img_idx+1}/{no_of_images} generated: {output_path}")
            # Add to gallery items
            gallery_items.append((output_path, "Flex.2-preview"))
        del prompt_embeds
        del pooled_prompt_embeds
        gc.collect()
        torch.cuda.empty_cache()
        modules.util.appstate.global_inference_in_progress = False        
        return gallery_items

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False
def extract_mask(image_data):
    if image_data is None:
        return None, None

    # Check if layers exist and extract the mask from the drawn layer
    if "layers" in image_data and len(image_data["layers"]) > 0:
        # The mask should be in the first layer (since we're drawing with a white brush)
        layer = image_data["layers"][0]  # First layer contains the drawn mask
        layer_img = Image.fromarray(layer).convert("RGBA")
        layer_np = np.array(layer_img)

        # Extract the alpha channel from the layer
        alpha = layer_np[:, :, 3]

        # Create a binary mask: white (255) where alpha > 0, black (0) elsewhere
        mask = np.zeros_like(alpha, dtype=np.uint8)
        mask[alpha > 0] = 255

    else:
        # Fallback: If no layers, try the composite image
        if "composite" not in image_data:
            return None, None
        composite = image_data["composite"]
        img = Image.fromarray(composite).convert("RGBA")
        rgba_np = np.array(img)
        alpha = rgba_np[:, :, 3]
        mask = np.zeros_like(alpha, dtype=np.uint8)
        mask[alpha > 0] = 255

    # Convert the mask to a PIL image in grayscale mode
    mask_image = Image.fromarray(mask, mode="L")

    mask_dir = os.path.join("output", "mask_images")
    os.makedirs(mask_dir, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    mask_image_path = os.path.join(mask_dir, f"mask_{timestamp}.png")

    # Save the mask image to a local file
    # output_path = "binary_mask.png"
    mask_image.save(mask_image_path)

    return mask_image #, output_path  # Return both the image and the file path

# Define fixed white brush
brush = gr.Brush(
    default_size=20,
    colors=["#ffffff"],
    default_color="#ffffff",
    color_mode="fixed"
)
def create_flex2_preview_tab():
    gr.HTML("""
        <style>
        .small-button { 
            max-width: 2.2em; 
            min-width: 2.2em !important; 
            height: 2.4em; 
            # align-self: end; 
            line-height: 1em; 
            border-radius: 0.5em; 
        }
        .align-right-column {
            display: flex !important;
            justify-content: flex-end !important; /* Align content to the right */
            align-items: center !important; /* Keep vertical centering if needed */
        }
        .align-left-column {
            display: flex !important;
            justify-content: flex-start !important; /* Align content to the left */
            align-items: center !important; /* Keep vertical centering if needed */
        }
        .center-button-container { display: flex !important; align-items: center !important; justify-content: center !important; height: 512px !important; width: 100% !important; }
        </style>
    """, visible=False)
    initial_state = state_manager.get_state("flex2_preview") or {}
    with gr.Row():
        flex2_preview_inference_type = gr.Radio(
            choices=["Text 2 image", "Control", "Inpaint"],
            label="Inference type",
            value=initial_state.get("inference_type", "Text 2 image"),
            interactive=True
        )
    with gr.Row():
        with gr.Accordion("Inference type - Control", open=False) as accordion_control:
            with gr.Row():
                with gr.Column(scale=3):
                    flex2_preview_control_input = gr.Radio(
                        choices=["Line", "Depth", "OpenPose", "DWPose"],
                        label="Control input",
                        value=initial_state.get("control_input", "OpenPose"),
                        interactive=False
                    )
                with gr.Column(scale=2):
                    flex2_preview_control_strength = gr.Slider(
                        label="Control strength", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=initial_state.get("control_strength", 0.5),
                        step=0.1,
                        interactive=True
                    )
                with gr.Column(scale=2):
                    flex2_preview_control_stop = gr.Slider(
                        label="Control stop", 
                        minimum=0.0, 
                        maximum=1.0, 
                        value=initial_state.get("control_stop", 0.33),
                        step=0.01,
                        interactive=True
                    )
            with gr.Row(elem_classes="debug-row"):
                with gr.Column(scale=3, elem_classes=["align-right-column", "debug-column"]):
                    flex2_preview_source_image = gr.Image(label="Source image", type="filepath", width=512, height=512, interactive=False)
                with gr.Column(scale=1, elem_classes=["debug-column"]):
                    preview_button_control = gr.Button("👁️", elem_classes="small-button", interactive=False)
                with gr.Column(scale=3, elem_classes=["align-left-column", "debug-column"]):
                    flex2_preview_control_image = gr.Image(label="Control image", type="filepath", width=512, height=512, interactive=False)

    with gr.Row():
        with gr.Accordion("Inference type - Inpaint", open=False) as accordion_inpaint:
            with gr.Row(elem_classes="debug-row"):
                with gr.Column(scale=3, elem_classes=["align-right-column", "debug-column"]):
                    flex2_preview_inpaint_source_image = gr.Image(label="Source image", type="filepath", width=512, height=512, interactive=False)
                with gr.Column(scale=1, elem_classes=["debug-column"]):
                    preview_button_inpaint = gr.Button("👁️", elem_classes="small-button", interactive=False)
                with gr.Column(scale=3, elem_classes=["align-left-column", "debug-column"]):
                    flex2_preview_inpaint_control_image = gr.Image(label="Control image - Depth", type="filepath", width=512, height=512, interactive=False)
            with gr.Row():
                with gr.Column():
                    flex2_preview_inpaint_mask_editor = gr.ImageEditor(image_mode="RGBA", type="numpy", brush=brush, sources="upload", width=512, height=512, interactive=True)
                with gr.Column(scale=1, elem_classes=["center-button-container"]):
                    preview_button_inpaint_mask = gr.Button("👁️", elem_classes="small-button", interactive=False)
                with gr.Column():
                    flex2_preview_inpaint_mask_image = gr.Image(label="Inpaint mask image", type="filepath", width=512, height=512, interactive=False)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                flex2_preview_prompt = gr.Textbox(
                    label="Prompt", 
                    lines=7,
                    interactive=True
                )
            with gr.Row():
                flex2_preview_width = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                flex2_preview_height = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
                seed = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("🔄", elem_classes="small-button")
            with gr.Row():
                flex2_preview_no_of_images = gr.Number(
                    label="Number of Images", 
                    value=1,
                    interactive=True
                )
                flex2_preview_randomize_seed = gr.Checkbox(label="Randomize seed", value=False, interactive=True)

        with gr.Column():
            with gr.Row():
                flex2_preview_model_type = gr.Radio(
                    choices=["No quantization", "int8", "int4"],
                    label="Model type",
                    value=initial_state.get("model_type", "int4"),
                    interactive=True
                )
            with gr.Row():
                flex2_preview_memory_optimization = gr.Radio(
                    choices=["No optimization", "Low VRAM", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    info="Extremely Low VRAM is not supported for int4",
                    value=initial_state.get("memory_optimization", "Low VRAM"),
                    interactive=True
                )
            with gr.Row():
                flex2_preview_vaetiling = gr.Checkbox(label="VAE Tiling", value=initial_state.get("vaetiling", True), interactive=True)
            with gr.Row():
                flex2_preview_guidance_scale = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 3.5),
                    step=0.1,
                    interactive=True
                )
                flex2_preview_num_inference_steps = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 30),
                    interactive=True
                )
    with gr.Row():
        generate_button = gr.Button("Generate image")
        save_state_button = gr.Button("Save State")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )

    def save_current_state(memory_optimization, vaetiling, width, height, guidance_scale, inference_steps):
        state_dict = {
            "memory_optimization": memory_optimization,
            "vaetiling": vaetiling,
            "width": int(width),
            "height": int(height),
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("flex2_preview") or {}
        state_manager.save_state("flex2_preview", state_dict)
        return memory_optimization, vaetiling, width, height, guidance_scale, inference_steps

    # Event handlers
    
    def toggle_accordion(model):
        # "Text 2 image", "Control", "Inpaint"
        return gr.update(open=(model == "Control")), gr.update(interactive=(model == "Control")), gr.update(interactive=(model == "Control")), gr.update(interactive=(model == "Control")), gr.update(interactive=(model == "Control")),    gr.update(open=(model == "Inpaint")), gr.update(interactive=(model == "Inpaint")), gr.update(interactive=(model == "Inpaint")), gr.update(interactive=(model == "Inpaint")), gr.update(interactive=(model == "Inpaint")), gr.update(interactive=(model == "Inpaint"))
    flex2_preview_inference_type.change(toggle_accordion, inputs=[flex2_preview_inference_type], outputs=[accordion_control, flex2_preview_control_input, flex2_preview_source_image, flex2_preview_control_image, preview_button_control, accordion_inpaint, flex2_preview_inpaint_source_image, preview_button_inpaint, flex2_preview_inpaint_control_image, flex2_preview_inpaint_mask_image, preview_button_inpaint_mask]) # 
    preview_button_inpaint_mask.click(
        extract_mask,
        inputs=flex2_preview_inpaint_mask_editor,
        outputs=[flex2_preview_inpaint_mask_image],
    )
    preview_button_control.click(
        fn=preview_control_image,
        inputs=[flex2_preview_source_image, flex2_preview_control_input],
        outputs=[flex2_preview_control_image]
    )
    preview_button_inpaint.click(
        fn=preview_inpaint_control_image,
        inputs=[flex2_preview_inpaint_source_image],
        outputs=[flex2_preview_inpaint_control_image]
    )
    random_button.click(fn=random_seed, outputs=[seed])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flex2_preview_memory_optimization, 
            flex2_preview_vaetiling, 
            flex2_preview_width, 
            flex2_preview_height, 
            flex2_preview_guidance_scale, 
            flex2_preview_num_inference_steps
        ],
        outputs=[
            flex2_preview_memory_optimization, 
            flex2_preview_vaetiling, 
            flex2_preview_width, 
            flex2_preview_height, 
            flex2_preview_guidance_scale, 
            flex2_preview_num_inference_steps
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            flex2_preview_inference_type, flex2_preview_control_input, 
            flex2_preview_source_image, flex2_preview_control_image, 
            flex2_preview_inpaint_source_image, flex2_preview_inpaint_control_image, 
            flex2_preview_inpaint_mask_image, 
            flex2_preview_prompt, 
            flex2_preview_width, flex2_preview_height, seed, 
            flex2_preview_model_type, flex2_preview_memory_optimization, 
            flex2_preview_vaetiling, flex2_preview_guidance_scale, 
            flex2_preview_num_inference_steps, flex2_preview_control_strength, 
            flex2_preview_control_stop, flex2_preview_no_of_images, flex2_preview_randomize_seed
        ],
        outputs=[output_gallery]
    )