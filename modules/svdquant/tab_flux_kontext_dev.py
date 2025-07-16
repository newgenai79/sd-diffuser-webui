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
from types import MethodType
from datetime import datetime
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from diffusers import FluxKontextPipeline, FluxKontextInpaintPipeline
from diffusers.utils import load_image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager
from nunchaku.utils import get_precision
from PIL import Image

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"
lora_path = "models/lora/flux_Kontext/"
RESOLUTIONS_kontext = [
    "672x1568",
    "688x1504",
    "720x1456",
    "752x1392",
    "800x1328",
    "832x1248",
    "880x1184",
    "944x1104",
    "1024x1024",
    "1104x944",
    "1184x880",
    "1248x832",
    "1328x800",
    "1392x752",
    "1456x720",
    "1504x688",
    "1568x672",
]
def get_lora_files():
    lora_files = []
    if os.path.exists(lora_path):
        for file in os.listdir(lora_path):
            if file.endswith(".safetensors"):
                # Remove .safetensors extension for display
                lora_name = os.path.splitext(file)[0]
                lora_files.append(lora_name)
    return lora_files

def refresh_lora_list():
    lora_files = get_lora_files()
    return gr.update(choices=lora_files)

def parse_lora_input(lora_input):
    if not lora_input:
        return []
    import re
    lora_matches = re.findall(r"<([^:>]+):([\d.]+)>", lora_input)
    return [(os.path.join(lora_path, f"{lora_name}.safetensors"), float(weight)) for lora_name, weight in lora_matches]

def compare_lora(selected_loras, global_selected_lora):
    if selected_loras is None and global_selected_lora is None:
        return True
    if selected_loras is None or global_selected_lora is None:
        return False
    if len(selected_loras) != len(global_selected_lora):
        return False
    # Convert lists to sets of tuples for order-independent comparison
    set1 = {(os.path.normpath(path), weight) for path, weight in selected_loras}
    set2 = {(os.path.normpath(path), weight) for path, weight in global_selected_lora}
    return set1 == set2

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, performance_optimization, inference_type, use_qencoder):
    print("----FluxKontextPipeline mode: ", memory_optimization, performance_optimization, inference_type, use_qencoder)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ in ["FluxKontextPipeline", "FluxKontextInpaintPipeline"] and 
            # modules.util.appstate.global_performance_optimization == performance_optimization and
            (( inference_type == "TI2I" and modules.util.appstate.global_inference_type == "TI2I" ) or ( inference_type in ["TIM2I", "TIMR2I"] and modules.util.appstate.global_inference_type in ["TIM2I", "TIMR2I"])) and 
            modules.util.appstate.global_use_qencoder == use_qencoder and 
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing FluxKontextPipeline pipe<<<<")
                return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
        torch.cuda.synchronize()

    dtype = torch.bfloat16
    precision = get_precision()
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors", 
        offload=True
    )
    pipeline_init_kwargs = {}
    if use_qencoder:
        text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
            "mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
        )
        pipeline_init_kwargs["text_encoder_2"] = text_encoder_2

    if inference_type == "TI2I":
        className = FluxKontextPipeline
    else:
        className = FluxKontextInpaintPipeline

    modules.util.appstate.global_pipe = className.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16, 
        **pipeline_init_kwargs
    )
    if performance_optimization == "nunchaku-fp16":
        modules.util.appstate.global_pipe.transformer.set_attention_impl("nunchaku-fp16")
    else:
        modules.util.appstate.global_pipe.transformer.set_attention_impl("flashattn2")

    if memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
        
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_performance_optimization = performance_optimization
    modules.util.appstate.global_inference_type = inference_type
    modules.util.appstate.global_use_qencoder = use_qencoder
    return modules.util.appstate.global_pipe

def stitch_images_horizontally(img1: Image.Image, img2: Image.Image) -> Image.Image:
    # assume both images are the same height and width
    width, height = img1.size

    # create a new blank image with width doubled
    new_img = Image.new("RGB", (width * 2, height))

    # paste first image at (0,0)
    new_img.paste(img1, (0, 0))

    # paste second image at (width, 0)
    new_img.paste(img2, (width, 0))

    return new_img

def generate_images(
    seed, guidance_scale, num_inference_steps, 
    memory_optimization, no_of_images, randomize_seed, input_image,
    performance_optimization, prompt, mask_input_image, 
    reference_image, use_qencoder, lora_input , width, height # , input_image2
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    if mask_input_image is not None and reference_image is not None:
        inference_type = "TIMR2I"
    elif mask_input_image is not None and reference_image is None:
        inference_type = "TIM2I"
    else:
        inference_type = "TI2I"
        
    gallery_items = []

    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, performance_optimization, inference_type, use_qencoder)

        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx+1}: (Step {i}/{num_inference_steps})")
            return callback_kwargs
        modules.util.appstate.global_inference_in_progress = True
        
        input_image_final = input_image
        """
        if input_image2 is not None and inference_type == "TI2I":
            input_image_final = None
            input_image_final = stitch_images_horizontally(input_image, input_image2)
        """
        # Apply LoRA weights if specified
        selected_loras = parse_lora_input(lora_input)
        if selected_loras:  # Apply new LoRA weights
            if modules.util.appstate.global_selected_lora and compare_lora(selected_loras, modules.util.appstate.global_selected_lora):
                # Reuse existing LoRA configuration
                print("Reuse lora")
                pass
            else:
                if modules.util.appstate.global_selected_lora is not None:
                    print("Reset lora")
                    pipe.transformer.reset_lora()
                from nunchaku.lora.flux.compose import compose_lora
                print("New lora", selected_loras)
                composed_lora = compose_lora(selected_loras)
                pipe.transformer.update_lora_params(composed_lora)
                modules.util.appstate.global_selected_lora = selected_loras
        else:  # No LoRA weights specified
            if modules.util.appstate.global_selected_lora is not None:
                print("Reset lora")
                pipe.transformer.reset_lora()
                modules.util.appstate.global_selected_lora = None
            
        image_input = load_image(input_image_final).convert("RGB")
        # Generate multiple images in a loop
        for img_idx in range(no_of_images):
           
            # If randomize_seed is True, generate a new random seed for each image
            current_seed = random_seed() if randomize_seed else seed
            
            # Create generator with the current seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            start_time = datetime.now()
            inference_params = {
                "prompt": prompt,
                "image": image_input,
                "guidance_scale": float(guidance_scale), 
                "width": width, 
                "height": height,
                "num_inference_steps": num_inference_steps,
                "callback_on_step_end": callback_on_step_end
            }
            if inference_type in ["TIMR2I", "TIM2I"]:
                inference_params["strength"] = 1.0
                if inference_type == "TIM2I":
                    mask_image = load_image(mask_input_image)
                    inference_params["mask_image"] = mask_image
                elif inference_type == "TIMR2I":
                    mask_image = load_image(mask_input_image)
                    image_reference = load_image(reference_image)
                    mask = pipe.mask_processor.blur(mask_image, blur_factor=12)
                    inference_params["mask_image"] = mask
                    inference_params["image_reference"] = image_reference
            image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Create filename with index
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_flux_kontext_{img_idx+1}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            metadata = {
                "model": "FLUX.1-dev-Kontext",
                "inference_type": inference_type,
                "prompt": prompt,
                "seed": current_seed,
                "guidance_scale": f"{float(guidance_scale):.2f}",
                "width": width, 
                "height": height, 
                "num_inference_steps": num_inference_steps,
                "memory_optimization": memory_optimization,
                "performance_optimization": performance_optimization,
                "timestamp": timestamp,
                "generation_time": generation_time,
                "lora_input": lora_input,
            }
            # Save the image
            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image {img_idx+1}/{no_of_images} generated: {output_path}")
            
            # Add to gallery items
            gallery_items.append((output_path, "FLUX.1-dev-Kontext"))
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

    return mask_image

brush = gr.Brush(
    default_size=20,
    colors=["#ffffff"],
    default_color="#ffffff",
    color_mode="fixed"
)

def update_lora_input(selected_loras, current_lora_input):
    import re
    # Parse current lora_input to maintain user-modified weights
    current_weights = {}
    if current_lora_input:
        matches = re.findall(r"<([^:>]+):([\d.]+)>", current_lora_input)
        for lora_name, weight in matches:
            current_weights[lora_name] = weight
    
    # Build new lora_input string
    lora_strings = []
    for lora in selected_loras:
        weight = current_weights.get(lora, "1.0")  # Use existing weight or default to 1.0
        lora_strings.append(f"<{lora}:{weight}>")
    
    return "".join(lora_strings)

def create_flux_kontext_tab():
    gr.HTML("<style>.small-button { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; line-height: 1em; border-radius: 0.5em; margin: 0 auto !important; } .center-button-container { display: flex !important; align-items: center !important; justify-content: center !important; height: 512px !important; width: 100% !important; }</style>", visible=False)

    initial_state = state_manager.get_state("flux-kontext") or {}
    selected_perf_opt = initial_state.get("performance_optimization", "no_optimization")
    with gr.Accordion("Select model / Optimization", open=True):
        with gr.Row():
            with gr.Column(scale=2):
                flux_memory_optimization = gr.Radio(
                    choices=["No optimization", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    value=initial_state.get("memory_optimization", "Extremely Low VRAM"),
                    interactive=True
                )
            with gr.Column(scale=2):
                flux_performance_optimization = gr.Radio(
                    choices=["No optimization", "nunchaku-fp16"],
                    label="Performance Optimization",
                    value=initial_state.get("performance_optimization", "No optimization"),
                    interactive=True
                )
            with gr.Column(scale=1):
                flux_use_qencoder = gr.Checkbox(label="Use qencoder (quantized text_encoder_2)", value=initial_state.get("use_qencoder", False), interactive=True)        
    with gr.Row():
        with gr.Column():
            flux_input_image = gr.Image(label="Input Image (only square resolution supported e.g. 1024 x 1024)", type="pil", width=512, height=512)
            # flux_input_image2 = gr.Image(label="Input Image 2 (optional)", type="pil", width=256, height=256)
        with gr.Column():
            with gr.Row():
                flux_prompt_input = gr.Textbox(
                    label="Prompt", 
                    lines=4,
                    value="Make Pikachu hold a sign that says 'Nunchaku is awesome', yarn art style, detailed, vibrant colors",
                    interactive=True
                )
            with gr.Row():
                flux_resolution_dropdown = gr.Dropdown(
                    choices=RESOLUTIONS_kontext,
                    label="Supported resolution(s) (width x height)",
                    info="Please note this is only for information purpose. Make sure to crop/resize the image and use here.",
                    interactive=True
                )
            with gr.Row():
                flux_width = gr.Slider(
                    label="Width", 
                    minimum=256, 
                    maximum=1536, 
                    value=initial_state.get("width", 1024),
                    step=16,
                    interactive=True
                )
                flux_height = gr.Slider(
                    label="Height", 
                    minimum=256, 
                    maximum=1536, 
                    value=initial_state.get("height", 1024),
                    step=16,
                    interactive=True
                )

            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("♻️", elem_classes="small-button")
                flux_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 28),
                    interactive=True
                )

            with gr.Row():
                flux_guidance_scale = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=50.0, 
                    value=initial_state.get("guidance_scale", 2.5),
                    step=0.1,
                    interactive=True
                )
            with gr.Row():
                flux_no_of_images_input = gr.Number(
                    label="Number of Images", 
                    value=1,
                    interactive=True
                )
                flux_randomize_seed = gr.Checkbox(label="Randomize seed", value=False, interactive=True)
    with gr.Accordion("Select Kontext lora (models/lora/flux_Kontext)", open=False):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    flux_lora_select = gr.CheckboxGroup(get_lora_files(), label="Select lora(s)", info="Select to enable, deselect to disable.")
                    refresh_button = gr.Button("♻️", elem_classes="small-button")
            with gr.Column():
                flux_lora_input = gr.Textbox(
                    label="Selected lora", 
                    lines=4,
                    interactive=True
                )
    with gr.Accordion("Image mask", open=False):
        with gr.Row():
            with gr.Column(scale=3):
                flux_preview_inpaint_mask_editor = gr.ImageEditor(image_mode="RGBA", type="numpy", brush=brush, sources="upload", width=512, height=512, interactive=True)
            with gr.Column(scale=1, elem_classes=["center-button-container"]):
                preview_button_inpaint_mask = gr.Button("👁️", elem_classes="small-button", interactive=True)
            with gr.Column(scale=3):
                flux_preview_inpaint_mask_image = gr.Image(label="Inpaint mask image", type="filepath", width=512, height=512, interactive=True)
            with gr.Column(scale=3):
                flux_preview_inpaint_reference_image = gr.Image(label="Reference image", type="filepath", width=512, height=512, interactive=True)

    with gr.Row():
        generate_button = gr.Button("🎨 Generate image")
        save_state_button = gr.Button("💾 Save State")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )
    preview_button_inpaint_mask.click(
        extract_mask,
        inputs=flux_preview_inpaint_mask_editor,
        outputs=[flux_preview_inpaint_mask_image],
    )
    def update_dimensions_from_image(image):
        if image is None:
            return gr.update(), gr.update()
        
        width, height = image.size
        
        # Check if dimensions are divisible by 16
        if width % 16 == 0 and height % 16 == 0:
            return gr.update(value=width), gr.update(value=height)
        else:
            # Don't update if not divisible by 16
            return gr.update(), gr.update()

    # Event handler (add this after your other event handlers)

    flux_input_image.change(
        fn=update_dimensions_from_image,
        inputs=[flux_input_image],
        outputs=[flux_width, flux_height]
    )

    def save_current_state(memory_optimization, guidance_scale, inference_steps, 
            performance_optimization, use_qencoder): # , width, height

        state_dict = {
            "memory_optimization": memory_optimization,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "performance_optimization": performance_optimization,
            "use_qencoder": use_qencoder,
            # "width": width,
            # "height": height,
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("flux-kontext") or {}
        state_manager.save_state("flux-kontext", state_dict)
        return memory_optimization, guidance_scale, inference_steps, performance_optimization, use_qencoder # , width, height
    
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    refresh_button.click(
        fn=lambda: (refresh_lora_list(), "", []),
        outputs=[flux_lora_select, flux_lora_input, flux_lora_select]
    )
    flux_lora_select.change(
        fn=update_lora_input,
        inputs=[flux_lora_select, flux_lora_input],
        outputs=[flux_lora_input]
    )
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flux_memory_optimization, 
            flux_guidance_scale, 
            flux_num_inference_steps_input,
            flux_performance_optimization,
            flux_use_qencoder,
            # flux_width, 
            # flux_height,
        ],
        outputs=[
            flux_memory_optimization, 
            flux_guidance_scale, 
            flux_num_inference_steps_input,
            flux_performance_optimization,
            flux_use_qencoder,
            # flux_width, 
            # flux_height,
        ]
    )

    generate_button.click(
        fn=generate_images,
        inputs=[
            seed_input, flux_guidance_scale, 
            flux_num_inference_steps_input, flux_memory_optimization, 
            flux_no_of_images_input, flux_randomize_seed, flux_input_image,
            flux_performance_optimization, flux_prompt_input, 
            flux_preview_inpaint_mask_image, 
            flux_preview_inpaint_reference_image, flux_use_qencoder,
            flux_lora_input , flux_width, flux_height # , flux_input_image2
        ],
        outputs=[output_gallery]
    )