"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import os
import gradio as gr
import gc
import torch
import pathlib
import re
import glob
from PIL import Image
from image_gen_aux import BEN2BackgroundRemover
from image_gen_aux.utils import load_image
from datetime import datetime
from loadimg import load_img
import modules.util.appstate
from datetime import datetime
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

OUTPUT_DIR = "output/ben2/"
os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%d%m%Y%H%M%S")

def get_image_info(image):
    if image is None:
        return "No image uploaded"
    try:
        img = load_img(image, output_type="pil")
        return f"{img.size[0]} x {img.size[1]}"
    except Exception:
        return "Error retrieving dimensions"
def rgba_to_hex(rgba_str):
    """Convert rgba(r, g, b, a) string to #RRGGBBAA hex format, handling integer or float RGB values."""
    try:
        # Updated regex to match integers or floating-point numbers for RGB, and 0-1 for alpha
        match = re.match(r'rgba\((\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*([0-1](\.\d+)?)\)', rgba_str)
        if not match:
            raise ValueError(f"Invalid rgba format: {rgba_str}")
        # Round RGB values to nearest integer, parse alpha as float
        r, g, b, a = round(float(match.group(1))), round(float(match.group(2))), round(float(match.group(3))), float(match.group(4))
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255 and 0 <= a <= 1):
            raise ValueError(f"RGBA values out of range: {rgba_str}")
        a = int(a * 255)
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}".upper()
    except Exception as e:
        raise ValueError(f"Failed to convert rgba to hex: {str(e)}")

def get_pipeline():
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "BEN2BackgroundRemover"):
        print(">>>>Reusing Ben2 pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()

    modules.util.appstate.global_pipe = BEN2BackgroundRemover.from_pretrained("PramaLLC/BEN2")
    modules.util.appstate.global_pipe.to("cuda")
    return modules.util.appstate.global_pipe

def process(image, save_flat, bg_colour, extract_mask):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    if isinstance(image, str):
        input_filename = pathlib.Path(image).stem + f"_{get_timestamp()}"
    else:
        input_filename = get_timestamp()
    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline()
        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Removing background (Step {i}/{num_inference_steps})")
            return callback_kwargs

        input_image = load_image(image)
        output_filename = f"{OUTPUT_DIR}{input_filename}.png"
        if extract_mask:
            foreground, mask = pipe(input_image, return_mask=True)
            im = load_img(foreground[0], output_type="pil", input_type="auto").convert("RGBA")
            image_size = im.size
            result_image = im
            if save_flat:
                if bg_colour.startswith('rgba'):
                    bg_colour = rgba_to_hex(bg_colour)
                if not (bg_colour.startswith('#') and len(bg_colour) == 9):
                    raise ValueError(f"Invalid background color format: {bg_colour}. Expected #RRGGBBAA (e.g., #FFFFFFFF)")
                try:
                    colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
                    background = Image.new("RGBA", image_size, colour_rgb)
                    result_image = Image.alpha_composite(background, im).convert("RGB")
                except ValueError as e:
                    raise ValueError(f"Failed to parse background color {bg_colour}: {str(e)}")
            result_image.save(output_filename)
            mask[0].save(f"{OUTPUT_DIR}{input_filename}_mask.png")
        else:
            foreground = pipe(input_image)[0]
            im = load_img(foreground, output_type="pil", input_type="auto").convert("RGBA")
            image_size = im.size
            result_image = im
            if save_flat:
                if bg_colour.startswith('rgba'):
                    bg_colour = rgba_to_hex(bg_colour)
                if not (bg_colour.startswith('#') and len(bg_colour) == 9):
                    raise ValueError(f"Invalid background color format: {bg_colour}. Expected #RRGGBBAA (e.g., #FFFFFFFF)")
                try:
                    colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
                    background = Image.new("RGBA", image_size, colour_rgb)
                    result_image = Image.alpha_composite(background, im).convert("RGB")
                except ValueError as e:
                    raise ValueError(f"Failed to parse background color {bg_colour}: {str(e)}")
            result_image.save(output_filename)
        del im
        modules.util.appstate.global_inference_in_progress = False
        if extract_mask:
            return result_image, output_filename, f"{image_size[0]} x {image_size[1]}", mask[0]
        else:
            return result_image, output_filename, f"{image_size[0]} x {image_size[1]}", None
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None, None, None, None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def batch_process(input_dir, save_flat, bg_colour, output_dir_save, extract_mask):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        os.makedirs(output_dir_save, exist_ok=True)
        image_extensions = ['.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.webp', '.avif']
        input_images = []
        for ext in image_extensions:
            input_images.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        
        if not input_images:
            raise ValueError(f"No images found in {input_dir}")

        processed_files = []  # List to store file paths for Gradio
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline()
        progress_bar = gr.Progress(track_tqdm=True)

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Removing background (Step {i}/{num_inference_steps})")
            return callback_kwargs

        for i, image_path in enumerate(input_images):
            # print(f"Ben2 [batch]: image {i+1}/{len(input_images)}", end='\r', flush=True)
            try:
                input_image = load_image(image_path)
                input_filename = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = os.path.join(output_dir_save, f"{input_filename}.png")

                if extract_mask:
                    foreground, mask = pipe(input_image, return_mask=True)
                    im = load_img(foreground[0], output_type="pil", input_type="auto").convert("RGBA")
                    image_size = im.size
                    result_image = im
                    if save_flat:
                        if bg_colour.startswith('rgba'):
                            bg_colour = rgba_to_hex(bg_colour)
                        if not (bg_colour.startswith('#') and len(bg_colour) == 9):
                            raise ValueError(f"Invalid background color format: {bg_colour}. Expected #RRGGBBAA (e.g., #FFFFFFFF)")
                        try:
                            colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
                            background = Image.new("RGBA", image_size, colour_rgb)
                            result_image = Image.alpha_composite(background, im).convert("RGB")
                        except ValueError as e:
                            raise ValueError(f"Failed to parse background color {bg_colour}: {str(e)}")
                    result_image.save(output_filename)
                    mask_filename = os.path.join(output_dir_save, f"{input_filename}_mask.png")
                    mask[0].save(mask_filename)
                    processed_files.append(output_filename)
                    processed_files.append(mask_filename)
                else:
                    foreground = pipe(input_image)[0]
                    im = load_img(foreground, output_type="pil", input_type="auto").convert("RGBA")
                    image_size = im.size
                    result_image = im
                    if save_flat:
                        if bg_colour.startswith('rgba'):
                            bg_colour = rgba_to_hex(bg_colour)
                        if not (bg_colour.startswith('#') and len(bg_colour) == 9):
                            raise ValueError(f"Invalid background color format: {bg_colour}. Expected #RRGGBBAA (e.g., #FFFFFFFF)")
                        try:
                            colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
                            background = Image.new("RGBA", image_size, colour_rgb)
                            result_image = Image.alpha_composite(background, im).convert("RGB")
                        except ValueError as e:
                            raise ValueError(f"Failed to parse background color {bg_colour}: {str(e)}")
                    result_image.save(output_filename)
                    processed_files.append(output_filename)
                
                del im
                # print(f"Image generated: {output_filename}")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        return processed_files
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False


def create_ben2_tab():
    initial_state = state_manager.get_state("ben2") or {}
    gr.Markdown("# BiRefNet for Background Removal")

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Upload an image", height=584, type="filepath")
                image_info = gr.Textbox(label="Image Dimensions", interactive=False)
                image_extract_mask = gr.Checkbox(label="Extract mask", value=False)
                with gr.Row():
                    image_save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                    image_bg_colour = gr.ColorPicker(label="Background color for flat images", value="#FFFFFFFF")
                go_image = gr.Button("Remove Background")
            with gr.Column():
                result1 = gr.Image(label="Ben2 Output", height=512)
                result1_info = gr.Textbox(label="Output Dimensions", interactive=False)
                result1_file = gr.Textbox(label="Output File", interactive=False)
                result1_mask = gr.Image(label="Ben2 Mask Output", height=512)

    with gr.Tab("URL"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="URL or local path to image", max_lines=1)
                url_extract_mask = gr.Checkbox(label="Extract mask", value=False)
                with gr.Row():
                    url_save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                    url_bg_colour = gr.ColorPicker(label="Background color for flat images", value="#FFFFFFFF")
                go_text = gr.Button("Remove Background")
            with gr.Column():
                result2 = gr.Image(label="BiRefNet Output", type="pil", height=512)
                result2_info = gr.Textbox(label="Output Dimensions", interactive=False)
                result2_file = gr.Textbox(label="Output File", interactive=False)
                result2_mask = gr.Image(label="Ben2 Mask Output", height=512)
    with gr.Tab("Batch"):
        with gr.Row():
            with gr.Column():
                input_dir = gr.Textbox(label="Input folder path", max_lines=1)
                output_dir_save = gr.Textbox(label="Output folder path", max_lines=1, value=f"{OUTPUT_DIR}")
                batch_extract_mask = gr.Checkbox(label="Extract mask", value=False)
                with gr.Row():
                    batch_save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                    batch_bg_colour = gr.ColorPicker(label="Background color for flat images", value="#FFFFFFFF")
                go_batch = gr.Button("Remove Background(s)")
            with gr.Column():
                result3 = gr.File(label="Processed image(s)", file_count="multiple")

    # Event handlers
    image.change(fn=get_image_info, inputs=image, outputs=image_info)

    go_image.click(
        fn=process, inputs=[image, image_save_flat, image_bg_colour, image_extract_mask], outputs=[result1, result1_file, result1_info, result1_mask]
    )
    go_text.click(
        fn=process, inputs=[text, url_save_flat, url_bg_colour,url_extract_mask], outputs=[result2, result2_file, result2_info, result2_mask]
    )
    go_batch.click(
        fn=batch_process, inputs=[input_dir, batch_save_flat, batch_bg_colour, output_dir_save, batch_extract_mask], outputs=result3
    )
