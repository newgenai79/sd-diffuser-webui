"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import os
import gradio as gr
import gc
import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import numpy as np
import imageio.v3 as imageio
import glob
import pathlib
import re
from datetime import datetime
from loadimg import load_img
import modules.util.appstate
from datetime import datetime
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager

OUTPUT_DIR = "output/birefnet/"
os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
try:
    import imageio_ffmpeg
    got_ffmpeg = True
except ImportError:
    got_ffmpeg = False

transform_image = None

def common_setup(w, h):
    global transform_image
    transform_image = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

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

def get_timestamp():
    return datetime.now().strftime("%d%m%Y%H%M%S")

def process(image, save_flat, bg_colour, model_name):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        get_pipeline(model_name)
        global transform_image
        if isinstance(image, str):
            input_filename = pathlib.Path(image).stem + f"_{get_timestamp()}"
        else:
            input_filename = get_timestamp()

        im = load_img(image, output_type="pil",input_type="auto").convert("RGB")
        image_size = im.size
        image = load_img(im, output_type="pil",input_type="auto")
        input_image = transform_image(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.float16)

        with torch.no_grad():
            preds = modules.util.appstate.global_pipe(input_image)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)

        if save_flat:
            if bg_colour.startswith('rgba'):
                bg_colour = rgba_to_hex(bg_colour)
            if not (bg_colour.startswith('#') and len(bg_colour) == 9):
                raise ValueError(f"Invalid background color format: {bg_colour}. Expected #RRGGBBAA (e.g., #FFFFFFFF)")
            try:
                colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
                background = Image.new("RGBA", image_size, colour_rgb)
                image = Image.alpha_composite(background, image).convert("RGB")
            except ValueError as e:
                raise ValueError(f"Failed to parse background color {bg_colour}: {str(e)}")

        # Save output to output/birefnet with timestamp
        output_filename = f"{OUTPUT_DIR}{input_filename}.png"
        image.save(output_filename)
        
        return image, output_filename, f"{image_size[0]} x {image_size[1]}"

    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")
    finally:
        modules.util.appstate.global_inference_in_progress = False

def get_image_info(image):
    """Return image dimensions as a string."""
    if image is None:
        return "No image uploaded"
    try:
        img = load_img(image, output_type="pil")
        return f"{img.size[0]} x {img.size[1]}"
    except Exception:
        return "Error retrieving dimensions"

def video_process(video_path, bg_colour, save_flat, model_name):
    if not got_ffmpeg:
        raise RuntimeError("imageio-ffmpeg is required for video processing but not installed.")
    get_pipeline(model_name)
    try:
        # reader = imageio.get_reader(video_path)
        # reader = imageio.imread(video_path, plugin='ffmpeg')
        reader = imageio.imopen(video_path, "r", plugin="pyav")
        # fps = reader.get_meta_data()['fps']
        meta = reader.metadata()
        fps = meta.get('fps', 24)
        frames = []
        while True:
            try:
                frame = reader.read()
                if frame is None:
                    break
                frames.append(frame)
            except Exception:
                break
        processed_frames = []
        image_size = None
        background = None
        global transform_image
        if bg_colour.startswith('rgba'):
            bg_colour = rgba_to_hex(bg_colour)
        if not (bg_colour.startswith('#') and len(bg_colour) == 9):
            raise ValueError(f"Invalid background color format: {bg_colour}. Expected #RRGGBBAA (e.g., #FFFFFFFF)")
        colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))

        for i, frame in enumerate(frames):
            print(f"BiRefNet [video]: frame {i+1}/{len(frames)}", end='\r', flush=True)
            image = Image.fromarray(frame).convert("RGB")

            if i == 0:
                image_size = image.size
                background = Image.new("RGBA", image_size, colour_rgb)

            input_image = transform_image(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.float16)
            with torch.no_grad():
                preds = modules.util.appstate.global_pipe(input_image)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(image_size)

            image.putalpha(mask)
            if save_flat:
                processed_image = Image.alpha_composite(background, image).convert("RGB")
            else:
                processed_image = image
            processed_frames.append(np.array(processed_image))

        os.makedirs("output/birefnet", exist_ok=True)
        output_filename = f"output/birefnet/{get_timestamp()}.mp4"
        imageio.mimwrite(output_filename, processed_frames, fps=fps, codec='h264')
        reader.close()
        return output_filename, f"{image_size[0]} x {image_size[1]}"
    except Exception as e:
        raise RuntimeError(f"Error processing video: {str(e)}")

def get_video_info(video_path):
    """Return video dimensions as a string."""
    if video_path is None:
        return "No video uploaded"
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        reader.close()
        return f"{meta['size'][0]} x {meta['size'][1]}"
    except Exception:
        return "Error retrieving dimensions"

def batch_process(input_folder, save_flat, bg_colour, model_name, output_dir_save):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    modules.util.appstate.global_inference_in_progress = True
    try:
        os.makedirs(f"{output_dir_save}", exist_ok=True)
        get_pipeline(model_name)
        global transform_image
        image_extensions = ['.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.webp', '.avif']
        input_images = []
        for ext in image_extensions:
            input_images.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
        
        if not input_images:
            raise ValueError(f"No images found in {input_folder}")

        processed_images = []
        if save_flat:
            if bg_colour.startswith('rgba'):
                bg_colour = rgba_to_hex(bg_colour)
            if not (bg_colour.startswith('#') and len(bg_colour) == 9):
                raise ValueError(f"Invalid background color format: {bg_colour}. Expected #RRGGBBAA (e.g., #FFFFFFFF)")
            colour_rgb = tuple(int(bg_colour[i:i+2], 16) for i in (1, 3, 5, 7))
        else:
            colour_rgb = None

        for i, image_path in enumerate(input_images):
            # print(f"BiRefNet [batch]: image {i+1}/{len(input_images)}", end='\r', flush=True)
            try:
                im = load_img(image_path, output_type="pil").convert("RGB")
                image_size = im.size
                image = load_img(im, output_type="pil")
                
                input_image = transform_image(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).to(torch.float16)
                with torch.no_grad():
                    preds = modules.util.appstate.global_pipe(input_image)[-1].sigmoid().cpu()
                
                pred = preds[0].squeeze()
                pred_pil = transforms.ToPILImage()(pred)
                mask = pred_pil.resize(image_size)
                image.putalpha(mask)
                
                output_filename = os.path.join(output_dir_save, f"{pathlib.Path(image_path).stem}.png")
                if save_flat:
                    background = Image.new("RGBA", image_size, colour_rgb)
                    image = Image.alpha_composite(background, image).convert("RGB")

                image.save(output_filename)
                processed_images.append(output_filename)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        return processed_images
    except Exception as e:
        raise RuntimeError(f"Error in batch processing: {str(e)}")
    finally:
        modules.util.appstate.global_inference_in_progress = False
# modules.util.appstate.global_inference_type
def get_pipeline(model_name):
    # print("----model: ", model_name)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "BiRefNet" and
        modules.util.appstate.global_inference_type == model_name):
        print(">>>>Reusing BiRefNet pipe<<<<")
        return modules.util.appstate.global_pipe
    else:
        # global transform_image
        # transform_image = None
        clear_previous_model_memory()

    modules.util.appstate.global_pipe = AutoModelForImageSegmentation.from_pretrained(
        f"ZhengPeng7/{model_name}",
        trust_remote_code=True,
    )
    modules.util.appstate.global_pipe.eval()
    modules.util.appstate.global_pipe.half()
    modules.util.appstate.global_pipe.to("cuda")

    modules.util.appstate.global_inference_type = model_name
    # return modules.util.appstate.global_pipe

def create_birefnet_tab():
    initial_state = state_manager.get_state("birefnet") or {}
    gr.Markdown("# BiRefNet for Background Removal")

    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Upload an image", height=584, type="filepath")
                image_info = gr.Textbox(label="Image Dimensions", interactive=False)
                with gr.Row():
                    image_save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                    image_bg_colour = gr.ColorPicker(label="Background color for flat images", value="#FFFFFFFF")
                go_image = gr.Button("Remove Background")
            with gr.Column():
                result1 = gr.Image(label="BiRefNet Output", type="pil", height=544)
                result1_info = gr.Textbox(label="Output Dimensions", interactive=False)
                result1_file = gr.Textbox(label="Output File", interactive=False)

    with gr.Tab("URL"):
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="URL or local path to image", max_lines=1)
                with gr.Row():
                    url_save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                    url_bg_colour = gr.ColorPicker(label="Background color for flat images", value="#FFFFFFFF")
                go_text = gr.Button("Remove Background")
            with gr.Column():
                result2 = gr.Image(label="BiRefNet Output", type="pil", height=544)
                result2_info = gr.Textbox(label="Output Dimensions", interactive=False)
                result2_file = gr.Textbox(label="Output File", interactive=False)

    if got_ffmpeg:
        with gr.Tab("Video", visible=False):
            with gr.Row():
                with gr.Column():
                    video = gr.Video(label="Upload a video", height=584)
                    video_info = gr.Textbox(label="Video Dimensions", interactive=False)
                    with gr.Row():
                        video_save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                        video_bg_colour = gr.ColorPicker(label="Background color for flat video", value="#FFFFFFFF")
                    go_video = gr.Button("Remove Background")
                with gr.Column():
                    result4 = gr.Video(label="BiRefNet Output", height=544, show_share_button=False)
                    result4_info = gr.Textbox(label="Output Dimensions", interactive=False)
                    result4_file = gr.Textbox(label="Output File", interactive=False)

    with gr.Tab("Batch"):
        with gr.Row():
            with gr.Column():
                input_dir = gr.Textbox(label="Input folder path", max_lines=1)
                output_dir_save = gr.Textbox(label="Output folder path", max_lines=1, value=f"{OUTPUT_DIR}")
                with gr.Row():
                    batch_save_flat = gr.Checkbox(label="Save flat (no mask)", value=False)
                    batch_bg_colour = gr.ColorPicker(label="Background color for flat images", value="#FFFFFFFF")
                go_batch = gr.Button("Remove Background(s)")
            with gr.Column():
                result3 = gr.File(label="Processed image(s)", file_count="multiple")

    with gr.Tab("Options"):
        gr.Markdown("*HR*: High resolution; *matting*: Better with transparency; *lite*: Faster.")
        model = gr.Dropdown(
            label="Model (downloads on selection, see console for progress)",
            choices=[
                "BiRefNet_512x512", "BiRefNet", "BiRefNet_HR", "BiRefNet-matting",
                "BiRefNet_HR-matting", "BiRefNet_lite", "BiRefNet_lite-2K",
                "BiRefNet-portrait", "BiRefNet-COD", "BiRefNet-DIS5K",
                "BiRefNet-DIS5k-TR_TEs", "BiRefNet-HRSOD"
            ],
            value="BiRefNet_HR"
        )
        gr.Markdown("Regular models trained at 1024×1024; HR models at 2048×2048; 2K model at 2560×1440.")
        gr.Markdown("Larger processing sizes improve accuracy but require more VRAM.")
        with gr.Row():
            proc_sizeW = gr.Slider(label="Processing image width", minimum=256, maximum=2560, value=2048, step=32)
            proc_sizeH = gr.Slider(label="Processing image height", minimum=256, maximum=2048, value=2048, step=32)
        with gr.Row():
            save_state_button = gr.Button("Save State")
        model.change(fn=get_pipeline, inputs=model, outputs=None)
        gr.Markdown("### https://github.com/ZhengPeng7/BiRefNet\n### https://huggingface.co/ZhengPeng7")

    # Event handlers
    image.change(fn=get_image_info, inputs=image, outputs=image_info)
    go_image.click(fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(
        fn=process, inputs=[image, image_save_flat, image_bg_colour, model], outputs=[result1, result1_file, result1_info]
    )
    go_text.click(fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(
        fn=process, inputs=[text, url_save_flat, url_bg_colour, model], outputs=[result2, result2_file, result2_info]
    )
    if got_ffmpeg:
        video.change(fn=get_video_info, inputs=video, outputs=video_info)
        go_video.click(fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(
            fn=video_process, inputs=[video, video_bg_colour, video_save_flat, model], outputs=[result4, result4_file, result4_info]
        )
    go_batch.click(fn=common_setup, inputs=[proc_sizeW, proc_sizeH]).then(
        fn=batch_process, inputs=[input_dir, batch_save_flat, batch_bg_colour, model, output_dir_save], outputs=result3
    )


    def save_current_state(model):
        state_dict = {
            "model": model,
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("birefnet") or {}
        state_manager.save_state("birefnet", state_dict)
        return model

    # Event handlers
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            model
        ],
        outputs=[
            model
        ]
    )

