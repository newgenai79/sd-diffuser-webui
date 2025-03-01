import gradio as gr
import torch
import cv2
import os
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import tempfile
from modules.util.utilities import clear_previous_model_memory

class VideoUpscaler:
    def __init__(self):
        self.models = {
            "RealESRGAN_x4plus": {
                "scale": 4,
                "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            },
            "RealESRNet_x4plus": {
                "scale": 4,
                "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
            },
            "RealESRGAN_x4plus_anime_6B": {
                "scale": 4,
                "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            },
            "RealESRGAN_x2plus": {
                "scale": 2,
                "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            },
            "realesr-animevideov3": {
                "scale": 4,
                "model": SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'),
                "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
            }
        }
        self.current_model = None
        self.face_enhancer = None
        os.makedirs("models/video_upscaler", exist_ok=True)
        os.makedirs("output/upscaled_video", exist_ok=True)
        os.makedirs("temp_output", exist_ok=True)

    def get_video_properties(self, video_path):
        if not video_path:
            return "No video selected"
            
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        cap.release()
        
        return f"Resolution: {width}x{height} | FPS: {fps} | Duration: {duration:.2f}s | Frames: {frame_count}"
        
    def load_model(self, model_name, denoise_strength=1):
        model_info = self.models[model_name]
        model = model_info["model"]
        scale = model_info["scale"]

        model_path = os.path.join("models/video_upscaler/", f"{model_name}.pth")
        if not os.path.exists(model_path):
            load_file_from_url(model_info["url"], model_dir="models/video_upscaler")

        # Handle Dynamic Noise Injection for specific models
        dni_weight = None
        if model_name == "realesr-animevideov3" and denoise_strength != 1:
            wdn_model_path = model_path.replace("realesr-general-x4v3", "realesr-general-wdn-x4v3")
            model_path = [model_path, wdn_model_path]
            dni_weight = [denoise_strength, 1 - denoise_strength]

        self.current_model = RealESRGANer(
            scale=scale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=10,
            half=torch.cuda.is_available(),
            gpu_id=0 if torch.cuda.is_available() else None,
        )

    def load_face_enhancer(self, outscale):
        if self.face_enhancer is None:
            self.face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
                upscale=outscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.current_model,
            )

    def process_video(self, video_path, model_name, denoise_strength, face_enhance, outscale, progress=gr.Progress()):
        if not video_path:
            return None, "", ""
        clear_previous_model_memory()
        import time
        start_time = time.time()

        # Generate output filename with absolute paths
        input_filename = os.path.basename(video_path)
        name, ext = os.path.splitext(input_filename)
        output_path = os.path.abspath(os.path.join("output/upscaled_video/", f"{name}_upscaled{ext}"))
        temp_output = os.path.abspath(os.path.join("temp_output", f"temp_{name}{ext}"))
        
        # Load model
        self.load_model(model_name, denoise_strength)
        if face_enhance:
            self.load_face_enhancer(outscale)

        # Set up video processing
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)  # Keep as float for accurate duration
        
        processed_frames = 0
        success, frame = cap.read()
        if not success:
            return None, "", ""

        h, w = frame.shape[:2]
        output_h, output_w = int(h * outscale), int(w * outscale)

        # Reset video capture to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Use more compatible codec
        writer = cv2.VideoWriter(
            temp_output,
            cv2.VideoWriter_fourcc(*"mp4v"),  # Try H.264 codec instead of mp4v
            fps,
            (output_w, output_h),
        )

        # Process video frames
        while True:
            success, frame = cap.read()
            if not success:
                break

            if face_enhance:
                _, _, frame = self.face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                frame, _ = self.current_model.enhance(frame, outscale=outscale)

            writer.write(frame)
            processed_frames += 1
            progress(processed_frames / total_frames)

        cap.release()
        writer.release()

        # Calculate processing time
        processing_time = time.time() - start_time
        processing_info = f"Processed {processed_frames}/{total_frames} frames in {processing_time:.2f} seconds ({processed_frames/processing_time:.2f} fps)"

        # Initialize video info output
        video_info_output = ""
        
        # Get output video info directly
        try:
            # Skip MoviePy and directly copy the video
            if os.path.exists(temp_output):
                # Use FFmpeg directly instead of MoviePy
                import subprocess
                
                # If the video has audio, try to copy it as well
                try:
                    audio_check = cv2.VideoCapture(video_path)
                    has_audio = False
                    if audio_check.isOpened():
                        has_audio = True
                    audio_check.release()
                    
                    if has_audio:
                        # Copy video with audio using ffmpeg
                        cmd = ["ffmpeg", "-y", "-i", temp_output, "-i", video_path, 
                               "-c:v", "copy", "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0?", 
                               "-shortest", output_path]
                    else:
                        # No audio, just copy the video
                        cmd = ["ffmpeg", "-y", "-i", temp_output, "-c:v", "copy", output_path]
                        
                    subprocess.run(cmd, check=True, capture_output=True)
                except Exception as e:
                    print(f"FFmpeg error: {str(e)}")
                    # Fallback - just move the temp file
                    if os.path.exists(temp_output):
                        import shutil
                        shutil.copy(temp_output, output_path)
                
                # Clean up temp file
                try:
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                except:
                    pass
            
            # Get output video info
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                video_info_output = f"Resolution: {width}x{height} | FPS: {fps:.2f} | Duration: {duration:.2f}s | Frames: {frame_count}"
                cap.release()
            else:
                video_info_output = "Could not get video information"

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            video_info_output = f"Error: {str(e)}"
            if os.path.exists(temp_output) and not os.path.exists(output_path):
                # Last resort - just use the temp file
                import shutil
                shutil.copy(temp_output, output_path)
        finally:
            if self.current_model is not None:
                self.current_model = None
            if self.face_enhancer is not None:
                self.face_enhancer = None
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        if self.current_model is not None:
            self.current_model = None
        if self.face_enhancer is not None:
            self.face_enhancer = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        return output_path, processing_info, video_info_output


def create_video_upscaler_interface():
    upscaler = VideoUpscaler()

    def update_outscale(model_name):
        scale = upscaler.models[model_name]["scale"]
        return scale

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video")
            video_info = gr.Textbox(label="Video Information", interactive=False)
            model_name = gr.Dropdown(
                choices=list(upscaler.models.keys()),
                value="RealESRGAN_x4plus",
                label="Model",
            )
            denoise_strength = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.1,
                label="Denoise Strength",
            )
            outscale = gr.Slider(
                minimum=1,
                maximum=4,
                value=4,
                step=1,
                label="Output Scale",
            )
            face_enhance = gr.Checkbox(label="Enable Face Enhancement (GFPGAN)")
            process_btn = gr.Button("Process Video")

        with gr.Column():
            output_video = gr.Video(label="Output Video")
            processing_info = gr.Textbox(label="Processing time", interactive=False)
            video_info_output = gr.Textbox(label="Video Information", interactive=False)

    def process(video_path, model, strength, enhance, scale):
        return upscaler.process_video(video_path, model, strength, enhance, scale)
    model_name.change(
        fn=update_outscale,
        inputs=[model_name],
        outputs=[outscale],
    )
    process_btn.click(
        fn=upscaler.process_video,
        inputs=[input_video, model_name, denoise_strength, face_enhance, outscale],
        outputs=[output_video, processing_info, video_info_output],
        show_progress=True,
    )

    input_video.change(
        fn=upscaler.get_video_properties,
        inputs=[input_video],
        outputs=[video_info],
    )

