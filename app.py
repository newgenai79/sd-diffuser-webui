"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import sys
import os
import gradio as gr
import torch
from pathlib import Path
import json

# Text 2 Image
from modules.text2image.tab_lumina import create_lumina_tab
from modules.text2image.tab_lumina2 import create_lumina2_tab
from modules.text2image.tab_flux import create_flux_tab
# from modules.text2image.tab_flex1_alpha import create_flex1_alpha_tab
from modules.text2image.tab_sana import create_sana_tab
from modules.text2image.tab_cogview3plus import create_cogView3Plus_tab
from modules.text2image.tab_hunyuandit import create_hunyuandit_tab
from modules.text2image.tab_kandinsky3 import create_kandinsky3_tab
from modules.text2image.tab_auraflow_gguf import create_auraflow_gguf_tab

# Text 2 Video
from modules.text2video.tab_ltxvideo091 import create_ltxvideo091_tab
from modules.text2video.tab_hunyuanvideo_gguf import create_hunyuanvideo_gguf_tab
from modules.text2video.tab_hunyuanvideo_bnb import create_hunyuanvideo_bnb_tab
from modules.text2video.tab_cogvideox155b import create_cogvideox155b_t2v_tab

# Image 2 Video
from modules.image2video.tab_ltximage2video091 import create_ltximage2video091_tab
from modules.image2video.tab_cogvideox155b_i2v import create_cogvideox155b_i2v_tab

# Video 2 Video
from modules.video2video.tab_cogvideox155b_f2v import create_cogvideox155b_f2v_tab
from modules.video2video.tab_cogvideox155b_v2v import create_cogvideox155b_v2v_tab

# Import utilities for metadata handling
from modules.util.utilities import read_metadata_from_file

def format_time(seconds):
    """Convert seconds to minutes and seconds format"""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}m {remaining_seconds}s"

def format_metadata(metadata_dict):
    """Format metadata dictionary into a readable string"""
    if not metadata_dict:
        return "No metadata found"
        
    formatted_text = ""
    for key, value in metadata_dict.items():
        # Format key by replacing underscores with spaces and capitalizing
        formatted_key = key.replace('_', ' ').title()
        
        # Format value based on type and key
        if isinstance(value, bool):
            formatted_value = str(value)
        elif key == "generation_time":
            formatted_value = format_time(value)
        elif isinstance(value, (int, float)):
            if key in ["seed", "width", "height", "num_inference_steps", "guidance_scale"]:
                formatted_value = str(int(value))  # Remove commas for numbers
            else:
                formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
        else:
            formatted_value = str(value)
            
        formatted_text += f"{formatted_key}: {formatted_value}\n"
    
    return formatted_text

def create_info_tab():
    """Create the Info tab with metadata display functionality"""
    with gr.Column():
        gr.Markdown("## Image Metadata Information")
        with gr.Row():
            # Image input for uploading
            image_input = gr.Image(
                label="Upload image to view metadata",
                type="filepath"
            )
            
            # Text area for displaying metadata
            metadata_display = gr.Textbox(
                label="Metadata Information",
                interactive=False,
                lines=20,
                max_lines=30
            )
            
        # Function to handle image upload and display metadata
        def update_metadata(image_path):
            if not image_path:
                return "Please upload an image to view its metadata"
            
            metadata = read_metadata_from_file(image_path)
            if metadata:
                return format_metadata(metadata)
            return "No metadata found in the image"
            
        # Connect the image input to the metadata display
        image_input.change(
            fn=update_metadata,
            inputs=[image_input],
            outputs=[metadata_display]
        )

# Create the main WebUI
with gr.Blocks() as dwebui:
    gr.Markdown("# WebUI for Image/Video generation")
    with gr.Tabs():
        # Text 2 Image Tab
        with gr.Tab("Text 2 Image"):
            with gr.Tabs():
                with gr.Tab("Lumina Image 2.0"):
                    create_lumina2_tab()
                with gr.Tab("Sana"):
                    create_sana_tab()
                with gr.Tab("Flux"):
                    create_flux_tab()
                with gr.Tab("Lumina Next SFT"):
                    create_lumina_tab()
                with gr.Tab("CogView3 Plus"):
                    create_cogView3Plus_tab()
                with gr.Tab("Hunyuan DiT"):
                    create_hunyuandit_tab()
                with gr.Tab("Kandinsky-3"):
                    create_kandinsky3_tab()
                with gr.Tab("AuraFlow 0.3-GGUF"):
                    create_auraflow_gguf_tab()
        
        # Text 2 Video Tab
        with gr.Tab("Text 2 Video"):
            with gr.Tabs():
                with gr.Tab("LTX-Video 0.9.1"):
                    create_ltxvideo091_tab()
                with gr.Tab("HunyuanVideo-GGUF"):
                    create_hunyuanvideo_gguf_tab()
                with gr.Tab("HunyuanVideo-BitsnBytes"):
                    create_hunyuanvideo_bnb_tab()
                with gr.Tab("CogVideoX v1.5-5b"):
                    create_cogvideox155b_t2v_tab()
        
        # Image 2 Video Tab
        with gr.Tab("Image 2 Video"):
            with gr.Tabs():
                with gr.Tab("LTX-Video 0.9.1"):
                    create_ltximage2video091_tab()
                with gr.Tab("CogVideoX v1.5-5b"):
                    create_cogvideox155b_i2v_tab()
        
        # Video 2 Video Tab
        with gr.Tab("Video 2 Video"):
            with gr.Tabs():
                with gr.Tab("CogVideoX-Fun v1.1-5b-Pose"):
                    create_cogvideox155b_f2v_tab()
                with gr.Tab("CogVideoX v1.5-5b"):
                    create_cogvideox155b_v2v_tab()
        
        with gr.Tab("Info"):
            create_info_tab()

# Launch the WebUI
dwebui.launch(share=False)