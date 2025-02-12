"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import sys
import os
import gradio as gr
import torch

"""
current_dir = os.path.dirname(os.path.abspath(__file__))
tabs_dir = os.path.join(current_dir, 'modules')
sys.path.append(tabs_dir)
"""

# Text 2 Image
from modules.text2image.tab_lumina import create_lumina_tab
from modules.text2image.tab_lumina2_bnb import create_lumina2_BnB_tab
from modules.text2image.tab_lumina2_gguf import create_lumina2_gguf_tab

with gr.Blocks() as dwebui:
    gr.Markdown("# WebUI for Image/Video generation")
    with gr.Tabs():
        with gr.Tab("Text 2 Image"):
            with gr.Tabs():
                with gr.Tab("Lumina Image 2.0 - BnB"):
                    create_lumina2_BnB_tab()
                with gr.Tab("Lumina Image 2.0 - GGUF"):
                    create_lumina2_gguf_tab()
                with gr.Tab("Lumina 1.0 Next SFT"):
                    create_lumina_tab()
        with gr.Tab("Text 2 Video"):
            print("")
        with gr.Tab("Image 2 Video"):
            print("")
        with gr.Tab("Video 2 Video"):
            print("")
        with gr.Tab("Extra"):
            print("")

dwebui.launch(share=False)