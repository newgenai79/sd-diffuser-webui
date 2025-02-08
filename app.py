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
from modules.text2image.tab_sana import create_sana_tab
from modules.text2image.tab_flex1_alpha_gguf import create_flex1_alpha_gguf_tab
from modules.text2image.tab_lumina import create_lumina_tab
from modules.text2image.tab_lumina2_bnb import create_lumina2_BnB_tab
# from modules.text2image.tab_lumina2_gguf import create_lumina2_gguf_tab
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

with gr.Blocks() as dwebui:
    gr.Markdown("# WebUI for Image/Video generation")
    with gr.Tabs():
        with gr.Tab("Text 2 Image"):
            with gr.Tabs():
                with gr.Tab("Sana"):
                    create_sana_tab()
                with gr.Tab("Lumina Image 2.0 - BnB"):
                    create_lumina2_BnB_tab()
                # with gr.Tab("Lumina Image 2.0 - GGUF"):
                    # create_lumina2_gguf_tab()
                with gr.Tab("Flex.1-alpha-GGUF"):
                    create_flex1_alpha_gguf_tab()
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
        with gr.Tab("Image 2 Video"):
            with gr.Tabs():
                with gr.Tab("LTX-Video 0.9.1"):
                    create_ltximage2video091_tab()
                with gr.Tab("CogVideoX v1.5-5b"):
                    create_cogvideox155b_i2v_tab()
        with gr.Tab("Video 2 Video"):
                with gr.Tab("CogVideoX-Fun v1.1-5b-Pose"):
                    create_cogvideox155b_f2v_tab()
                with gr.Tab("CogVideoX v1.5-5b"):
                    create_cogvideox155b_v2v_tab()

dwebui.launch(share=False)