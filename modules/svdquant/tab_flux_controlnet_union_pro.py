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
from datetime import datetime
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from controlnet_aux import (
    CannyDetector, LineartDetector, LineartStandardDetector, LineartAnimeDetector,
    MidasDetector, OpenposeDetector
)
from image_gen_aux import (
    DepthPreprocessor
)
from diffusers import FluxControlNetModel, FluxControlNetPipeline
from diffusers.models import FluxMultiControlNetModel
from diffusers.utils import load_image
from modules.util.utilities import clear_previous_model_memory, clear_controlnet_model_memory
from modules.util.appstate import state_manager
from nunchaku.utils import get_precision
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.caching.teacache import TeaCache
from contextlib import nullcontext

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"
os.makedirs(OUTPUT_DIR, exist_ok=True)
control_output_dir = "output/control_images"
os.makedirs(control_output_dir, exist_ok=True)

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, inference_type, performance_optimization, diff_multi, diff_single):
    print("----FluxControlNetPipeline mode: ", memory_optimization, inference_type, performance_optimization, diff_multi, diff_single)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        (type(modules.util.appstate.global_pipe).__name__ == "FluxControlNetPipeline" or type(modules.util.appstate.global_pipe).__name__ == "FluxFillPipeline") and 
            modules.util.appstate.global_inference_type == inference_type and 
            modules.util.appstate.global_performance_optimization == performance_optimization and
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing FluxControlNetPipeline pipe<<<<")
                return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
        torch.cuda.synchronize()

    dtype = torch.bfloat16
    precision = get_precision()
    flux_repo = "black-forest-labs/FLUX.1-dev"
    controlnet_model_union = f"Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
    controlnet_union = FluxControlNetModel.from_pretrained(
        controlnet_model_union, 
        torch_dtype=torch.bfloat16
    )
    controlnet = FluxMultiControlNetModel([controlnet_union])

    transformer_repo = f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        transformer_repo, 
        offload=True
    )
    if performance_optimization == "nunchaku-fp16":
        transformer.set_attention_impl("nunchaku-fp16")

    text_encoder_2_repo = f"mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"
    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        text_encoder_2_repo,
    )
    modules.util.appstate.global_pipe = FluxControlNetPipeline.from_pretrained(
        flux_repo, 
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
    )

    if performance_optimization == "double_cache":
        apply_cache_on_pipe(
            modules.util.appstate.global_pipe,
            use_double_fb_cache=True,
            residual_diff_threshold_multi=float(diff_multi),
            residual_diff_threshold_single=float(diff_single),
        )
    if memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
        
    # Update global variables
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_inference_type = inference_type
    modules.util.appstate.global_performance_optimization = performance_optimization
    modules.util.appstate.global_text_encoder_2 = text_encoder_2
    modules.util.appstate.global_controlnet = controlnet
    modules.util.appstate.global_transformer = transformer
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, width, height, guidance_scale, num_inference_steps,
    memory_optimization, no_of_images, randomize_seed, 
    performance_optimization, diff_multi, diff_single, tea_cache_thresh,
    control_image, control_mode, controlnet_conditioning_scale
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    if modules.util.appstate.global_controlnet_model is not None:
        clear_controlnet_model_memory()
    gallery_items = []

    try:
        inference_type = "FLUX.1-dev-ControlNet-Union-Pro"
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, inference_type, performance_optimization, diff_multi, diff_single)

        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx+1}: (Step {i}/{num_inference_steps})")
            return callback_kwargs
        guidance_scale = float(guidance_scale)
        modules.util.appstate.global_inference_in_progress = True

        base_filename = "controlnet_pro"
        # Generate multiple images in a loop
        for img_idx in range(no_of_images):
            
            # If randomize_seed is True, generate a new random seed for each image
            current_seed = random_seed() if randomize_seed else seed
            
            # Create generator with the current seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            # Prepare inference parameters
            # if control_image:
                # width, height = control_image[0].size
            inference_params = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "control_image": control_image,
                "control_mode": control_mode, 
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if performance_optimization != "teacache":
                inference_params["num_inference_steps"] = num_inference_steps
            print(inference_params)
            print(f"Generating image {img_idx+1}/{no_of_images} with seed: {current_seed}")
            
            start_time = datetime.now()
            # Generate image
            with (
                TeaCache(model=modules.util.appstate.global_transformer, num_steps=num_inference_steps, rel_l1_thresh=float(tea_cache_thresh), enabled=True)
                if performance_optimization == "teacache" else nullcontext()
            ):
                image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            # Create filename with index
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_flux_{base_filename}_{img_idx+1}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            metadata = {
                "model": inference_type,
                "control_mode": control_mode, 
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
                "prompt": prompt,
                "seed": current_seed,
                "guidance_scale": f"{float(guidance_scale):.2f}",
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "memory_optimization": memory_optimization,
                "performance_optimization": performance_optimization,
                "timestamp": timestamp,
                "generation_time": generation_time,
            }
            
            # Save the image
            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image {img_idx+1}/{no_of_images} generated: {output_path}")
            
            # Add to gallery items
            gallery_items.append((output_path, f"FLUX.1-dev-{inference_type}"))

        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        modules.util.appstate.global_inference_in_progress = False

def create_flux_controlnet_union_pro_tab():
    gr.HTML("<style>.small-button { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; align-self: end; line-height: 1em; border-radius: 0.5em; }</style>", visible=False)
    initial_state = state_manager.get_state("flux-controlnet_union_pro") or {}
    selected_perf_opt = initial_state.get("performance_optimization", "no_optimization")
    with gr.Row():
        with gr.Accordion("Inference type / Optimizations", open=True):
            with gr.Row():
                flux_memory_optimization = gr.Radio(
                    choices=["No optimization", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    value=initial_state.get("memory_optimization", "Extremely Low VRAM"),
                    interactive=True
                )
            gr.Markdown("### Performance Optimizations")
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    flux_no_optimization = gr.Checkbox(
                        label="No optimization", 
                        value=selected_perf_opt == "no_optimization", 
                        interactive=True
                    )
                with gr.Column(scale=1, min_width=200):
                    flux_nunchaku_fp16 = gr.Checkbox(
                        label="nunchaku-fp16", 
                        value=selected_perf_opt == "nunchaku_fp16", 
                        interactive=True
                    )
                with gr.Column(scale=2, min_width=200):
                    flux_double_cache = gr.Checkbox(
                        label="Double cache", 
                        value=selected_perf_opt == "double_cache", 
                        interactive=True
                    )
                    flux_diff_multi = gr.Slider(
                        label="Residual_diff_threshold_multi", 
                        minimum=0, 
                        maximum=1, 
                        value=initial_state.get("diff_threshold_multi", 0.09),
                        step=0.01,
                        interactive=True
                    )
                    flux_diff_single = gr.Slider(
                        label="Residual_diff_threshold_single", 
                        minimum=0, 
                        maximum=1, 
                        value=initial_state.get("diff_threshold_single", 0.12),
                        step=0.01,
                        interactive=True
                    )
                with gr.Column(scale=2, min_width=300):
                    flux_teacache = gr.Checkbox(
                        label="Teacache", 
                        value=selected_perf_opt == "teacache", 
                        interactive=True
                    )
                    flux_tea_cache_l1_thresh_slider = gr.Slider(
                        label="Tea cache (rel_l1_thresh)", 
                        minimum=0, 
                        maximum=1, 
                        value=initial_state.get("tea_cache_threshold", 0.3),
                        step=0.01,
                        interactive=True
                    )

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Tabs():
                    with gr.Tab("Canny"):
                        with gr.Row():
                            enable_canny = gr.Checkbox(label="Enable Canny", value=False, interactive=True)
                            canny_controlnet_conditioning_scale = gr.Slider(label="Controlnet conditioning scale", minimum=0, maximum=2, value=0.6, step=0.1, interactive=True)
                        with gr.Row():
                            canny_pre_processor = gr.Radio(
                                choices=["Canny", "Lineart", "Lineart anime", "Lineart standard"],
                                label="Preprocessor",
                                value="Canny",
                                interactive=False
                            )
                        with gr.Row():
                            canny_input_image = gr.Image(label="Source image", type="filepath", width=256, height=256, interactive=False)
                            canny_preview_button = gr.Button("👁️", elem_classes="small-button", interactive=False)
                            canny_output_image = gr.Image(label="Control image (mandatory)", type="filepath", width=256, height=256, interactive=False)
                        with gr.Row():
                            canny_input_size = gr.Markdown(value="", visible=True)
                            canny_test = gr.Markdown(value="    ", visible=True)
                            canny_output_size = gr.Markdown(value="", visible=True)
                        with gr.Row():
                            tab_canny = gr.Accordion("Canny settings", open=False, visible=False)
                            with tab_canny:
                                canny_low_threshold = gr.Slider(label="Low threshold", minimum=0, maximum=256, value=50, step=1, interactive=True)
                                canny_high_threshold = gr.Slider(label="High threshold", minimum=0, maximum=256, value=200, step=1, interactive=True)
                                canny_detect_resolution = gr.Slider(label="Detect resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                                canny_image_resolution = gr.Slider(label="Image resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                        with gr.Row():
                            tab_lineart_anime = gr.Accordion("Lineart/anime settings", open=False, visible=False)
                            with tab_lineart_anime:
                                lineart_detect_resolution = gr.Slider(label="Detect resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                                lineart_image_resolution = gr.Slider(label="Image resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                        with gr.Row():
                            tab_linearts = gr.Accordion("Lineart standard settings", open=False, visible=False)
                            with tab_linearts:
                                linearts_detect_resolution = gr.Slider(label="Detect resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                                linearts_guassian_sigma = gr.Slider(label="Guassian_sigma", minimum=0.0, maximum=10.0, step=0.1, value=6.0, interactive=True)
                                linearts_intensity_threshold = gr.Slider(label="Intensity threshold", minimum=0, maximum=10, step=1, value=8, interactive=True)

                    with gr.Tab("Depth"):
                        with gr.Row():
                            enable_depth = gr.Checkbox(label="Enable Depth", value=False, interactive=True)
                            depth_controlnet_conditioning_scale = gr.Slider(label="Controlnet conditioning scale", minimum=0, maximum=2, value=0.6, step=0.1, interactive=True)
                        with gr.Row():
                            depth_pre_processor = gr.Radio(
                                choices=["HED", "depth-anything/Depth-Anything-V2-Large-hf"],
                                label="Preprocessor",
                                value="depth-anything/Depth-Anything-V2-Large-hf",
                                interactive=False
                            )
                        with gr.Row():
                            depth_input_image = gr.Image(label="Source image", type="filepath", width=256, height=256, interactive=False)
                            depth_preview_button = gr.Button("👁️", elem_classes="small-button", interactive=False)
                            depth_output_image = gr.Image(label="Control image (mandatory)", type="filepath", width=256, height=256, interactive=False)
                        with gr.Row():
                            depth_input_size = gr.Markdown(value="", visible=True)
                            depth_test = gr.Markdown(value="    ", visible=True)
                            depth_output_size = gr.Markdown(value="", visible=True)
                        with gr.Row():
                            tab_hed = gr.Accordion("HED settings", open=False, visible=False)
                            with tab_hed:
                                hed_depth_and_Normal = gr.Checkbox(label="Depth and Normal", value=False, interactive=True)
                                hed_bg_th = gr.Slider(label="BG TH", minimum=0, maximum=1, value=0.1, step=0.1, interactive=True)
                                hed_detect_resolution = gr.Slider(label="Detect resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                                hed_image_resolution = gr.Slider(label="Image resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                        with gr.Row():
                            tab_depth = gr.Accordion("Depth settings", open=False, visible=False)
                            with tab_depth:
                                depth_invert = gr.Checkbox(label="Invert", value=False, interactive=True)
                                depth_resolution_scale = gr.Slider(label="Resolution scale", minimum=1, maximum=4, value=1, step=1, interactive=True)

                    with gr.Tab("Pose"):
                        with gr.Row():
                            enable_pose = gr.Checkbox(label="Enable Pose", value=False, interactive=True)
                            pose_controlnet_conditioning_scale = gr.Slider(label="Controlnet conditioning scale", minimum=0, maximum=2, value=0.6, step=0.1, interactive=True)
                        with gr.Row():
                            pose_pre_processor = gr.Radio(
                                choices=["OpenPose", "DWPose"],
                                label="Preprocessor",
                                value="OpenPose",
                                interactive=False
                            )
                        with gr.Row():
                            pose_input_image = gr.Image(label="Source image", type="filepath", width=256, height=256, interactive=False)
                            pose_preview_button = gr.Button("👁️", elem_classes="small-button", interactive=False)
                            pose_output_image = gr.Image(label="Control image (mandatory)", type="filepath", width=256, height=256, interactive=False)
                        with gr.Row():
                            pose_input_size = gr.Markdown(value="", visible=True)
                            pose_test = gr.Markdown(value="    ", visible=True)
                            pose_output_size = gr.Markdown(value="", visible=True)
                        with gr.Row():
                            tab_openpose = gr.Accordion("OpenPose settings", open=False, visible=False)
                            with tab_openpose:
                                with gr.Row():
                                    openpose_include_body = gr.Checkbox(label="Include body", value=True, interactive=True)
                                    openpose_include_hand = gr.Checkbox(label="Include hand", value=True, interactive=True)
                                    openpose_include_face = gr.Checkbox(label="Include face", value=True, interactive=True)
                                with gr.Row():
                                    openpose_detect_resolution = gr.Slider(label="Detect resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                                    openpose_image_resolution = gr.Slider(label="Image resolution", minimum=128, maximum=2048, step=8, value=1024, interactive=True)
                        with gr.Row():
                            tab_dwpose = gr.Accordion("DWPose settings", open=False, visible=False)
                            with tab_dwpose:
                                with gr.Row():
                                    dwpose_include_hand = gr.Checkbox(label="Include hand", value=True, interactive=True)
                                    dwpose_include_face = gr.Checkbox(label="Include face", value=True, interactive=True)
#                    with gr.Tab("Tile"):
#                        enable_tile = gr.Checkbox(label="Enable Tile", value=False, interactive=True)
                    with gr.Tab("Blur"):
                        enable_blur = gr.Checkbox(label="Enable Blur", value=False, interactive=True)
                        blur_controlnet_conditioning_scale = gr.Slider(label="Controlnet conditioning scale", minimum=0, maximum=2, value=0.6, step=0.1, interactive=True)
                        blur_input_image = gr.Image(label="Source image (mandatory)", type="filepath", width=256, height=256, interactive=False)
                    with gr.Tab("Gray"):
                        enable_gray = gr.Checkbox(label="Enable Gray", value=False, interactive=True)
                        gray_controlnet_conditioning_scale = gr.Slider(label="Controlnet conditioning scale", minimum=0, maximum=2, value=0.6, step=0.1, interactive=True)
                        gray_input_image = gr.Image(label="Source image (mandatory)", type="filepath", width=256, height=256, interactive=False)
                    with gr.Tab("Low quality"):
                        enable_low_quality = gr.Checkbox(label="Enable Low quality", value=False, interactive=True)
                        low_quality_controlnet_conditioning_scale = gr.Slider(label="Controlnet conditioning scale", minimum=0, maximum=2, value=0.6, step=0.1, interactive=True)
                        low_quality_input_image = gr.Image(label="Source image (mandatory)", type="filepath", width=256, height=256, interactive=False)
        with gr.Column(scale=1):
            with gr.Row():
                flux_prompt_input = gr.Textbox(
                    label="Prompt", 
                    lines=9,
                    interactive=True
                )
            with gr.Row():
                flux_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1024),
                    interactive=True
                )
                flux_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1024),
                    interactive=True
                )
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("♻️ Randomize seed")
            with gr.Row():
                flux_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=50.0, 
                    value=initial_state.get("guidance_scale", 3.5),
                    step=0.1,
                    interactive=True
                )
                flux_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 28),
                    interactive=True
                )
            with gr.Row():
                flux_no_of_images_input = gr.Number(
                    label="Number of Images", 
                    value=1,
                    interactive=True
                )
                flux_randomize_seed = gr.Checkbox(label="Randomize seed", value=False, interactive=True)
    with gr.Row():
        generate_button = gr.Button("🎨 Generate image")
        save_state_button = gr.Button("💾 Save State")
    output_gallery = gr.Gallery(
        label="Generated Image(s)",
        columns=3,
        rows=None,
        height="auto"
    )
    
    # Add function to get image dimensions
    def get_image_dimensions(image_path):
        if image_path:
            try:
                image = load_image(image_path)
                width, height = image.size
                return f"{width} x {height} (w x h)"
            except:
                return "Invalid image"
        return ""

    components = [
        [canny_input_image, canny_input_size],
        [canny_output_image, canny_output_size],
        [depth_input_image, depth_input_size],
        [depth_output_image, depth_output_size],
        [pose_input_image, pose_input_size],
        [pose_output_image, pose_output_size],
    ]
    for input_comp, output_comp in components:
        input_comp.change(
            fn=get_image_dimensions,
            inputs=[input_comp],
            outputs=[output_comp]
        )

    
    def toggle_tabs_canny(pre):
        return {
            "Canny": [gr.update(open=True), gr.update(open=False), gr.update(open=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)],
            "Lineart": [gr.update(open=False), gr.update(open=True), gr.update(open=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
            "Lineart anime": [gr.update(open=False), gr.update(open=True), gr.update(open=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)],
            "Lineart standard": [gr.update(open=False), gr.update(open=False), gr.update(open=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)],
        }[pre]

    canny_pre_processor.change(
        fn=toggle_tabs_canny,
        inputs=[canny_pre_processor],
        outputs=[tab_canny, tab_lineart_anime, tab_linearts, tab_canny, tab_lineart_anime, tab_linearts]
    )
    
    def toggle_tabs_depth(pre):
        return {
            "HED": [gr.update(open=True), gr.update(open=False), gr.update(visible=True), gr.update(visible=False)],
            "depth-anything/Depth-Anything-V2-Large-hf": [gr.update(open=False), gr.update(open=True), gr.update(visible=False), gr.update(visible=True)],
        }[pre]

    depth_pre_processor.change(
        fn=toggle_tabs_depth,
        inputs=[depth_pre_processor],
        outputs=[tab_hed, tab_depth, tab_hed, tab_depth]
    )

    def toggle_tabs_pose(pre):
        return {
            "OpenPose": [gr.update(open=True), gr.update(open=False), gr.update(visible=True), gr.update(visible=False)],
            "DWPose": [gr.update(open=False), gr.update(open=True), gr.update(visible=False), gr.update(visible=True)],
        }[pre]

    pose_pre_processor.change(
        fn=toggle_tabs_pose,
        inputs=[pose_pre_processor],
        outputs=[tab_openpose, tab_dwpose, tab_openpose, tab_dwpose]
    )
    
    def toggle_canny_components(is_enabled, pre_processor):
        return [
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(visible=is_enabled and pre_processor == "Canny", open=is_enabled and pre_processor == "Canny"),  # tab_canny
            gr.update(visible=is_enabled and pre_processor in ["Lineart", "Lineart anime"], open=is_enabled and pre_processor in ["Lineart", "Lineart anime"]),  # tab_lineart_anime
            gr.update(visible=is_enabled and pre_processor == "Lineart standard", open=is_enabled and pre_processor == "Lineart standard")  # tab_linearts
        ]

    enable_canny.change(
        fn=toggle_canny_components,
        inputs=[enable_canny, canny_pre_processor],
        outputs=[
            canny_pre_processor,
            canny_input_image,
            canny_preview_button,
            canny_output_image,
            tab_canny,
            tab_lineart_anime,
            tab_linearts
        ]
    )
    
    def toggle_pose_components(is_enabled, pre_processor):
        return [
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(visible=is_enabled and pre_processor == "OpenPose", open=is_enabled and pre_processor == "OpenPose"),
            gr.update(visible=is_enabled and pre_processor == "DWPose", open=is_enabled and pre_processor == "DWPose")
        ]

    enable_pose.change(
        fn=toggle_pose_components,
        inputs=[enable_pose, pose_pre_processor],
        outputs=[
            pose_pre_processor,
            pose_input_image,
            pose_preview_button,
            pose_output_image,
            tab_openpose,
            tab_dwpose
        ]
    )
    
    def toggle_depth_components(is_enabled, pre_processor):
        return [
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(interactive=is_enabled),
            gr.update(visible=is_enabled and pre_processor == "HED", open=is_enabled and pre_processor == "HED"),
            gr.update(visible=is_enabled and pre_processor in ["depth-anything/Depth-Anything-V2-Large-hf", "depth-anything/Depth-Anything-V2-Base-hf", "depth-anything/Depth-Anything-V2-Small-hf", "Intel/zoedepth-nyu-kitti", "Intel/zoedepth-kitti", "Intel/zoedepth-nyu"], open=is_enabled and pre_processor in ["depth-anything/Depth-Anything-V2-Large-hf", "depth-anything/Depth-Anything-V2-Base-hf", "depth-anything/Depth-Anything-V2-Small-hf", "Intel/zoedepth-nyu-kitti", "Intel/zoedepth-kitti", "Intel/zoedepth-nyu"]),
        ]

    enable_depth.change(
        fn=toggle_depth_components,
        inputs=[enable_depth, depth_pre_processor],
        outputs=[
            depth_pre_processor,
            depth_input_image,
            depth_preview_button,
            depth_output_image,
            tab_hed,
            tab_depth,
        ]
    )
    toggle_pairs = [
        [enable_blur, blur_input_image],
        [enable_gray, gray_input_image],
        [enable_low_quality, low_quality_input_image]
    ]

    # Toggle function
    def toggle_component(is_enabled):
        return gr.update(interactive=is_enabled)

    # Register events in a loop
    for checkbox, image_input in toggle_pairs:
        checkbox.change(
            fn=toggle_component,
            inputs=[checkbox],
            outputs=[image_input]
        )

    
    def generate_canny_control_image(canny_input_image, pre_processor, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution, lineart_detect_resolution, lineart_image_resolution, linearts_detect_resolution, linearts_guassian_sigma, linearts_intensity_threshold):
        if not canny_input_image:
            return gr.update(value=None)

        image = load_image(canny_input_image)
        if pre_processor == "Canny":
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "CannyDetector":
                clear_controlnet_model_memory()
            processor = CannyDetector()
            control_image = processor(
                image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold, detect_resolution=canny_detect_resolution, image_resolution=canny_image_resolution
            )
            base_filename = "canny"
        elif pre_processor == "Lineart":
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "LineartDetector":
                clear_controlnet_model_memory()
            processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
            processor.to("cuda")
            control_image = processor(
                image, detect_resolution=lineart_detect_resolution, image_resolution=lineart_image_resolution
            )
            base_filename = "lineart"
        elif pre_processor == "Lineart standard":
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "LineartStandardDetector":
                clear_controlnet_model_memory()
            processor = LineartStandardDetector()
            control_image = processor(
                image, guassian_sigma=float(linearts_guassian_sigma), intensity_threshold=linearts_intensity_threshold, detect_resolution=linearts_detect_resolution
            )
            base_filename = "lineart_standard"
        elif pre_processor == "Lineart anime":
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "LineartAnimeDetector":
                clear_controlnet_model_memory()
            processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
            processor.to("cuda")
            control_image = processor(
                image, detect_resolution=lineart_detect_resolution, image_resolution=lineart_image_resolution
            )
            base_filename = "lineart_anime"
        modules.util.appstate.global_controlnet_model = processor
        filename = datetime.now().strftime(f"%Y%m%d_%H%M%S_{base_filename}.png")
        output_path = os.path.join(control_output_dir, filename)
        control_image.save(output_path)
        return gr.update(value=output_path)
    canny_preview_button.click(
        fn=generate_canny_control_image,
        inputs=[
            canny_input_image, canny_pre_processor, canny_low_threshold, canny_high_threshold, canny_detect_resolution, canny_image_resolution, lineart_detect_resolution, lineart_image_resolution, linearts_detect_resolution, linearts_guassian_sigma, linearts_intensity_threshold
        ],
        outputs=[canny_output_image]
    )

    def generate_depth_control_image(input_image, pre_processor, hed_depth_and_Normal, hed_bg_th, hed_detect_resolution, hed_image_resolution, depth_invert, depth_resolution_scale):
        if not input_image:
            return gr.update(value=None)

        image = load_image(input_image)
        if pre_processor == "HED":
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "MidasDetector":
                clear_controlnet_model_memory()
            processor = MidasDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
            control_image = processor( image, bg_th = float(hed_bg_th), depth_and_normal = hed_depth_and_Normal, detect_resolution=hed_detect_resolution, image_resolution=hed_image_resolution )
            base_filename = "HED"
        else:
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "DepthPreprocessor":
                clear_controlnet_model_memory()
            processor = DepthPreprocessor.from_pretrained(pre_processor).to("cuda")
            control_image = processor(image, invert=depth_invert, resolution_scale=depth_resolution_scale)[0]
            base_filename = "depth"
        modules.util.appstate.global_controlnet_model = processor
        filename = datetime.now().strftime(f"%Y%m%d_%H%M%S_{base_filename}.png")
        output_path = os.path.join(control_output_dir, filename)
        control_image.save(output_path)
        return gr.update(value=output_path)
    depth_preview_button.click(
        fn=generate_depth_control_image,
        inputs=[
            depth_input_image, depth_pre_processor, hed_depth_and_Normal, hed_bg_th, hed_detect_resolution, hed_image_resolution, depth_invert, depth_resolution_scale
        ],
        outputs=[depth_output_image]
    )

    def generate_pose_control_image(input_image, pre_processor, openpose_include_body, openpose_include_hand, openpose_include_face, openpose_detect_resolution, openpose_image_resolution, dwpose_include_hand, dwpose_include_face):
        if not input_image:
            return gr.update(value=None)

        image = load_image(input_image).convert("RGB")
        if pre_processor == "OpenPose":
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "OpenposeDetector":
                clear_controlnet_model_memory()
            processor = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
            control_image = processor( image, include_body=openpose_include_body, include_hand=openpose_include_hand, include_face=openpose_include_face, detect_resolution=openpose_detect_resolution, image_resolution=openpose_image_resolution )
            base_filename = "openpose"
        else:
            if modules.util.appstate.global_controlnet_model is not None and type(modules.util.appstate.global_controlnet_model).__name__ != "DWposeDetector":
                clear_controlnet_model_memory()
            from easy_dwpose import DWposeDetector
            processor = DWposeDetector("cuda")
            control_image = processor(image, include_hands=True, include_face=True)
            base_filename = "dwpose"
        modules.util.appstate.global_controlnet_model = processor
        filename = datetime.now().strftime(f"%Y%m%d_%H%M%S_{base_filename}.png")
        output_path = os.path.join(control_output_dir, filename)
        control_image.save(output_path)
        return gr.update(value=output_path)
    pose_preview_button.click(
        fn=generate_pose_control_image,
        inputs=[
            pose_input_image, pose_pre_processor, openpose_include_body, openpose_include_hand, openpose_include_face, openpose_detect_resolution, openpose_image_resolution, dwpose_include_hand, dwpose_include_face
        ],
        outputs=[pose_output_image]
    )


    def update_performance_checkboxes(no_opt, nunchaku, double_cache, teacache, clicked_option):
        new_no_opt = False
        new_nunchaku = False
        new_double_cache = False
        new_teacache = False
        if clicked_option == "no_optimization":
            new_no_opt = True
        elif clicked_option == "nunchaku_fp16":
            new_nunchaku = True
        elif clicked_option == "double_cache":
            new_double_cache = True
        elif clicked_option == "teacache":
            new_teacache = True
        return new_no_opt, new_nunchaku, new_double_cache, new_teacache
    
    flux_no_optimization.change(
        fn=lambda x, n, d, t: update_performance_checkboxes(x, n, d, t, "no_optimization") if x else (x, n, d, t),
        inputs=[flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache],
        outputs=[flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache]
    )
    
    flux_nunchaku_fp16.change(
        fn=lambda n, x, d, t: update_performance_checkboxes(x, n, d, t, "nunchaku_fp16") if n else (x, n, d, t),
        inputs=[flux_nunchaku_fp16, flux_no_optimization, flux_double_cache, flux_teacache],
        outputs=[flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache]
    )
    
    flux_double_cache.change(
        fn=lambda d, x, n, t: update_performance_checkboxes(x, n, d, t, "double_cache") if d else (x, n, d, t),
        inputs=[flux_double_cache, flux_no_optimization, flux_nunchaku_fp16, flux_teacache],
        outputs=[flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache]
    )
    
    flux_teacache.change(
        fn=lambda t, x, n, d: update_performance_checkboxes(x, n, d, t, "teacache") if t else (x, n, d, t),
        inputs=[flux_teacache, flux_no_optimization, flux_nunchaku_fp16, flux_double_cache],
        outputs=[flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache]
    )

    def save_current_state(memory_optimization, width, height, guidance_scale, inference_steps,  
        no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, tea_cache_thresh):
       
        performance_optimization = "no_optimization"
        if nunchaku:
            performance_optimization = "nunchaku_fp16"
        elif double_cache:
            performance_optimization = "double_cache"
        elif teacache:
            performance_optimization = "teacache"
        
        state_dict = {
            "memory_optimization": memory_optimization,
            "performance_optimization": performance_optimization,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "diff_threshold_multi": diff_multi,
            "diff_threshold_single": diff_single,
            "tea_cache_threshold": tea_cache_thresh,
        }
        
        state_manager.save_state("flux-controlnet_union_pro", state_dict)
        return (memory_optimization, width, height, guidance_scale, inference_steps,  
                no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, 
                tea_cache_thresh)

    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flux_memory_optimization, 
            flux_width_input, 
            flux_height_input, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input,
            flux_no_optimization,
            flux_nunchaku_fp16,
            flux_double_cache,
            flux_teacache,
            flux_diff_multi,
            flux_diff_single,
            flux_tea_cache_l1_thresh_slider
        ],
        outputs=[
            flux_memory_optimization, 
            flux_width_input, 
            flux_height_input, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input,
            flux_no_optimization,
            flux_nunchaku_fp16,
            flux_double_cache,
            flux_teacache,
            flux_diff_multi,
            flux_diff_single,
            flux_tea_cache_l1_thresh_slider
        ]
    )
    def generate_with_performance_opts(
        seed, prompt, width, height, guidance_scale_slider, num_inference_steps,
        memory_optimization, no_of_images, randomize_seed, 
        no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, 
        tea_cache_thresh,
        enable_canny, canny_output_image, canny_controlnet_conditioning_scale,
        enable_depth, depth_output_image, depth_controlnet_conditioning_scale,
        enable_pose, pose_output_image, pose_controlnet_conditioning_scale,
        enable_blur, blur_input_image, blur_controlnet_conditioning_scale,
        enable_gray, gray_input_image, gray_controlnet_conditioning_scale,
        enable_low_quality, low_quality_input_image, low_quality_controlnet_conditioning_scale
    ):
     
        if nunchaku:
            performance_optimization = "nunchaku_fp16"
        elif double_cache:
            performance_optimization = "double_cache"
        elif teacache:
            performance_optimization = "teacache"
        else:
            performance_optimization = "no_optimization"

        control_image = []
        control_mode = []
        controlnet_conditioning_scale = []
        if enable_blur and blur_input_image is not None:
            control_image_blur = load_image(blur_input_image)
            control_image.append(control_image_blur)
            control_mode.append(3)
            controlnet_conditioning_scale.append(float(blur_controlnet_conditioning_scale))
        if enable_gray and gray_input_image is not None:
            control_image_gray = load_image(gray_input_image)
            control_image.append(control_image_gray)
            control_mode.append(5)
            controlnet_conditioning_scale.append(float(gray_controlnet_conditioning_scale))
        if enable_low_quality and low_quality_input_image is not None:
            control_image_low_quality = load_image(low_quality_input_image)
            control_image.append(control_image_low_quality)
            control_mode.append(6)
            controlnet_conditioning_scale.append(float(low_quality_controlnet_conditioning_scale))
        if enable_canny and canny_output_image is not None:
            control_image_canny = load_image(canny_output_image)
            control_image.append(control_image_canny)
            control_mode.append(0)
            controlnet_conditioning_scale.append(float(canny_controlnet_conditioning_scale))
        if enable_pose and pose_output_image is not None:
            control_image_pose = load_image(pose_output_image)
            control_image.append(control_image_pose)
            control_mode.append(4)
            controlnet_conditioning_scale.append(float(pose_controlnet_conditioning_scale))
        if enable_depth and depth_output_image is not None:
            control_image_depth = load_image(depth_output_image)
            control_image.append(control_image_depth)
            control_mode.append(2)
            controlnet_conditioning_scale.append(float(depth_controlnet_conditioning_scale))

        return generate_images(
            seed, prompt, width, height, guidance_scale_slider, num_inference_steps,
            memory_optimization, no_of_images, randomize_seed, 
            performance_optimization, diff_multi, diff_single, tea_cache_thresh,
            control_image, control_mode, controlnet_conditioning_scale
        )
    generate_button.click(
        fn=generate_with_performance_opts,
        inputs=[
            seed_input, flux_prompt_input, flux_width_input, flux_height_input, 
            flux_guidance_scale_slider,  flux_num_inference_steps_input,
            flux_memory_optimization, flux_no_of_images_input, flux_randomize_seed, 
            flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache, flux_diff_multi, flux_diff_single, 
            flux_tea_cache_l1_thresh_slider,
            enable_canny, canny_output_image, canny_controlnet_conditioning_scale,
            enable_depth, depth_output_image, depth_controlnet_conditioning_scale,
            enable_pose, pose_output_image, pose_controlnet_conditioning_scale,
            enable_blur, blur_input_image, blur_controlnet_conditioning_scale,
            enable_gray, gray_input_image, gray_controlnet_conditioning_scale,
            enable_low_quality, low_quality_input_image, low_quality_controlnet_conditioning_scale
        ],
        outputs=[output_gallery]
    )