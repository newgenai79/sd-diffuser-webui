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
from controlnet_aux import CannyDetector
from image_gen_aux import DepthPreprocessor
from diffusers import FluxControlPipeline, FluxFillPipeline
from diffusers.utils import load_image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager
from nunchaku.utils import get_precision
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.caching.teacache import TeaCache
from contextlib import nullcontext
from nunchaku.lora.flux.compose import compose_lora

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"
lora_path = "models/lora/flux.1-dev-canny/"

def get_lora_files():
    lora_files = []
    if os.path.exists(lora_path):
        for file in os.listdir(lora_path):
            if file.endswith(".safetensors"):
                lora_files.append(file)
    return lora_files

def refresh_lora_list():
    lora_files = get_lora_files()
    return gr.update(choices=lora_files)

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, inference_type, performance_optimization, diff_multi, diff_single):
    print("----FluxPipeline mode: ", memory_optimization, inference_type, performance_optimization, diff_multi, diff_single)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        (type(modules.util.appstate.global_pipe).__name__ == "FluxControlPipeline" or type(modules.util.appstate.global_pipe).__name__ == "FluxFillPipeline") and 
            modules.util.appstate.global_inference_type == inference_type and 
            modules.util.appstate.global_performance_optimization == performance_optimization and
            modules.util.appstate.global_memory_mode == memory_optimization):
                print(">>>>Reusing FluxPipeline pipe<<<<")
                return modules.util.appstate.global_pipe
    else:
        clear_previous_model_memory()
        torch.cuda.synchronize()

    dtype = torch.bfloat16
    precision = get_precision()
    
    if(inference_type == "Canny"):
        transformer_repo = f"mit-han-lab/nunchaku-flux.1-canny-dev/svdq-{precision}_r32-flux.1-canny-dev.safetensors"    
        flux_repo = "black-forest-labs/FLUX.1-Canny-dev"
        class_name = FluxControlPipeline
    elif(inference_type == "Depth"):
        transformer_repo = f"mit-han-lab/nunchaku-flux.1-depth-dev/svdq-{precision}_r32-flux.1-depth-dev.safetensors"    
        flux_repo = "black-forest-labs/FLUX.1-Depth-dev"
        class_name = FluxControlPipeline
    elif(inference_type == "Fill"):
        transformer_repo = f"mit-han-lab/nunchaku-flux.1-fill-dev/svdq-{precision}_r32-flux.1-fill-dev.safetensors"    
        flux_repo = "black-forest-labs/FLUX.1-Fill-dev"
        class_name = FluxFillPipeline
    else:
        transformer_repo = ""
        flux_repo = ""
        class_name = None
    
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
    modules.util.appstate.global_pipe = class_name.from_pretrained(
        flux_repo, 
        transformer=transformer,
        text_encoder_2=text_encoder_2,
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
    modules.util.appstate.global_transformer = transformer
    return modules.util.appstate.global_pipe

def generate_images(
    seed, prompt, width, height, guidance_scale, num_inference_steps,
    memory_optimization, no_of_images, randomize_seed, input_image, 
    low_threshold, high_threshold, detect_resolution, image_resolution,
    performance_optimization, diff_multi, diff_single, tea_cache_thresh,
    inference_type, mask_image
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    
    gallery_items = []

    try:

        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, inference_type, performance_optimization, diff_multi, diff_single)

        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx+1}: (Step {i}/{num_inference_steps})")
            return callback_kwargs
        guidance_scale = float(guidance_scale)
        modules.util.appstate.global_inference_in_progress = True


        if(inference_type == "Canny"):
            control_image = load_image(input_image)
            processor = CannyDetector()
            control_image = processor(
                control_image, 
                low_threshold=low_threshold, 
                high_threshold=high_threshold, 
                detect_resolution=detect_resolution, 
                image_resolution=image_resolution
            )
            base_filename = "canny"
        elif(inference_type == "Depth"):
            control_image = load_image(input_image)
            processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
            control_image = processor(control_image)[0].convert("RGB")
            base_filename = "depth"
        elif(inference_type == "Fill"):
            base_filename = "fill"

        # Generate multiple images in a loop
        for img_idx in range(no_of_images):
            
            # If randomize_seed is True, generate a new random seed for each image
            current_seed = random_seed() if randomize_seed else seed
            
            # Create generator with the current seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            # Prepare inference parameters

            inference_params = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if(inference_type == "Fill"):
                inference_params["image"] = input_image
                inference_params["mask_image"] = mask_image
            else:
                inference_params["control_image"] = control_image

            if performance_optimization != "teacache":
                inference_params["num_inference_steps"] = num_inference_steps

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
            
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Create filename with index
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_flux_{base_filename}_{img_idx+1}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            metadata = {
                "model": inference_type,
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

        modules.util.appstate.global_inference_in_progress = False
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        if inference_type in ["Canny", "Depth"]:
            del control_image
            del processor
            gc.collect()
        modules.util.appstate.global_inference_in_progress = False

def create_flux_canny_tab():
    gr.HTML("<style>.small-button { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; align-self: end; line-height: 1em; border-radius: 0.5em; }</style>", visible=False)
    initial_state = state_manager.get_state("flux-canny_depth") or {}
    selected_perf_opt = initial_state.get("performance_optimization", "no_optimization")
    with gr.Row():
        with gr.Accordion("Inference type / Optimizations", open=True):
            with gr.Row():
                flux_inference_type = gr.Radio(
                    choices=["Canny", "Depth", "Fill"],
                    label="Inference type",
                    value=initial_state.get("inference_type", "Canny"),
                    interactive=True
                )
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
        with gr.Column():
            with gr.Row():
                flux_input_image = gr.Image(label="Input Image", type="pil", width=512, height=512)
            with gr.Row():
                flux_low_threshold_slider = gr.Slider(
                    label="Low threshold", 
                    minimum=0, 
                    maximum=200, 
                    value=50,
                    step=1,
                    interactive=True,
                    visible = initial_state.get("inference_type", "Canny") == "Canny"
                )
            with gr.Row():
                flux_high_threshold_slider = gr.Slider(
                    label="High threshold", 
                    minimum=0, 
                    maximum=200, 
                    value=200,
                    step=1,
                    interactive=True,
                    visible = initial_state.get("inference_type", "Canny") == "Canny"
                )
            with gr.Row():
                flux_detect_resolution_input = gr.Number(
                    label="Detect resolution", 
                    value=1024,
                    interactive=True,
                    visible = initial_state.get("inference_type", "Canny") == "Canny"
                )
                flux_image_resolution_input = gr.Number(
                    label="Image resolution", 
                    value=1024,
                    interactive=True,
                    visible = initial_state.get("inference_type", "Canny") == "Canny"
                )
            with gr.Row():
                flux_mask_image = gr.Image(
                    label="Mask Image", 
                    type="pil",
                    width=512,
                    height=512,
                    visible = initial_state.get("inference_type", "Canny") == "Fill"
                )
        with gr.Column():
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
                    value=initial_state.get("guidance_scale", 30),
                    step=0.1,
                    interactive=True
                )
                flux_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 50),
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
        no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, tea_cache_thresh, inference_type):
       
        performance_optimization = "no_optimization"
        if nunchaku:
            performance_optimization = "nunchaku_fp16"
        elif double_cache:
            performance_optimization = "double_cache"
        elif teacache:
            performance_optimization = "teacache"
        
        state_dict = {
            "inference_type":inference_type, 
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
        
        state_manager.save_state("flux-canny_depth", state_dict)
        return (memory_optimization, width, height, guidance_scale, inference_steps,  
                no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, 
                tea_cache_thresh, inference_type)

    def toggle_visibility(inference_type):
        return gr.update(visible=(inference_type == "Canny")), gr.update(visible=(inference_type == "Canny")), gr.update(visible=(inference_type == "Canny")), gr.update(visible=(inference_type == "Canny")), gr.update(visible=(inference_type == "Fill"))
    flux_inference_type.change(toggle_visibility, inputs=[flux_inference_type], outputs=[flux_low_threshold_slider, flux_high_threshold_slider, flux_detect_resolution_input, flux_image_resolution_input, flux_mask_image])

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
            flux_tea_cache_l1_thresh_slider,
            flux_inference_type
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
            flux_tea_cache_l1_thresh_slider,
            flux_inference_type
        ]
    )
    def generate_with_performance_opts(
        seed, prompt, width, height, guidance_scale_slider, num_inference_steps,
        memory_optimization, no_of_images, randomize_seed, 
        input_image, low_threshold_slider, high_threshold_slider, 
        detect_resolution, image_resolution,
        no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, 
        tea_cache_thresh, inference_type, mask_image
    ):
     
        if nunchaku:
            performance_optimization = "nunchaku_fp16"
        elif double_cache:
            performance_optimization = "double_cache"
        elif teacache:
            performance_optimization = "teacache"
        else:
            performance_optimization = "no_optimization"
        
        return generate_images(
            seed, prompt, width, height, guidance_scale_slider, num_inference_steps,
            memory_optimization, no_of_images, randomize_seed, 
            input_image, low_threshold_slider, high_threshold_slider, 
            detect_resolution, image_resolution,
            performance_optimization, diff_multi, diff_single, tea_cache_thresh,
            inference_type, mask_image
        )
    generate_button.click(
        fn=generate_with_performance_opts,
        inputs=[
            seed_input, flux_prompt_input, flux_width_input, flux_height_input, 
            flux_guidance_scale_slider,  flux_num_inference_steps_input,
            flux_memory_optimization, flux_no_of_images_input, flux_randomize_seed, 
            flux_input_image, flux_low_threshold_slider, flux_high_threshold_slider, 
            flux_detect_resolution_input, flux_image_resolution_input,
            flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache, flux_diff_multi, flux_diff_single, 
            flux_tea_cache_l1_thresh_slider, flux_inference_type, flux_mask_image
        ],
        outputs=[output_gallery]
    )