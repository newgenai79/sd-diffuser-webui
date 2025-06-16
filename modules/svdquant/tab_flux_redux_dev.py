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
from nunchaku import NunchakuFluxTransformer2dModel
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image
from modules.util.utilities import clear_previous_model_memory
from modules.util.appstate import state_manager
from nunchaku.utils import get_precision
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.caching.teacache import TeaCache
from contextlib import nullcontext

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"

def random_seed():
    return torch.randint(0, MAX_SEED, (1,)).item()

def get_pipeline(memory_optimization, inference_type, performance_optimization, diff_multi, diff_single):
    print("----FluxPipeline mode: ", memory_optimization, inference_type, performance_optimization, diff_multi, diff_single)
    # If model is already loaded with same configuration, reuse it
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "FluxPipeline" and 
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
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
    )
    if performance_optimization == "nunchaku-fp16":
        transformer.set_attention_impl("nunchaku-fp16")
    modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
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
    return modules.util.appstate.global_pipe

def generate_images(
    seed, guidance_scale, num_inference_steps, 
    memory_optimization, no_of_images, randomize_seed, input_image,
    performance_optimization, diff_multi, diff_single, tea_cache_thresh
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    
    gallery_items = []

    try:
        # Get pipeline (either cached or newly loaded)
        pipe = get_pipeline(memory_optimization, "flux.1-dev-redux", performance_optimization, diff_multi, diff_single)
        progress_bar = gr.Progress(track_tqdm=True)
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx+1}: (Step {i}/{num_inference_steps})")
            return callback_kwargs
        guidance_scale = float(guidance_scale)
        modules.util.appstate.global_inference_in_progress = True
        input_image = load_image(input_image)
        pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
        ).to("cuda")
        pipe_prior_output = pipe_prior_redux(input_image)
        
        del pipe_prior_redux
        pipe_prior_redux = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        # Generate multiple images in a loop
        for img_idx in range(no_of_images):
           
            # If randomize_seed is True, generate a new random seed for each image
            current_seed = random_seed() if randomize_seed else seed
            
            # Create generator with the current seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            
            start_time = datetime.now()
            inference_params = {
                "guidance_scale": guidance_scale, 
                **pipe_prior_output
            }
            if performance_optimization != "teacache":
                inference_params["num_inference_steps"] = num_inference_steps
            # Generate image
            with (
                TeaCache(model=pipe.transformer, num_steps=num_inference_steps, rel_l1_thresh=float(tea_cache_thresh), enabled=True)
                if performance_optimization == "teacache" else nullcontext()
            ):
                image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Create filename with index
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_flux_redux_{img_idx+1}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            metadata = {
                "model": "FLUX.1-dev-redux",
                "seed": current_seed,
                "guidance_scale": f"{float(guidance_scale):.2f}",
                "num_inference_steps": num_inference_steps,
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
            gallery_items.append((output_path, "FLUX.1-dev-redux"))
        modules.util.appstate.global_inference_in_progress = False
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        del pipe_prior_redux
        del pipe_prior_output
        gc.collect()
        modules.util.appstate.global_inference_in_progress = False

def create_flux_redux_tab():
    initial_state = state_manager.get_state("flux-redux") or {}
    selected_perf_opt = initial_state.get("performance_optimization", "no_optimization")
    with gr.Row():
        with gr.Accordion("Select model / Optimization", open=True):
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
                    flux_tea_cache_l1_thresh = gr.Slider(
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
                flux_input_image = gr.Image(label="Input Image", type="pil")
        with gr.Column():
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("♻️ Randomize seed")
            with gr.Row():
                flux_guidance_scale = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=50.0, 
                    value=initial_state.get("guidance_scale", 10),
                    step=0.1,
                    interactive=True
                )
                flux_num_inference_steps_input = gr.Number(
                    label="Number of Inference Steps", 
                    value=initial_state.get("inference_steps", 30),
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
    def save_current_state(memory_optimization, guidance_scale, inference_steps, 
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
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "performance_optimization": performance_optimization,
            "diff_threshold_multi": diff_multi,
            "diff_threshold_single": diff_single,
            "tea_cache_threshold": tea_cache_thresh,
        }
        # print("Saving state:", state_dict)
        initial_state = state_manager.get_state("flux-redux") or {}
        state_manager.save_state("flux-redux", state_dict)
        return memory_optimization, guidance_scale, inference_steps, no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, tea_cache_thresh
    
    # Event handlers
    random_button.click(fn=random_seed, outputs=[seed_input])
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flux_memory_optimization, 
            flux_guidance_scale, 
            flux_num_inference_steps_input,
            flux_no_optimization,
            flux_nunchaku_fp16,
            flux_double_cache,
            flux_teacache,
            flux_diff_multi,
            flux_diff_single,
            flux_tea_cache_l1_thresh,
        ],
        outputs=[
            flux_memory_optimization, 
            flux_guidance_scale, 
            flux_num_inference_steps_input,
            flux_no_optimization,
            flux_nunchaku_fp16,
            flux_double_cache,
            flux_teacache,
            flux_diff_multi,
            flux_diff_single,
            flux_tea_cache_l1_thresh,
        ]
    )
    def generate_with_performance_opts(
        seed_input, flux_guidance_scale, 
        flux_num_inference_steps_input, flux_memory_optimization, 
        flux_no_of_images_input, flux_randomize_seed, flux_input_image,
        no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, tea_cache_thresh
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
            seed_input, flux_guidance_scale, 
            flux_num_inference_steps_input, flux_memory_optimization, 
            flux_no_of_images_input, flux_randomize_seed, flux_input_image,
            performance_optimization, diff_multi, diff_single, tea_cache_thresh
        )
    generate_button.click(
        fn=generate_with_performance_opts,
        inputs=[
            seed_input, flux_guidance_scale, 
            flux_num_inference_steps_input, flux_memory_optimization, 
            flux_no_of_images_input, flux_randomize_seed, flux_input_image,
            flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache,
            flux_diff_multi, flux_diff_single, flux_tea_cache_l1_thresh
        ],
        outputs=[output_gallery]
    )