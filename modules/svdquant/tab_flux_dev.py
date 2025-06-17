"""
Copyright NewGenAI
Do not remove this copyright. No derivative code allowed.
"""
import torch
import gradio as gr
import numpy as np
import os
import re
import gc
import modules.util.appstate
from datetime import datetime
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel
from nunchaku.utils import get_precision
from modules.util.utilities import clear_previous_model_memory, pad_tensors_to_equal_length, clear_text_model_memory
from modules.util.appstate import state_manager
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.caching.teacache import TeaCache
from contextlib import nullcontext
from nunchaku.lora.flux.compose import compose_lora

MAX_SEED = np.iinfo(np.int32).max
OUTPUT_DIR = "output/t2i/Flux"
lora_path = "models/lora/flux.1-dev/"

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

def get_pipeline(model_type, memory_optimization, selected_loras, bypass_token_limit, performance_optimization, diff_multi, diff_single):
    print(f"----{model_type} mode: ", model_type, memory_optimization, selected_loras, bypass_token_limit, performance_optimization, diff_multi, diff_single)
    if (modules.util.appstate.global_pipe is not None and 
        type(modules.util.appstate.global_pipe).__name__ == "FluxPipeline" and 
        modules.util.appstate.global_model_type == model_type and
        modules.util.appstate.global_bypass_token_limit == bypass_token_limit and 
        modules.util.appstate.global_performance_optimization == performance_optimization and
        modules.util.appstate.global_memory_mode == memory_optimization):
        print(">>>>Reusing Flux pipe<<<<")
        if selected_loras != modules.util.appstate.global_selected_lora and hasattr(modules.util.appstate.global_pipe.transformer, "reset_lora"):
            print("Unloading lora....")
            modules.util.appstate.global_pipe.transformer.reset_lora()
            modules.util.appstate.global_selected_lora = None
            if selected_loras:
                print("Loading lora....")
                composed_lora = compose_lora(selected_loras)
                modules.util.appstate.global_pipe.transformer.update_lora_params(composed_lora)
                modules.util.appstate.global_selected_lora = selected_loras
        elif not selected_loras and modules.util.appstate.global_selected_lora:
            print("Unloading lora (no selection)....")
            modules.util.appstate.global_pipe.transformer.reset_lora()
            modules.util.appstate.global_selected_lora = None
        if bypass_token_limit:
            return modules.util.appstate.global_pipe, modules.util.appstate.global_text_encoder_2
        else:
            return modules.util.appstate.global_pipe, None
    else:
        clear_previous_model_memory()
        torch.cuda.synchronize()
    precision = get_precision()
    if model_type == "FLUX.1-dev":
        transformer_repo = f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors"
        bfl_repo = "black-forest-labs/FLUX.1-dev"
    elif model_type == "FLUX.1-schnell":
        transformer_repo = f"mit-han-lab/nunchaku-flux.1-schnell/svdq-{precision}_r32-flux.1-schnell.safetensors"
        bfl_repo = "black-forest-labs/FLUX.1-schnell"
    elif model_type == "Shuttle-jaguar":
        transformer_repo = f"mit-han-lab/nunchaku-shuttle-jaguar/svdq-{precision}_r32-shuttle-jaguar.safetensors"
        bfl_repo = "shuttleai/shuttle-jaguar"       

    text_encoder_2_repo = f"mit-han-lab/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors"

    dtype = torch.bfloat16

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        transformer_repo, 
        offload=True,
    )
    if performance_optimization == "nunchaku-fp16":
        transformer.set_attention_impl("nunchaku-fp16")

    text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
        text_encoder_2_repo,
    )

    if bypass_token_limit:
        modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
            bfl_repo, 
            text_encoder=None,
            text_encoder_2=None,
            tokenizer=None,
            tokenizer_2=None,
            transformer=transformer, 
            torch_dtype=torch.bfloat16,
        )
    else:
        modules.util.appstate.global_pipe = FluxPipeline.from_pretrained(
            bfl_repo, 
            text_encoder_2=text_encoder_2, 
            transformer=transformer, 
            torch_dtype=torch.bfloat16,
        )

    if selected_loras:
        print("Loading lora....")
        composed_lora = compose_lora(selected_loras)
        modules.util.appstate.global_pipe.transformer.update_lora_params(composed_lora)
        modules.util.appstate.global_selected_lora = selected_loras
    else:
        modules.util.appstate.global_selected_lora = None

    if performance_optimization == "double_cache":
        apply_cache_on_pipe(
            modules.util.appstate.global_pipe,
            use_double_fb_cache=True,
            residual_diff_threshold_multi=float(diff_multi),
            residual_diff_threshold_single=float(diff_single),
        )
    if memory_optimization == "Extremely Low VRAM":
        modules.util.appstate.global_pipe.enable_sequential_cpu_offload()
    elif memory_optimization == "Low VRAM":
        modules.util.appstate.global_pipe.enable_model_cpu_offload()
    else:
        modules.util.appstate.global_pipe.to("cuda")
        
    modules.util.appstate.global_memory_mode = memory_optimization
    modules.util.appstate.global_model_type = model_type
    modules.util.appstate.global_bypass_token_limit = bypass_token_limit
    modules.util.appstate.global_text_encoder_2 = text_encoder_2
    modules.util.appstate.global_transformer = transformer
    modules.util.appstate.global_performance_optimization = performance_optimization
    if bypass_token_limit:
        return modules.util.appstate.global_pipe, modules.util.appstate.global_text_encoder_2
    else:
        return modules.util.appstate.global_pipe, None

def generate_images(
    seed, prompt, width, height, guidance_scale,
    num_inference_steps, memory_optimization, 
    no_of_images, randomize_seed, lora_selections, 
    bypass_token_limit, negative_prompt,
    performance_optimization, diff_multi, diff_single, tea_cache_thresh, model_type
):
    if modules.util.appstate.global_inference_in_progress == True:
        print(">>>>Inference in progress, can't continue<<<<")
        return None
    # model_type = "FLUX.1-dev"
    gallery_items = []
    # Create composed_lora from selected LoRAs and weights
    selected_loras = [(os.path.join(lora_path, lora), float(weight)) for lora, weight in lora_selections.items() if lora]
    
    try:
        pipe, text_encoder_2 = get_pipeline(model_type, memory_optimization, selected_loras, bypass_token_limit, performance_optimization, diff_multi, diff_single)
        print("Pipeline ready for inference.....")
        progress_bar = gr.Progress(track_tqdm=True)
        
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            progress_bar(i / num_inference_steps, desc=f"Generating image {img_idx+1}: (Step {i}/{num_inference_steps})")
            return callback_kwargs
        guidance_scale = float(guidance_scale)
        
        if bypass_token_limit:
            if model_type == "FLUX.1-dev":
                bfl_repo = "black-forest-labs/FLUX.1-dev"
            elif model_type == "FLUX.1-schnell":
                bfl_repo = "black-forest-labs/FLUX.1-schnell"
            elif model_type == "Shuttle-jaguar":
                bfl_repo = "shuttleai/shuttle-jaguar"
            text_pipeline = FluxPipeline.from_pretrained(
                bfl_repo,
                text_encoder_2=text_encoder_2,
                transformer=None,
                vae=None,
                torch_dtype=torch.bfloat16,
            )
            text_pipeline.enable_model_cpu_offload()
            from sd_embed.embedding_funcs import get_weighted_text_embeddings_flux1
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
                    pipe=text_pipeline,
                    prompt=prompt,
                )
                if negative_prompt.strip():
                    negative_prompt_embeds, negative_pooled_prompt_embeds = get_weighted_text_embeddings_flux1(
                        pipe=text_pipeline,
                        prompt=negative_prompt,
                    )
            if negative_prompt.strip():
                prompt_embeds, negative_prompt_embeds = pad_tensors_to_equal_length(
                    prompt_embeds, negative_prompt_embeds
                )
            clear_text_model_memory(text_pipeline)
        for img_idx in range(no_of_images):
            modules.util.appstate.global_inference_in_progress = True
            current_seed = random_seed() if randomize_seed else seed
            generator = torch.Generator(device="cpu").manual_seed(current_seed)
            inference_params = {
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "callback_on_step_end": callback_on_step_end,
            }
            if performance_optimization != "teacache":
                inference_params["num_inference_steps"] = num_inference_steps
            if bypass_token_limit:
                inference_params["prompt_embeds"] = prompt_embeds
                inference_params["pooled_prompt_embeds"] = pooled_prompt_embeds
                if negative_prompt.strip():
                    inference_params["negative_prompt_embeds"] = negative_prompt_embeds
                    inference_params["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            else:
                inference_params["prompt"] = prompt
                inference_params["negative_prompt"] = negative_prompt
            print(f"Generating image {img_idx+1}/{no_of_images} with seed: {current_seed}")
            start_time = datetime.now()
            with (
                TeaCache(model=modules.util.appstate.global_transformer, num_steps=num_inference_steps, rel_l1_thresh=float(tea_cache_thresh), enabled=True)
                if performance_optimization == "teacache" else nullcontext()
            ):
                image = pipe(**inference_params).images[0]
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if model_type == "FLUX.1-dev":
                base_filename = "dev"
            elif model_type == "FLUX.1-schnell":
                base_filename = "schnell"
            elif model_type == "Shuttle-jaguar":
                base_filename = "shuttle_jaguar"
            filename = f"{timestamp}_flux_{base_filename}_{img_idx+1}.png"
            output_path = os.path.join(OUTPUT_DIR, filename)
            metadata = {
                "model": model_type,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
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
            if selected_loras:
                metadata["loras"] = [lora for lora, _ in selected_loras]
                metadata["lora_weights"] = [f"{float(weight):.2f}" for _, weight in selected_loras]
            image.save(output_path)
            modules.util.utilities.save_metadata_to_file(output_path, metadata)
            print(f"Image {img_idx+1}/{no_of_images} generated: {output_path}")
            gallery_items.append((output_path, model_type))
            modules.util.appstate.global_inference_in_progress = False
        return gallery_items
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None
    finally:
        if bypass_token_limit:
            del prompt_embeds, pooled_prompt_embeds
            if negative_prompt.strip():
                del negative_prompt_embeds, negative_pooled_prompt_embeds
            gc.collect()
            torch.cuda.empty_cache()
        modules.util.appstate.global_inference_in_progress = False

def create_flux_dev_tab():
    gr.HTML("<style>.small-button { max-width: 2.2em; min-width: 2.2em !important; height: 2.4em; align-self: end; line-height: 1em; border-radius: 0.5em; }</style>", visible=False)
    initial_state = state_manager.get_state("flux-dev_schnell") or {}
    selected_perf_opt = initial_state.get("performance_optimization", "no_optimization")
    
    with gr.Row():
        with gr.Accordion("Select model / Optimization", open=True):
            with gr.Row():
                flux_model_type = gr.Dropdown(
                    choices=["FLUX.1-dev", "FLUX.1-schnell", "Shuttle-jaguar"],
                    value=initial_state.get("model_type", "FLUX.1-dev"),
                    label="Select model",
                    interactive=True,
                    visible=True
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
        flux_bypass_token_limit = gr.Checkbox(label="Bypass 77 tokens limit (consumes more VRAM during text encoding) (DO NOT USE WITH SMALL PROMPTS)", value=initial_state.get("bypass_token_limit", False), interactive=True)
    with gr.Row():
        with gr.Column():
            flux_prompt_input = gr.Textbox(
                label="Prompt (support max 77 tokens)", 
                lines=6,
                interactive=True
            )
        with gr.Column():
            flux_negative_prompt_input = gr.Textbox(
                label="Negative prompt (support max 77 tokens)", 
                lines=6,
                interactive=True
            )
    with gr.Accordion("LoRA Selection (Max 3). New lora addition require app to be restarted.", open=False):
        """
        with gr.Row():
            gr.Markdown("### LoRA Selection (Max 3)")
            lora_refresh_btn = gr.Button("🔄", elem_classes="small-button")
        """
        lora_checkboxes = {}
        lora_weights = {}
        lora_files = get_lora_files()
        lora_state = initial_state.get("lora_selections", {})
        num_files = len(lora_files)
        col_size = (num_files + 2) // 3  # Ceiling division to distribute files

        with gr.Row():
            for col in range(3):
                with gr.Column():
                    start_idx = col * col_size
                    end_idx = min(start_idx + col_size, num_files)
                    for i in range(start_idx, end_idx):
                        lora_file = lora_files[i]
                        with gr.Row():
                            lora_checkboxes[lora_file] = gr.Checkbox(
                                label=lora_file,
                                value=lora_state.get(lora_file, {}).get("selected", False),
                                interactive=True,
                                scale=7
                            )
                            lora_weights[lora_file] = gr.Textbox(
                                value=lora_state.get(lora_file, {}).get("weight", "1.0"),
                                show_label=False,
                                interactive=True,
                                scale=1
                            )
        
    with gr.Row():
        with gr.Column():
            with gr.Row():
                flux_width_input = gr.Number(
                    label="Width", 
                    value=initial_state.get("width", 1072),
                    interactive=True
                )
                flux_height_input = gr.Number(
                    label="Height", 
                    value=initial_state.get("height", 1920),
                    interactive=True
                )
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, minimum=0, maximum=MAX_SEED, interactive=True)
                random_button = gr.Button("🔄", elem_classes="small-button")
        with gr.Column():
            with gr.Row():
                flux_guidance_scale_slider = gr.Slider(
                    label="Guidance Scale", 
                    minimum=1.0, 
                    maximum=20.0, 
                    value=initial_state.get("guidance_scale", 3.5),
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

    def limit_lora_selection(*checkboxes):
        selected_count = sum(1 for cb in checkboxes if cb)
        if selected_count > 3:
            return [False if i >= 3 else cb for i, cb in enumerate(checkboxes)]
        return checkboxes

    for lora_file in lora_files:
        lora_checkboxes[lora_file].change(
            fn=limit_lora_selection,
            inputs=list(lora_checkboxes.values()),
            outputs=list(lora_checkboxes.values())
        )
    """
    lora_refresh_btn.click(
        fn=lambda: [gr.Checkbox(label=f, value=False, interactive=True) for f in get_lora_files()],
        outputs=list(lora_checkboxes.values())
    )
    """
    def save_current_state(memory_optimization, width, height, guidance_scale, inference_steps, bypass_token_limit, 
                          no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, tea_cache_thresh, model_type, *lora_inputs):
        # Split lora_inputs into checkboxes and weights
        lora_files = get_lora_files()
        lora_checkboxes_vals = lora_inputs[:len(lora_files)]
        lora_weights_vals = lora_inputs[len(lora_files):]
        lora_selections = {lora_file: {"selected": lora_checkboxes_vals[i], "weight": lora_weights_vals[i]} for i, lora_file in enumerate(lora_files)}
        
        performance_optimization = "no_optimization"
        if nunchaku:
            performance_optimization = "nunchaku_fp16"
        elif double_cache:
            performance_optimization = "double_cache"
        elif teacache:
            performance_optimization = "teacache"
        
        state_dict = {
            "model_type": model_type,
            "memory_optimization": memory_optimization,
            "performance_optimization": performance_optimization,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
            "bypass_token_limit": bypass_token_limit,
            "diff_threshold_multi": diff_multi,
            "diff_threshold_single": diff_single,
            "tea_cache_threshold": tea_cache_thresh,
            "lora_selections": lora_selections
        }
        
        state_manager.save_state("flux-dev_schnell", state_dict)
        return (memory_optimization, width, height, guidance_scale, inference_steps, bypass_token_limit, 
                no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, tea_cache_thresh, model_type) + lora_inputs

    random_button.click(fn=random_seed, outputs=[seed_input])
    
    save_state_button.click(
        fn=save_current_state,
        inputs=[
            flux_memory_optimization, 
            flux_width_input, 
            flux_height_input, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input,
            flux_bypass_token_limit,
            flux_no_optimization,
            flux_nunchaku_fp16,
            flux_double_cache,
            flux_teacache,
            flux_diff_multi,
            flux_diff_single,
            flux_tea_cache_l1_thresh_slider,
            flux_model_type,
            *lora_checkboxes.values(),
            *lora_weights.values()
        ],
        outputs=[
            flux_memory_optimization, 
            flux_width_input, 
            flux_height_input, 
            flux_guidance_scale_slider, 
            flux_num_inference_steps_input,
            flux_bypass_token_limit,
            flux_no_optimization,
            flux_nunchaku_fp16,
            flux_double_cache,
            flux_teacache,
            flux_diff_multi,
            flux_diff_single,
            flux_tea_cache_l1_thresh_slider,
            flux_model_type,
            *lora_checkboxes.values(),
            *lora_weights.values()
        ]
    )

    def generate_with_performance_opts(
        seed_input, flux_prompt_input, flux_width_input, flux_height_input, 
        flux_guidance_scale_slider, flux_num_inference_steps_input, flux_memory_optimization, 
        flux_no_of_images_input, flux_randomize_seed, 
        flux_bypass_token_limit, flux_negative_prompt_input,
        no_opt, nunchaku, double_cache, teacache, diff_multi, diff_single, tea_cache_thresh, model_type, 
        *lora_inputs
    ):
        lora_files = get_lora_files()
        lora_checkboxes_vals = lora_inputs[:len(lora_files)]
        lora_weights_vals = lora_inputs[len(lora_files):]
        lora_selections = {lora_file: lora_weights_vals[i] for i, lora_file in enumerate(lora_files) if lora_checkboxes_vals[i]}
        
        if nunchaku:
            performance_optimization = "nunchaku_fp16"
        elif double_cache:
            performance_optimization = "double_cache"
        elif teacache:
            performance_optimization = "teacache"
        else:
            performance_optimization = "no_optimization"
        
        return generate_images(
            seed_input, flux_prompt_input, flux_width_input, flux_height_input,
            flux_guidance_scale_slider, flux_num_inference_steps_input, flux_memory_optimization,
            flux_no_of_images_input, flux_randomize_seed, lora_selections,
            flux_bypass_token_limit, flux_negative_prompt_input,
            performance_optimization, diff_multi, diff_single, tea_cache_thresh, model_type
        )

    generate_button.click(
        fn=generate_with_performance_opts,
        inputs=[
            seed_input, flux_prompt_input,  
            flux_width_input, flux_height_input, flux_guidance_scale_slider, 
            flux_num_inference_steps_input, flux_memory_optimization, 
            flux_no_of_images_input, flux_randomize_seed, 
            flux_bypass_token_limit, flux_negative_prompt_input,
            flux_no_optimization, flux_nunchaku_fp16, flux_double_cache, flux_teacache,
            flux_diff_multi, flux_diff_single, flux_tea_cache_l1_thresh_slider, flux_model_type, 
            *lora_checkboxes.values(),
            *lora_weights.values()
        ],
        outputs=[output_gallery]
    )