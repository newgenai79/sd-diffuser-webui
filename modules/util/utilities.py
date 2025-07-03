import torch
import gc
import time
import modules.util.appstate
from PIL import Image
import json
import piexif
import piexif.helper
from pathlib import Path

def clear_controlnet_model_memory():
    print(">>>>clear_controlnet_model_memory<<<<")
    del modules.util.appstate.global_controlnet_model
    modules.util.appstate.global_controlnet_model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

def clear_previous_model_memory():

    if modules.util.appstate.global_pipe is not None:
        print(">>>>clear_previous_model_memory: Removing model from memory<<<<")
        
        # Remove hooks if present
        if hasattr(modules.util.appstate.global_pipe, 'remove_all_hooks'):
            modules.util.appstate.global_pipe.remove_all_hooks()
        
        # Explicitly delete model components
        if modules.util.appstate.global_model_manager is None:
            if hasattr(modules.util.appstate.global_pipe, '__dict__'):
                for attr_name, attr_value in list(modules.util.appstate.global_pipe.__dict__.items()):
                    if hasattr(attr_value, 'to'):
                        try:
                            attr_value.to("cpu")  # Move to CPU to free VRAM
                        except:
                            pass
                    delattr(modules.util.appstate.global_pipe, attr_name)
        if modules.util.appstate.global_model_manager is not None:
            del modules.util.appstate.global_model_manager
        # Delete the pipeline
    del modules.util.appstate.global_pipe
        
    # Reset all global state variables
    modules.util.appstate.global_pipe = None
    modules.util.appstate.global_memory_mode = None
    modules.util.appstate.global_inference_type = None
    modules.util.appstate.global_model_type = None
    modules.util.appstate.global_quantization = None
    modules.util.appstate.global_selected_gguf = None
    del modules.util.appstate.global_textencoder
    modules.util.appstate.global_textencoder = None
    del modules.util.appstate.global_transformer
    modules.util.appstate.global_transformer = None   
    modules.util.appstate.global_selected_lora = None
    modules.util.appstate.global_bypass_token_limit = None
    del modules.util.appstate.global_text_encoder_2
    modules.util.appstate.global_text_encoder_2 = None
    del modules.util.appstate.global_controlnet
    modules.util.appstate.global_controlnet = None
    del modules.util.appstate.global_controlnet_model
    modules.util.appstate.global_controlnet_model = None
    modules.util.appstate.global_performance_optimization = None
    modules.util.appstate.global_model_manager = None
    modules.util.appstate.global_use_qencoder = None
    # Force garbage collection and CUDA memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    
    # Debugging memory usage
    print("CUDA allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
    print("CUDA reserved:", torch.cuda.memory_reserved() / 1e6, "MB")

"""
def clear_previous_model_memory():
    if modules.util.appstate.global_pipe is not None:
        print(">>>>clear_previous_model_memory: Removing model from memory<<<<")
        
        # 1. Remove hooks if present (works for both pipeline types)
        if hasattr(modules.util.appstate.global_pipe, 'remove_all_hooks'):
            modules.util.appstate.global_pipe.remove_all_hooks()
        
        # 2. Identify pipeline type and handle special components
        is_diffsynth = hasattr(modules.util.appstate.global_pipe, 'model_names')
        
        # 3. Handle DiffSynth-specific components if applicable
        if is_diffsynth:
            model_names = getattr(modules.util.appstate.global_pipe, 'model_names', [])
            for component_name in model_names:
                if hasattr(modules.util.appstate.global_pipe, component_name):
                    component = getattr(modules.util.appstate.global_pipe, component_name)
                    if component is not None:
                        # print(f"Clearing DiffSynth component: {component_name}")
                        try:
                            # Move to CPU first
                            if hasattr(component, 'to'):
                                component.to("cpu")
                            
                            # Clear parameters if it's a module
                            if isinstance(component, torch.nn.Module):
                                for param in component.parameters():
                                    if param.data is not None:
                                        param.data = None
                                    if param.grad is not None:
                                        param.grad.data = None
                                        param.grad = None
                            
                            # Set to None
                            setattr(modules.util.appstate.global_pipe, component_name, None)
                        except Exception as e:
                            print(f"Error clearing {component_name}: {e}")
        
        # 4. General cleanup for all pipeline types
        if hasattr(modules.util.appstate.global_pipe, '__dict__'):
            # Get all attributes
            attrs_to_process = list(modules.util.appstate.global_pipe.__dict__.items())
            
            # First pass: move everything to CPU
            for attr_name, attr_value in attrs_to_process:
                # Skip special attributes
                if attr_name.startswith('_') and attr_name in ('_parameters', '_buffers', '_modules'):
                    continue
                
                # Skip already processed DiffSynth components
                if is_diffsynth and attr_name in getattr(modules.util.appstate.global_pipe, 'model_names', []):
                    continue
                    
                # Move to CPU if possible
                if hasattr(attr_value, 'to'):
                    try:
                        # print(f"Moving {attr_name} to CPU")
                        attr_value.to("cpu")
                    except Exception as e:
                        print(f"Error moving {attr_name} to CPU: {e}")
                
                # Handle modules specifically
                if isinstance(attr_value, torch.nn.Module):
                    try:
                        for param in attr_value.parameters():
                            if param.data is not None:
                                param.data = None
                            if param.grad is not None:
                                param.grad.data = None
                                param.grad = None
                    except Exception as e:
                        print(f"Error clearing parameters for {attr_name}: {e}")
            
            # Second pass: delete attributes
            for attr_name, _ in attrs_to_process:
                # Skip special PyTorch attributes
                if attr_name.startswith('_') and attr_name in ('_parameters', '_buffers', '_modules'):
                    continue
                
                # Skip already processed DiffSynth components
                if is_diffsynth and attr_name in getattr(modules.util.appstate.global_pipe, 'model_names', []):
                    continue
                    
                # print(f"Deleting {attr_name}")
                try:
                    delattr(modules.util.appstate.global_pipe, attr_name)
                except Exception as e:
                    print(f"Error deleting {attr_name}: {e}")
        
        # 5. Delete the pipeline itself
        del modules.util.appstate.global_pipe
    
    # 6. Reset all global state variables
    modules.util.appstate.global_pipe = None
    modules.util.appstate.global_memory_mode = None
    modules.util.appstate.global_inference_type = None
    modules.util.appstate.global_model_type = None
    modules.util.appstate.global_quantization = None
    modules.util.appstate.global_selected_gguf = None
    
    # 7. Handle text encoders safely
    for encoder_name in ['global_textencoder', 'global_text_encoder_2']:
        if hasattr(modules.util.appstate, encoder_name):
            encoder = getattr(modules.util.appstate, encoder_name)
            if encoder is not None:
                del encoder
            setattr(modules.util.appstate, encoder_name, None)
    
    modules.util.appstate.global_selected_lora = None
    modules.util.appstate.global_bypass_token_limit = None
    modules.util.appstate.global_performance_optimization = None
    
    # 8. Thorough memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    for _ in range(2):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
  
    # 9. Display memory usage
    print("CUDA allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
    print("CUDA reserved:", torch.cuda.memory_reserved() / 1e6, "MB")
"""

def pad_tensors_to_equal_length(prompt_embeds, negative_prompt_embeds):
    """Pad the shorter tensor to match the longer one's sequence length"""
    prompt_shape = prompt_embeds.shape
    negative_shape = negative_prompt_embeds.shape
    
    # If shapes are already equal, return as is
    if prompt_shape[1] == negative_shape[1]:
        return prompt_embeds, negative_prompt_embeds
    
    # Determine which one is shorter and pad it
    if prompt_shape[1] < negative_shape[1]:
        # Pad prompt_embeds
        padding_length = negative_shape[1] - prompt_shape[1]
        # Create padding that matches the embedding dimensions
        padding = torch.zeros((prompt_shape[0], padding_length, prompt_shape[2]), 
                              dtype=prompt_embeds.dtype,
                              device=prompt_embeds.device)
        # Concatenate along sequence dimension
        padded_prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
        return padded_prompt_embeds, negative_prompt_embeds
    else:
        # Pad negative_prompt_embeds
        padding_length = prompt_shape[1] - negative_shape[1]
        padding = torch.zeros((negative_shape[0], padding_length, negative_shape[2]),
                             dtype=negative_prompt_embeds.dtype,
                             device=negative_prompt_embeds.device)
        padded_negative_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)
        return prompt_embeds, padded_negative_embeds

def clear_text_model_memory(_pipe):
    if hasattr(_pipe, 'remove_all_hooks'):
        _pipe.remove_all_hooks()
    # Explicitly delete model components
    if hasattr(_pipe, '__dict__'):
        for attr_name, attr_value in list(_pipe.__dict__.items()):
            if hasattr(attr_value, 'to'):
                try:
                    attr_value.to("cpu")
                except:
                    pass
            delattr(_pipe, attr_name)
    del _pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def save_metadata_to_file(file_path, metadata):
    """Save metadata to image or video file."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.png', '.jpg', '.jpeg']:
        return save_image_metadata(file_path, metadata)
    elif file_ext in ['.mp4', '.mov']:
        return save_video_metadata(file_path, metadata)
    else:
        print(f"Unsupported file type for metadata: {file_ext}")
        return False

def save_image_metadata(image_path, metadata):
    """Save metadata to image using Exif UserComment."""
    try:
        image = Image.open(image_path)
        
        # Ensure numeric values maintain their precision during serialization
        # By using a custom JSON encoder class
        class PrecisionEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float):
                    # Format float with enough decimal places to preserve precision
                    # This prevents 3.5 from becoming 3
                    return float(f"{obj:.2f}")
                return json.JSONEncoder.default(self, obj)
        
        # Convert metadata dict to string with custom encoder
        metadata_str = piexif.helper.UserComment.dump(json.dumps(metadata, cls=PrecisionEncoder))
        
        # Create Exif dict
        exif_dict = {"Exif": {piexif.ExifIFD.UserComment: metadata_str}}
        
        # Convert to bytes
        exif_bytes = piexif.dump(exif_dict)
        
        # Save image with metadata
        image.save(image_path, exif=exif_bytes)
        
        return True
    except Exception as e:
        print(f"Error saving image metadata: {str(e)}")
        return False

def save_video_metadata(video_path, metadata):
    """Save metadata to video file using sidecar JSON."""
    try:
        json_path = video_path + '.json'
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving video metadata: {str(e)}")
        return False

def read_metadata_from_file(file_path):
    """Read metadata from image or video file."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext in ['.png', '.jpg', '.jpeg']:
        return read_image_metadata(file_path)
    elif file_ext in ['.mp4', '.mov']:
        return read_video_metadata(file_path)
    else:
        print(f"Unsupported file type for metadata: {file_ext}")
        return None

def read_image_metadata(image_path):
    """Read metadata from image Exif UserComment."""
    try:
        image = Image.open(image_path)
        exif = image.getexif()
        
        if exif is None:
            return None
            
        exif_dict = piexif.load(image.info["exif"])
        user_comment = exif_dict["Exif"][piexif.ExifIFD.UserComment]
        metadata = json.loads(piexif.helper.UserComment.load(user_comment))
        return metadata
    except Exception as e:
        print(f"Error reading image metadata: {str(e)}")
        return None

def read_video_metadata(video_path):
    """Read metadata from video sidecar JSON."""
    try:
        json_path = video_path + '.json'
        if not Path(json_path).exists():
            return None
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading video metadata: {str(e)}")
        return None