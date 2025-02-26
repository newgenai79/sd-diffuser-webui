import torch
import gc
import modules.util.appstate
from PIL import Image
import json
import piexif
import piexif.helper
from pathlib import Path

def clear_previous_model_memory():
    if modules.util.appstate.global_pipe is not None:
        print(">>>>clear_previous_model_memory: Removing model from memory<<<<")
        if hasattr(modules.util.appstate.global_pipe, 'remove_all_hooks'):
            modules.util.appstate.global_pipe.remove_all_hooks()
        del modules.util.appstate.global_pipe
        modules.util.appstate.global_pipe = None
        modules.util.appstate.global_memory_mode = None
        modules.util.appstate.global_inference_type = None
        modules.util.appstate.global_model_type = None
        modules.util.appstate.global_quantization = None
        modules.util.appstate.global_selected_gguf = None
        modules.util.appstate.global_textencoder = None
        modules.util.appstate.global_selected_lora = None
        gc.collect()
        torch.cuda.empty_cache()


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
        
        # Convert metadata dict to string
        metadata_str = piexif.helper.UserComment.dump(json.dumps(metadata))
        
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