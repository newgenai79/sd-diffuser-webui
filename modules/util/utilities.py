import torch
import modules.util.appstate

def clear_previous_model_memory():
    if modules.util.appstate.global_pipe is not None:
        print(">>>>clear_previous_model_memory: Removing model from memory<<<<")
        modules.util.appstate.global_pipe.remove_all_hooks()
        del modules.util.appstate.global_pipe
        modules.util.appstate.global_pipe = None
        modules.util.appstate.global_memory_mode = None
        modules.util.appstate.global_inference_type = None
        modules.util.appstate.global_quantization = None
        modules.util.appstate.global_selected_gguf = None
        modules.util.appstate.global_textencoder = None
        torch.cuda.empty_cache()