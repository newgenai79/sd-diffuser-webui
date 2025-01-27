import torch
import modules.util.config

def clear_previous_model_memory():
    if modules.util.config.global_pipe is not None:
        print(">>>>clear_previous_model_memory: Removing model from memory<<<<")
        modules.util.config.global_pipe.remove_all_hooks()
        del modules.util.config.global_pipe
        modules.util.config.global_pipe = None
        modules.util.config.global_memory_mode = None
        modules.util.config.global_vaeslicing = None
        modules.util.config.global_vaetiling = None
        modules.util.config.global_inference_type = None
        modules.util.config.global_quantization = None
        modules.util.config.global_selected_gguf = None
        torch.cuda.empty_cache()