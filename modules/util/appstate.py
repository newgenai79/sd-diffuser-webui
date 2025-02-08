"""
Global application state and browser state management for the WebUI
"""
import os
import json

# Existing global variables
global_pipe = None
global_memory_mode = None
global_inference_type = None
global_quantization = None
global_inference_in_progress = False
global_selected_gguf = None
global_textencoder = None

class StateManager:
    def __init__(self):
        self.state_dir = "saved_state"
        # Create the directory if it doesn't exist
        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)
    
    def get_state(self, model_key):
        """
        Retrieves the saved state for a specific model from JSON file
        
        Args:
            model_key (str): The model identifier (e.g., 'flex1_alpha', 'sana')
            
        Returns:
            dict: The saved state for the model or None if not found
        """
        try:
            file_path = os.path.join(self.state_dir, f"{model_key}_state.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"Error retrieving state for {model_key}: {str(e)}")
            return None

    def save_state(self, model_key, state_dict):
        """
        Saves the state for a specific model to JSON file
        
        Args:
            model_key (str): The model identifier (e.g., 'flex1_alpha', 'sana')
            state_dict (dict): The state to save for the model
            
        Returns:
            str: Success or error message
        """
        try:
            file_path = os.path.join(self.state_dir, f"{model_key}_state.json")
            with open(file_path, 'w') as f:
                json.dump(state_dict, f, indent=4)
            return "State saved successfully"
        except Exception as e:
            error_msg = f"Error saving state for {model_key}: {str(e)}"
            print(error_msg)
            return error_msg

state_manager = StateManager()