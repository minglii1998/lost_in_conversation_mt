def get_model_class(model_name):
    """Get the appropriate model instance based on model name"""
    if model_name.startswith("claude"):
        from model_claude_3_7_sonnet import Claude_Model
        return Claude_Model()
    elif model_name == "deepseek-r1":
        from model_deepseek_r1 import DeepseekR1_Model
        return DeepseekR1_Model()
    else:
        # Fallback to open-source HF models (e.g., Qwen, Llama-3, etc.)
        from model_opensourse import OpenSource_Model
        return OpenSource_Model(model_name=model_name) 