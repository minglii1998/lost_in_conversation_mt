import os
import time
import json
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from peft import PeftModel, PeftConfig

# -----------------------------------------------------------------------------
# Helper utilities (copied from model_openai.py for consistency)
# -----------------------------------------------------------------------------

def format_messages(messages: List[Dict[str, Any]], variables: Dict[str, str] = {}):
    """Replace [[VAR]] placeholders in the *last* user message with their values."""
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert isinstance(v, str), f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # Warn if some keys were not replaced
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

# -----------------------------------------------------------------------------
# Model wrapper
# -----------------------------------------------------------------------------

class OpenSource_Model:
    """A minimal local model wrapper that mimics the OpenAI interface."""

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):

        self.model_name = model_name
        if "lora" in model_name:
            self.tokenizer, self.model = self._load_model_lora(model_name)
        else:
            self.tokenizer, self.model = self._load_model(model_name)
        print(f"Model loaded successfully: {model_name}")

    # ---------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------

    def _load_model(self, model_id: str, device: Optional[int] = None):
        """Load the tokenizer & model for the requested HF checkpoint."""
        
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        return tokenizer, model

    def _load_model_lora(self, model_id: str, device: Optional[int] = None):
        """Load the tokenizer & model for the requested HF checkpoint."""
        """Load trained LoRA weights"""

        lora_dir = os.path.join(model_id, "actor/lora_adapter")
        lora_config_dir = os.path.join(lora_dir, "adapter_config.json")
        lora_config = json.load(open(lora_config_dir))
        base_model_id = lora_config["base_model_name_or_path"]
        
        kwargs = {"device_map": "auto", "trust_remote_code": True}
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **kwargs)

        model = PeftModel.from_pretrained(model, lora_dir)

        return tokenizer, model

    # ---------------------------------------------------------------------
    # Public API (mirrors OpenAI_Model)
    # ---------------------------------------------------------------------

    def cost_calculator(self, model: str, usage: Dict[str, int], is_batch_model: bool = False):
        """For open-source local models we consider the cost to be zero."""
        return 0.0

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 1.0,
        is_json: bool = False,
        return_metadata: bool = False,
        max_tokens: Optional[int] = None,
        variables: Dict[str, str] = {},
    ):
        """Generate a response from the loaded model.

        Args:
            model: Ignored. Model was already specified during initialization.
                   Kept for API compatibility with other model classes.
        
        Note: *timeout* and *max_retries* are kept for API parity but are not
        strictly enforced when running locally.
        """
        messages = format_messages(messages, variables)

        if 'Qwen3' in self.model_name:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        # Use the pre-loaded model and tokenizer
        gen_kwargs = {
            "max_new_tokens": max_tokens or 512,
            "temperature": temperature,
            "do_sample": temperature > 0.0,
            "top_p": 0.95,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        output_ids = self.model.generate(**model_inputs, **gen_kwargs)
        generated_ids = output_ids[0][model_inputs["input_ids"].size(1):]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        response_text_clean = response_text
        # Token accounting (approximate using tokenizer length)
        prompt_tokens = model_inputs["input_ids"].size(1)
        completion_tokens = output_ids.size(1)-model_inputs["input_ids"].size(1)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        total_usd = 0.0

        if not return_metadata:
            return response_text_clean

        return {
            "message": response_text_clean,
            "total_tokens": usage["total_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "prompt_tokens_cached": 0,  # No caching for local models
            "completion_tokens": usage["completion_tokens"],
            "total_usd": total_usd,
        }


# -----------------------------------------------------------------------------
# Simple manual test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    model = OpenSource_Model()
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    print("Running local inference… (this may download the model on first run)")
    result = model.generate(test_messages, return_metadata=True, temperature=0.0)
    print(json.dumps(result, indent=2, ensure_ascii=False)) 