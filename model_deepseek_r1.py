import json
import logging
import re
import tenacity
import boto3
import botocore
from botocore.config import Config
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _clean_messages_for_deepseek(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean messages for Deepseek R1 format."""
    cleaned_messages = []
    for msg in messages:
        if isinstance(msg["content"], str):
            cleaned_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })
        else:
            cleaned_messages.append(msg)
    return cleaned_messages

def _build_deepseek_request(messages: List[Dict[str, Any]], system_prompt: str = '', temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """Build request body for Deepseek R1."""
    stop_sequences = ["</api_ca", "<|im_end", "</respon"]
    
    inf_params = {
        "maxTokens": max_tokens,
        "temperature": temperature,
        "topP": 0.999,
        'stopSequences': stop_sequences
    }

    additionalModelRequestFields = {
        "inferenceConfig": {
            "top_k": 250,
        }
    }

    if system_prompt:
        wrapped_system_prompt = [{"text": system_prompt}]
        return json.dumps({
            "messages": _clean_messages_for_deepseek(messages),
            "system": wrapped_system_prompt,
            "modelId": "us.deepseek.r1-v1:0",
            "inferenceConfig": inf_params,
            "additionalModelRequestFields": additionalModelRequestFields
        })
    else:
        return json.dumps({
            "messages": _clean_messages_for_deepseek(messages),
            "modelId": "us.deepseek.r1-v1:0",
            "inferenceConfig": inf_params,
            "additionalModelRequestFields": additionalModelRequestFields
        })

def truncate_by_stop(response_text: str, stop_sequences: List[str]) -> str:
    """Truncate response text at stop sequences."""
    for stop in stop_sequences:
        idx = response_text.find(stop)
        if idx != -1:
            return response_text[:idx]
    return response_text

class DeepseekR1_Model:
    def __init__(self):
        self.config = Config(read_timeout=1000)
        self.bedrock = boto3.client('bedrock-runtime', region_name="us-east-1", config=self.config)
        
    def cost_calculator(self, model: str, usage: Dict[str, int]) -> float:

        return 0.0

    def generate(self, messages: List[Dict[str, Any]], model: str = "deepseek-r1", timeout: int = 30, 
                max_retries: int = 3, temperature: float = 1.0, is_json: bool = False, 
                return_metadata: bool = False, max_tokens: Optional[int] = None, 
                variables: Dict[str, Any] = {}) -> Any:
        """Generate response using Deepseek R1 model."""
        messages = _clean_messages_for_deepseek(messages)
        
        # Extract system message if present
        system_prompt = ""
        if len(messages) > 0 and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"][0]["text"]
            messages = messages[1:]

        # Set max tokens if not specified
        if max_tokens is None:
            max_tokens = 2048

        # Configure retry logic
        logger = logging.getLogger(__name__)
        retryer = tenacity.Retrying(
            stop=tenacity.stop_after_attempt(max_retries),
            wait=tenacity.wait_fixed(4),  # 4 second wait between retries
            retry=tenacity.retry_if_exception_type(botocore.exceptions.ClientError) | 
                  tenacity.retry_if_exception_type(botocore.exceptions.ValidationError) |
                  tenacity.retry_if_exception_type(Exception),
            before_sleep=tenacity.before_sleep_log(logger, logging.ERROR),
            reraise=True,
        )

        # Build inference parameters
        inf_params = {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.999,
            "stopSequences": ["</api_ca", "<|im_end", "</respon"]
        }

        try:
            # Call Deepseek R1 API using converse
            if system_prompt:
                wrapped_system_prompt = [{"text": system_prompt}]
                response = retryer(self.bedrock.converse, 
                                 messages=messages, 
                                 system=wrapped_system_prompt, 
                                 modelId="us.deepseek.r1-v1:0",
                                 inferenceConfig=inf_params)
            else:
                response = retryer(self.bedrock.converse, 
                                 messages=messages, 
                                 modelId="us.deepseek.r1-v1:0",
                                 inferenceConfig=inf_params)
            
            # # Extract response text
            # think_output = response["output"]["message"]["content"][1]['reasoningContent']['reasoningText']['text']
            # think_output = '<think>\n' + think_output + '\n</think>\n'
            # response_text = response["output"]["message"]["content"][0]["text"].strip()
            # response_text = think_output + response_text

            response_text = response["output"]["message"]["content"][0]["text"].strip()

            # Calculate usage and cost
            # Note: Deepseek R1 API doesn't provide token counts, so we estimate
            estimated_prompt_tokens = sum(len(msg["content"][0]["text"].split()) * 1.3 for msg in messages)
            estimated_completion_tokens = len(response_text.split()) * 1.3
            
            usage = {
                'prompt_tokens': int(estimated_prompt_tokens),
                'completion_tokens': int(estimated_completion_tokens),
                'total_tokens': int(estimated_prompt_tokens + estimated_completion_tokens)
            }
            
            total_usd = self.cost_calculator(model, usage)

            if not return_metadata:
                return response_text
                
            return {
                "message": response_text,
                "total_tokens": usage['total_tokens'],
                "prompt_tokens": usage['prompt_tokens'],
                "prompt_tokens_cached": 0,  # Deepseek R1 doesn't support caching
                "completion_tokens": usage['completion_tokens'],
                "total_usd": total_usd
            }

        except Exception as e:
            raise Exception(f"Failed to get response from Deepseek R1: {str(e)}")

if __name__ == "__main__":
    # Test the model
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    model = DeepseekR1_Model()
    response = model.generate(messages, return_metadata=True)
    print(json.dumps(response, indent=2)) 