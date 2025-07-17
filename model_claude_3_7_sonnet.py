import json
import logging
import tenacity
import botocore.exceptions
import boto3
import os
import time
import re

# AWS Bedrock clients for different regions
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_west_2 = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

def format_messages(messages, variables={}):
    last_user_msg = [msg for msg in messages if msg["role"] == "user"][-1]

    for k, v in variables.items():
        key_string = f"[[{k}]]"
        if key_string not in last_user_msg["content"]:
            print(f"[prompt] Key {k} not found in prompt; effectively ignored")
        assert type(v) == str, f"[prompt] Variable {k} is not a string"
        last_user_msg["content"] = last_user_msg["content"].replace(key_string, v)

    # find all the keys that are still in the prompt using regex [[STR]] where STR is alnum witout space
    keys_still_in_prompt = re.findall(r"\[\[([^\]]+)\]\]", last_user_msg["content"])
    if len(keys_still_in_prompt) > 0:
        print(f"[prompt] The following keys were not replaced: {keys_still_in_prompt}")

    return messages

def _clean_messages_for_claude(messages):
    return list(
        map(
            lambda item: {k: v for k, v in item.items() if k in ["role", "content"]},
            messages,
        )
    )

def _build_claude_request(messages, system_prompt, temperature=0.0, max_tokens=2048):
    if system_prompt:
        return json.dumps(
            {
                "max_tokens": max_tokens,
                "messages": _clean_messages_for_claude(messages),
                "system": system_prompt,
                "anthropic_version": "bedrock-2023-05-31",
                'stop_sequences': ["</api_call>", "<|im_end|>", "</response>"],
                'temperature': temperature
            }
        )
    else:
        return json.dumps(
            {
                "max_tokens": max_tokens,
                "messages": _clean_messages_for_claude(messages),
                "anthropic_version": "bedrock-2023-05-31",
                'stop_sequences': ["</api_call>", "<|im_end|>", "</response>"],
                'temperature': temperature
            }
        )

class Claude_Model:
    def __init__(self):
        # No need to check credentials as they are handled by boto3
        pass

    def cost_calculator(self, model, usage):
        return 0.0

    def generate(self, messages, model="claude-3-7-sonnet", timeout=30, max_retries=3, temperature=1.0, is_json=False, return_metadata=False, max_tokens=None, variables={}):
        # Map model names to their corresponding model IDs
        model_id_map = {
            "claude-3-7-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0"
        }
        
        # Get the model ID, default to Claude 3.7 Sonnet if not found
        model_id = model_id_map.get(model, model_id_map["claude-3-7-sonnet"])

        messages = format_messages(messages, variables)
        
        # Extract system message if present
        system_prompt = ""
        if len(messages) > 0 and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        # Set max tokens if not specified
        if max_tokens is None:
            max_tokens = 2048

        # Build request body
        body = _build_claude_request(messages, system_prompt, temperature, max_tokens)

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

        try:
            # Call Claude API
            response = retryer(bedrock.invoke_model, body=body, modelId=model_id)
            response_body = json.loads(response.get("body").read())
            
            # Extract response text
            if "content" in response_body:
                response_text = response_body["content"][0]["text"]
            else:
                response_text = response_body["completion"]

            # If JSON is requested, try to extract JSON from the response
            if is_json:
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)
                else:
                    # If no JSON found, wrap the response in a JSON object
                    response_text = json.dumps({"response": response_text})

            # Calculate usage and cost
            # Note: Claude API doesn't provide token counts in response, so we estimate
            # This is a rough estimate and may not be accurate
            estimated_prompt_tokens = sum(len(msg["content"].split()) * 1.3 for msg in messages)
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
                "prompt_tokens_cached": 0,  # Claude doesn't support caching
                "completion_tokens": usage['completion_tokens'],
                "total_usd": total_usd
            }

        except Exception as e:
            raise Exception(f"Failed to get response from Claude: {str(e)}")


if __name__ == "__main__":
    # Test the model
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    model = Claude_Model()
    response = model.generate(messages, return_metadata=True)
    print(json.dumps(response, indent=2)) 