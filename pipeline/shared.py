import aisuite as ai
import json
import re
import time
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

AI_FIREWORKS_MODEL = "fireworks:accounts/fireworks/models/gpt-oss-120b"
AI_CLAUDE_MODEL = "anthropic:claude-sonnet-4-5-20250929"
AI_GPT_MODEL = "openai:gpt-5.2"

def get_client():
    return ai.Client()


def llm_generate(
    user_prompt: str,
    model_name: str,
    client: ai.Client = None,
    system_prompt: str = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    top_p: Optional[float] = None,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    reasoning_effort: str = "high",
) -> str:
    """Get response from AI Suite chat completion with retry logic.
    
    Args:
        user_prompt: The user message to send
        model_name: Model identifier (e.g., AI_GPT_MODEL, AI_CLAUDE_MODEL)
        client: AI Suite client instance (created if None)
        system_prompt: Optional system message
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter (optional, mutually exclusive with temperature for Claude)
        frequency_penalty: Frequency penalty (GPT models only)
        presence_penalty: Presence penalty (GPT models only)
        max_retries: Number of retry attempts on failure
        retry_delay: Base delay between retries (uses exponential backoff)
        reasoning_effort: Reasoning effort level for GPT models ("low", "medium", "high")
    
    Returns:
        The model's response text
        
    Raises:
        Exception: If all retry attempts fail
    """
    if client is None:
        client = get_client()

    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Check if top_p is a valid number
    has_valid_top_p = top_p is not None and isinstance(top_p, (int, float))

    last_exception = None
    for attempt in range(max_retries):
        try:
            # Build kwargs based on model type
            kwargs = {
                "model": model_name,
                "messages": messages,
            }
            
            if model_name == AI_GPT_MODEL:
                # OpenAI GPT-specific parameters
                kwargs["temperature"] = temperature
                kwargs["max_completion_tokens"] = max_tokens
                kwargs["reasoning_effort"] = reasoning_effort
                kwargs["frequency_penalty"] = frequency_penalty
                kwargs["presence_penalty"] = presence_penalty
                if has_valid_top_p:
                    kwargs["top_p"] = float(top_p)
            elif "anthropic" in model_name:
                # Anthropic Claude: temperature and top_p are mutually exclusive
                kwargs["max_tokens"] = max_tokens
                if has_valid_top_p:
                    kwargs["top_p"] = float(top_p)
                else:
                    kwargs["temperature"] = temperature
            else:
                # Fireworks and other models
                kwargs["temperature"] = temperature
                kwargs["max_tokens"] = max_tokens
                if has_valid_top_p:
                    kwargs["top_p"] = float(top_p)
            
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content is None:
                finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                logger.warning(f"LLM returned None content. finish_reason={finish_reason}, model={model_name}")
            return content
            
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"LLM call failed after {max_retries} attempts: {e}")
    
    raise last_exception

def extract_between_tags(text, start_tag, end_tag):
    pattern = re.escape(start_tag) + r'(.*?)' + re.escape(end_tag)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# save file in json format
def save_json(data: dict, outpath: str):
    """Save data in json format."""
    with open(outpath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {outpath}")

def save_jsonl(data: list, outpath: str):
    """Save data in jsonl format."""
    with open(outpath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved data to {outpath}")

def safe_json_load(s: str) -> Optional[Dict[str, Any]]:
    """Try to parse JSON even if model adds small garbage before/after."""
    s = s.strip()
    # Try direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try extracting first JSON object
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None