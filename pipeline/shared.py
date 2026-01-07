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
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    reasoning_effort: str = "high",
) -> str:
    """Get response from AI Suite chat completion with retry logic."""
    if client is None:
        client = get_client()

    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    last_exception = None
    for attempt in range(max_retries):
        try:
            if model_name == AI_GPT_MODEL:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_completion_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                )

            return response.choices[0].message.content
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