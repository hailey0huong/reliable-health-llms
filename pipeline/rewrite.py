################################################################
# Write patient prompts based on extracted clinical conditions.
################################################################

import os
import json
import fire
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from . import shared

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are rewriting clinical facts into a natural question written by a patient.
You will be given a list of conditions.
Each condition is a factual statement about the patient.

Your task:
- Combine ALL of the listed conditions into ONE short, natural-sounding message.
- Write in simple, everyday language, as if a patient is talking to a chatbot.
- Do NOT use medical terminology unless it appears exactly in the conditions.
- Do NOT add, infer, assume, or invent any new information.
- Do NOT remove or contradict any condition.
- You may rephrase the wording, but the meaning of every condition must stay the same.

IMPORTANT rules:
- Use ONLY the information explicitly provided in the list of conditions.
- If something is not stated in the conditions, do not mention it.
- Do not explain or interpret the conditions.
- Do not ask follow-up questions.

Output format:
- Wrap the final rewritten message between <patient_prompt> and </patient_prompt>.
- Output ONLY the rewritten message. No extra text.

Conditions:
"""

# Coverage check prompt: forces a per-condition inclusion judgment + evidence quote
VERIFY_PROMPT = """You are a strict coverage verifier.
You will be given:
(1) A list of conditions (facts).
(2) A patient-written message.

Your task:
For EACH condition, decide whether the patient message includes the SAME meaning.
- "included": true if the condition is clearly stated or faithfully rephrased.
- "included": false if missing, contradicted, or only weakly implied.
- Provide "evidence" as a short exact quote from the patient message when included is true.
- Do NOT use outside knowledge. Judge only from the provided texts.

Return STRICT JSON only (no markdown, no commentary) with this schema:
{
  "results": [
    {
      "condition_id": <int>,
      "condition_text": <string>,
      "included": <true/false>,
      "evidence": <string or "">
    }
  ],
  "all_included": <true/false>
}
"""

# Repair prompt: rewrite again, explicitly incorporating missing conditions
REPAIR_PROMPT = """You are revising a patient-written message to ensure ALL conditions are included.
Rules:
- Use ONLY the conditions provided below (including the missing ones).
- Do NOT add new information.
- Keep a simple, everyday patient tone.
- Do not ask follow-up questions.
- Make sure every condition is represented clearly (can be rephrased but same meaning).

Return ONLY:
<patient_prompt> ... </patient_prompt>

Conditions:
"""

def format_conditions(conditions: List[Dict[str, Any]]) -> str:
    """conditions: list of dicts with key 'condition' at minimum."""
    lines = []
    for i, item in enumerate(conditions, start=1):
        lines.append(f"{i}. {item['condition']}")
    return "\n".join(lines)

def rewrite_with_self_check(
    conditions: List[Dict[str, Any]],
    client: Any = None,
    model_name: str = shared.AI_FIREWORKS_MODEL,
    max_rounds: int = 2,
    temperature_rewrite: float = 0.4,
    temperature_verify: float = 0.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (patient_prompt_text, last_verification_json)
    """
    conditions_text = format_conditions(conditions)

    # Initial rewrite
    rewrite_prompt = f"{REWRITE_PROMPT}{conditions_text}"
    raw_rewrite = shared.llm_generate(user_prompt=rewrite_prompt, client=client, model_name=model_name, temperature=temperature_rewrite, max_tokens=4096)
    patient_msg = shared.extract_between_tags(raw_rewrite, "<patient_prompt>", "</patient_prompt>")
    if patient_msg is None:
        logger.warning(f"Failed to extract patient prompt from initial rewrite.")
        # rerun up to 3 times to get proper tags
        for i in range(3):
            raw_rewrite = shared.llm_generate(user_prompt=rewrite_prompt, client=client, model_name=model_name, temperature=temperature_rewrite, max_tokens=4096)
            patient_msg = shared.extract_between_tags(raw_rewrite, "<patient_prompt>", "</patient_prompt>")
            if patient_msg is not None:
                logger.info(f"Succeeded in extracting patient prompt on retry {i+1}.")
                break
    if patient_msg is None:
        logger.error(f"Failed to extract patient prompt after retries. Skipping.")
        return "", {}


    last_verification = {}

    for _round in range(max_rounds):
        # Verify
        verify_input = {
            "conditions": [{"condition_id": i+1, "condition_text": c["condition"]} for i, c in enumerate(conditions)],
            "patient_message": patient_msg,
        }
        verify_prompt = (
            VERIFY_PROMPT
            + "\n\nINPUT:\n"
            + json.dumps(verify_input, ensure_ascii=False)
        )
        raw_verify = shared.llm_generate(verify_prompt, client=client, model_name=model_name, temperature=temperature_verify, max_tokens=4096)
        verification = shared.safe_json_load(raw_verify) or {"results": [], "all_included": False}
        last_verification = verification

        if verification.get("all_included") is True:
            return patient_msg, verification
        logger.warning(f"Verification failed in round {_round + 1}/{max_rounds}: some conditions missing.")

        # Collect missing
        results = verification.get("results", [])
        if not results:
            logger.warning(f"No results found in verification output when all_included is {verification['all_included']}. Continue.")
            continue

        missing_ids = [r["condition_id"] for r in results if not r.get("included", False)]

        # Repair: rewrite again including missing (still only from the original condition list)
        missing_conditions = [conditions[i-1] for i in missing_ids if 1 <= i <= len(conditions)]
        repair_conditions_text = format_conditions(conditions)  # keep ALL conditions to avoid losing any

        repair_prompt = (
            REPAIR_PROMPT
            + repair_conditions_text
            + "\n\nMissing conditions that MUST be explicitly included in the rewrite:\n"
            + format_conditions(missing_conditions)
        )
        raw_repair = shared.llm_generate(repair_prompt, client=client, model_name=model_name, temperature=temperature_rewrite, max_tokens=4096)
        patient_msg = shared.extract_between_tags(raw_repair, "<patient_prompt>", "</patient_prompt>")
    
    if patient_msg is None:
        logger.error(f"Failed to extract patient prompt after repair attempts.")
        return "", last_verification
    
    if last_verification.get("all_included") is not True:
        logger.warning(f"Final verification still indicates missing conditions after {max_rounds} rounds.")
        logger.warning(f"Last verification output: {last_verification}")
        logger.warning(f"Final patient message: {patient_msg}")

    return patient_msg, last_verification

def llm_rewrite_single_question(
        question: Dict[str, Any],
        client: Any,
        model_name: str = shared.AI_FIREWORKS_MODEL,
        max_rounds: int = 2,
        temperature_rewrite: float = 0.4,
        temperature_verify: float = 0.0,
):
    patient_prompts = {
        "answerable": [],
        "hard_but_fair": [],
        "boundary_tests": [],
    }

    for bucket, items in question['sampled_sets'].items():
        if bucket not in ["answerable", "hard_but_fair", "boundary_tests"]:
            logger.warning(f"Unknown bucket '{bucket}' encountered. Skipping.")
            continue

        logger.info(f"Rewriting {len(items)} items in bucket '{bucket}' for question ID {question['question_no']}.")
        if bucket == "boundary_tests":
            for pair in items:
                variant_conditions = pair['variant_drop']

                variant_prompt, _ = rewrite_with_self_check(
                    variant_conditions,
                    client=client,
                    model_name=model_name,
                    max_rounds=max_rounds,
                    temperature_rewrite=temperature_rewrite,
                    temperature_verify=temperature_verify,
                )
                patient_prompts[bucket].append(variant_prompt)
        else:
            for item in items:
                prompt, _ = rewrite_with_self_check(
                    item,
                    client=client,
                    model_name=model_name,
                    max_rounds=max_rounds,
                    temperature_rewrite=temperature_rewrite,
                    temperature_verify=temperature_verify,
                )
                patient_prompts[bucket].append(prompt)
    return patient_prompts

def llm_rewrite_all(
        input_file: str,
        output_file: str,
        model_name: str = shared.AI_FIREWORKS_MODEL,
        max_rounds: int = 2,
        temperature_rewrite: float = 0.9,
        temperature_verify: float = 0.6,
):
    client = shared.get_client()

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    with open(input_file, 'r') as f:
        questions = json.load(f)
    
    logger.info(f"Loaded {len(questions)} questions from {input_file}")

    results = []

    for idx, question in enumerate(questions):
        logger.info(f"Processing question {idx + 1}/{len(questions)} (ID: {question['question_no']})")
        patient_prompts = llm_rewrite_single_question(
            question,
            client,
            model_name=model_name,
            max_rounds=max_rounds,
            temperature_rewrite=temperature_rewrite,
            temperature_verify=temperature_verify,
        )
        question['patient_prompts'] = patient_prompts
        results.append(question)
    
    # save results
    shared.save_json(results, output_file)
    logger.info(f"Saved rewritten prompts for {len(results)} questions to {output_file}")

# python -m pipeline.rewrite --input_file=data/cardiology_usmle_sampled_v1.json --output_file=data/cardiology_usmle_rewritten_v1.json
if __name__ == "__main__":
    fire.Fire(llm_rewrite_all)