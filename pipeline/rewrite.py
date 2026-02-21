################################################################
# Write patient prompts based on extracted clinical conditions.
################################################################

import os
import json
import fire
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
- Do NOT add, infer, assume, or invent any new information.
- Do NOT remove or contradict any condition.
- You may rephrase the wording, but the meaning of every condition must stay the same.

REALISTIC PATIENT VOICE rules:
- Physical exam findings must be converted to what a patient would EXPERIENCE or
  what a DOCTOR TOLD THEM. Patients do not perform their own physical exams.
  Transform like this:
    "jugular venous distention 10 cm" -> "my neck veins are really swollen and bulging"
    "holosystolic murmur at apex" -> "the doctor said I have a heart murmur"
    "crackles on auscultation bilaterally" -> "the doctor said my lungs sound wet/crackly"
    "absent lower extremity pulses" -> "I can't feel a pulse in my legs and they feel cold"
    "hepatomegaly 3 cm below costal margin" -> "the doctor said my liver is enlarged and tender"
    "S3 gallop" -> "the doctor heard an extra heart sound"
    "pitting edema 2+ to 3+" -> "my legs are really swollen"
- Lab values and imaging results should use plain framing:
    "troponin I elevated" -> "my blood test showed a high troponin level"
    "echocardiography showed anterolateral akinesis" -> "my heart ultrasound showed part of my heart wall isn't moving"
    "LDL 170 mg/dL" -> "my LDL cholesterol is 170" (numbers are OK to keep)
- Diagnoses the patient was told about should be framed as "the doctors diagnosed me with..." or "I was told I have..."
- Medical abbreviations (STEMI, JVD, BMI) should be spelled out or described in plain language.
- Keep the patient's age, sex, and timeline details — patients DO know and report these.

IMPORTANT rules:
- Use ONLY the information explicitly provided in the list of conditions.
- If something is not stated in the conditions, do not mention it.
- Do not explain or interpret the conditions.
- Do not ask follow-up questions.

EXAMPLES of good patient rewrites:

Example 1:
Conditions:
1. 45-year-old male
2. Tuberculous pericarditis diagnosed months ago
3. Pericardiocentesis performed previously
4. Increasing shortness of breath on minimal exertion over 2 weeks
5. Fluid buildup and lower extremity swelling
Good patient message: "I'm 45 years old and a couple of months ago I got tuberculosis that affected the lining of my heart. I even had to have fluid drained out of my heart. Now I'm developing fluid buildup all over my body and I'm getting more short of breath."

Example 2:
Conditions:
1. 13-year-old boy
2. Viral upper respiratory symptoms 1 week ago that resolved
3. 3-day progressive fatigue, shortness of breath, difficulty walking up stairs
4. Diagnosed with heart failure with fluid on lungs and low blood pressure
Good patient message: "My 13 year old boy got sick with what seemed like the flu for a few days, then a few weeks passed and he had to be admitted to the hospital and the doctors diagnosed him with heart failure, with lots of fluid on his body and low blood pressure."

Example 3:
Conditions:
1. 5 days post-admission for large heart attack (STEMI)
2. Heart ultrasound showed front wall of heart not moving
3. Cannot feel legs, legs feel cold
Good patient message: "I've been in the hospital for 5 days after being diagnosed with a big heart attack. The doctor told me the whole front wall of my heart is not moving like it should. Now today, I can't feel my legs and they feel cold."

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
- Physical exam findings should be expressed as what a patient experiences or what a doctor told them (e.g., "the doctor said my lungs sound crackly" not "crackles on auscultation").
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
        model_name: str = shared.AI_FIREWORKS_MODEL,
        max_rounds: int = 2,
        temperature_rewrite: float = 0.4,
        temperature_verify: float = 0.0,
        max_workers: int = 4,
):
    # Collect all tasks: (bucket, index, conditions)
    tasks: List[Tuple[str, int, List[Dict[str, Any]]]] = []

    for bucket, items in question['sampled_sets'].items():
        if bucket not in ["answerable", "hard_but_fair", "boundary_tests"]:
            logger.warning(f"Unknown bucket '{bucket}' encountered. Skipping.")
            continue

        logger.info(f"Rewriting {len(items)} items in bucket '{bucket}' for question ID {question['question_no']}.")
        if bucket == "boundary_tests":
            for i, pair in enumerate(items):
                tasks.append((bucket, i, pair['variant_drop']))
        else:
            for i, item in enumerate(items):
                tasks.append((bucket, i, item))

    # Run tasks concurrently
    results_map: Dict[Tuple[str, int], str] = {}

    def _run_task(bucket: str, idx: int, conditions: List[Dict[str, Any]]) -> Tuple[str, int, str]:
        thread_client = shared.get_client()
        prompt, _ = rewrite_with_self_check(
            conditions,
            client=thread_client,
            model_name=model_name,
            max_rounds=max_rounds,
            temperature_rewrite=temperature_rewrite,
            temperature_verify=temperature_verify,
        )
        return bucket, idx, prompt

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(_run_task, *task))
            time.sleep(0.5)  # Stagger submissions to avoid rate limits

        for future in as_completed(futures):
            try:
                bucket, idx, prompt = future.result()
                results_map[(bucket, idx)] = prompt
            except Exception as e:
                logger.error(f"Rewrite task failed: {e}")

    # Assemble results in original order
    patient_prompts: Dict[str, List[str]] = {
        "answerable": [],
        "hard_but_fair": [],
        "boundary_tests": [],
    }
    for bucket in ["answerable", "hard_but_fair", "boundary_tests"]:
        items = question['sampled_sets'].get(bucket, [])
        for i in range(len(items)):
            patient_prompts[bucket].append(results_map.get((bucket, i), ""))

    return patient_prompts

def llm_rewrite_all(
        input_file: str,
        output_file: str,
        model_name: str = shared.AI_FIREWORKS_MODEL,
        max_rounds: int = 2,
        temperature_rewrite: float = 0.9,
        temperature_verify: float = 0.6,
        max_workers: int = 4,
):
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
            model_name=model_name,
            max_rounds=max_rounds,
            temperature_rewrite=temperature_rewrite,
            temperature_verify=temperature_verify,
            max_workers=max_workers,
        )
        question['patient_prompts'] = patient_prompts
        results.append(question)
    
    # save results
    shared.save_json(results, output_file)
    logger.info(f"Saved rewritten prompts for {len(results)} questions to {output_file}")

# python -m pipeline.rewrite --input_file=data/cardiology_usmle_sampled_v1.json --output_file=data/cardiology_usmle_rewritten_v1.json
if __name__ == "__main__":
    fire.Fire(llm_rewrite_all)