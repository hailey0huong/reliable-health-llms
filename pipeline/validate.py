import os
import json
import logging
import random
import time
import fire
from typing import Any, Dict, List, Optional

from . import shared, sample as sample_mod

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COMPLETENESS_PROMPT = """You are a clinical reasoning validator.

You will be given:
1. A set of clinical facts about a patient.
2. A multiple-choice question (the final question stem + answer choices).

Your task:
Decide whether the clinical facts provided contain ENOUGH information to
confidently select a SINGLE best answer from the choices.

Rules:
- Judge ONLY from the facts provided. Do NOT use outside knowledge to fill gaps.
- "Enough information" means a competent clinician could select the best answer
  with reasonable confidence using these facts alone.
- If a critical piece of information (e.g., patient age, key lab value, vital sign,
  timing of symptoms) is missing and its absence makes two or more answer choices
  equally plausible, answer NO.

Return STRICT JSON only (no markdown, no commentary):
{
  "answerable": true or false,
  "reasoning": "Brief explanation (1-2 sentences) of why the facts are or are not sufficient.",
  "missing_categories": ["list of missing info categories if answerable is false, else empty"]
}
"""


def _build_validation_input(
    conditions: List[Dict[str, Any]],
    question: Dict[str, Any],
) -> str:
    """Build the user prompt for the completeness check."""
    facts = "\n".join(
        f"{i}. {c['condition']}"
        for i, c in enumerate(conditions, 1)
    )
    # Grab the question
    question_text = question.get("question", "").strip()
    return (
        f"Clinical facts:\n{facts}\n\n"
        f"Question: {question_text}"
    )


def validate_set(
    conditions: List[Dict[str, Any]],
    question: Dict[str, Any],
    client: Any,
    model_name: str,
) -> Dict[str, Any]:
    """Run the completeness LLM check on one set of conditions.
    """
    user_prompt = (
        COMPLETENESS_PROMPT
        + "\n\nINPUT:\n"
        + _build_validation_input(conditions, question)
    )
    for attempt in range(3):
        raw = shared.llm_generate(
            user_prompt=user_prompt,
            client=client,
            model_name=model_name,
            temperature=0.1,
            max_tokens=1024,
        )
        if raw is not None:
            break
        wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
        logger.warning(f"LLM returned None for completeness check (attempt {attempt + 1}/3). Waiting {wait}s...")
        time.sleep(wait)
    result = shared.safe_json_load(raw) if raw is not None else None
    if result is None:
        logger.warning(f"Could not parse completeness-check response as JSON. \n{raw}")
        return {"answerable": False, "reasoning": "Parse failure", "missing_categories": []}
    return result


def validate_and_resample_question(
    question: Dict[str, Any],
    client: Any,
    model_name: str,
    max_retries: int = 2,
    sample_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Validate every answerable set for a question; re-sample failures.

    Mutates question['sampled_sets'] in place and returns it.
    Adds question['validation_flags'] with per-set results.
    """
    sampled = question.get("sampled_sets", {})
    answerable_sets: List[List[Dict]] = sampled.get("answerable", [])
    items = question.get("llm_extracted_info", [])

    rng = random.Random(sample_seed)
    flags: List[Dict[str, Any]] = []

    new_answerable: List[List[Dict]] = []

    for idx, cond_set in enumerate(answerable_sets):
        if idx > 0:
            time.sleep(1)  # Throttle to avoid rate limiting
        result = validate_set(cond_set, question, client, model_name)
        passed = result.get("answerable", False)

        if passed:
            flags.append({"set_index": idx, "passed": True})
            new_answerable.append(cond_set)
            continue

        logger.warning(
            f"Q {question.get('question_no')}: answerable set {idx} failed "
            f"completeness check — {result.get('reasoning', '')}. "
            f"Missing: {result.get('missing_categories', [])}"
        )

        # Re-sample up to max_retries times
        replaced = False
        for attempt in range(max_retries):
            candidate = sample_mod._sample_answerable(rng, items)
            if not candidate:
                continue
            retry_result = validate_set(candidate, question, client, model_name)
            if retry_result.get("answerable", False):
                new_answerable.append(candidate)
                replaced = True
                logger.info(
                    f"Q {question.get('question_no')}: replaced set {idx} "
                    f"on attempt {attempt + 1}."
                )
                break

        if not replaced:
            # Keep the original but flag it
            new_answerable.append(cond_set)
            logger.warning(
                f"Q {question.get('question_no')}: could not find valid "
                f"replacement for set {idx} after {max_retries} retries. "
                f"Keeping original (flagged)."
            )

        flags.append({
            "set_index": idx,
            "passed": False,
            "replaced": replaced,
            "reasoning": result.get("reasoning", ""),
            "missing_categories": result.get("missing_categories", []),
        })

    sampled["answerable"] = new_answerable
    question["sampled_sets"] = sampled
    question["validation_flags"] = flags
    return question


def validate_all(
    input_file: str,
    output_file: str,
    model_name: str = shared.AI_FIREWORKS_MODEL,
    max_retries: int = 2,
    sample_seed: int = 42,
) -> None:
    """Run completeness validation on all questions in sampled JSON."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    with open(input_file, 'r') as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} questions from {input_file}")
    client = shared.get_client()

    total_flagged = 0
    total_replaced = 0
    total_sets = 0

    for idx, question in enumerate(questions):
        validate_and_resample_question(
            question,
            client=client,
            model_name=model_name,
            max_retries=max_retries,
            sample_seed=sample_seed,
        )
        flags = question.get("validation_flags", [])
        n_failed = sum(1 for f in flags if not f.get("passed", True))
        n_replaced = sum(1 for f in flags if f.get("replaced", False))
        total_flagged += n_failed
        total_replaced += n_replaced
        total_sets += len(flags)

        if (idx + 1) % 5 == 0:
            logger.info(f"Validated {idx + 1}/{len(questions)} questions.")

    shared.save_json(questions, output_file)

    logger.info("=" * 20 + " VALIDATION SUMMARY " + "=" * 20)
    logger.info(f"Total answerable sets checked: {total_sets}")
    logger.info(f"Sets that failed completeness check: {total_flagged}")
    logger.info(f"Sets successfully replaced via re-sampling: {total_replaced}")
    logger.info(f"Sets still flagged (kept original): {total_flagged - total_replaced}")
    logger.info(f"Saved validated output to {output_file}")


# python -m pipeline.validate --input_file=data/cardiology_usmle_sampled_v1.json --output_file=data/cardiology_usmle_validated_v1.json
if __name__ == "__main__":
    fire.Fire(validate_all)
