#########################################
# CLASSIFY CLINICAL CONDITIONS
#########################################

import os
import logging
import json
import fire
from typing import List, Dict, Any
from . import shared

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """
You are a clinical information classifier. You will be given a list of clinical items. Each item is a single, standalone factual statement.
Your task is to classify EACH item into exactly ONE bucket and assign a confidence score.

## BUCKET DEFINITIONS
Choose ONE of the following buckets for each item:
1. DEMOGRAPHICS: Age, sex, pregnancy status, ethnicity, baseline characteristics
2. RISK_FACTORS: Past medical history, lifestyle, genetics, exposures, medications, comorbidities
3. SYMPTOMS: Subjective complaints reported by the patient
4. TIMING_COURSE: Onset, duration, progression, acuity, sequence, episodic vs chronic
5. PHYSICAL_EXAM: Objective findings on physical examination
6. LABS_IMAGING: Laboratory values, imaging findings, diagnostic test results
7. NEGATIVE_FINDINGS: Explicit denials or absence of findings (e.g., “no fever”, “denies chest pain”)
8. CONTEXT: Situational or environmental context (recent travel, hospitalization, surgery, delivery, trauma)
9. OTHER: Information that does not clearly fit any category above (use only if necessary)

## RULES
- Assign EXACTLY ONE bucket per condition.
- Do NOT infer diagnosis, intent, or implications.
- Do NOT rephrase or modify the condition text.
- Classify based ONLY on the content of the condition itself.
- If uncertain between two buckets, choose the MOST DIRECT interpretation.
- Use OTHER only as a last resort.
- Confidence reflects how clearly the information fits the chosen bucket, NOT clinical importance.

## CONFIDENCE SCORE
Assign a confidence score from 0.00 to 1.00:
- 0.90–1.00: Clear, unambiguous fit
- 0.70–0.89: Minor ambiguity
- 0.50–0.69: Moderate ambiguity
- <0.50: Poor fit (should be rare)

## INPUT FORMAT
<conditions>
1. Condition text here
2. Condition text here
...
</conditions>

## OUTPUT FORMAT (STRICT JSON)
Return a JSON array.
Each element MUST contain ONLY the following fields:
- "condition_text": string (exact copy) without numbered list prefix
- "bucket": one of the allowed bucket names
- "confidence": number between 0.00 and 1.00
- "rationale": string (≤ 20 words)

Do NOT include any additional keys.
Do NOT include extra commentary or formatting.

## YOUR TURN:
"""

def create_classification_prompt(question: Dict[str, Any]) -> str:
    condition_list_str = "\n".join([f"{i+1}. {item['condition']}" for i, item in enumerate(question['llm_extracted_info'])])
    user_prompt_classify = f"{CLASSIFY_PROMPT}\n\n<conditions>\n{condition_list_str}\n</conditions>"
    return user_prompt_classify

def verify_bucket_classification(output_classify: Dict[str, Any]) -> bool:
    valid_buckets = {
        "DEMOGRAPHICS", "RISK_FACTORS", "SYMPTOMS", "TIMING_COURSE",
        "PHYSICAL_EXAM", "LABS_IMAGING", "NEGATIVE_FINDINGS", "CONTEXT", "OTHER"
    }
    bucket = output_classify.get("bucket", "").strip()
    confidence = output_classify.get("confidence", -1)
    rationale = output_classify.get("rationale", "").strip()
    
    if bucket not in valid_buckets:
        logger.error(f"Invalid bucket: {bucket}")
        return False
    if not (0.0 <= confidence <= 1.0):
        logger.error(f"Invalid confidence score: {confidence}")
        return False
    if len(rationale) == 0:
        logger.error("Rationale is empty!")
        return False
    return True

def classify_clinical_items_per_question(
    question: Dict[str, Any],
    client: Any,
    model_name: str = shared.AI_FIREWORKS_MODEL,
) -> List[Dict[str, Any]]:
    user_prompt_classify = create_classification_prompt(question)
    
    response_classify = shared.llm_generate(
        user_prompt=user_prompt_classify,
        model_name=model_name,
        client=client,
        temperature=0.6,
        top_p=1,
        max_tokens=4090,
    )
    return response_classify

def get_bucket_distribution_per_question(
    question: Dict[str, Any]
) -> Dict[str, int]:
    bucket_counts = {
        "DEMOGRAPHICS": 0,
        "RISK_FACTORS": 0,
        "SYMPTOMS": 0,
        "TIMING_COURSE": 0,
        "PHYSICAL_EXAM": 0,
        "LABS_IMAGING": 0,
        "NEGATIVE_FINDINGS": 0,
        "CONTEXT": 0,
        "OTHER": 0
    }
    for item in question.get('llm_extracted_info', []):
        bucket = item.get('bucket', None)
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1
    return bucket_counts    

def classify_all(
    input_file: str,
    output_file: str,
    model_name: str = shared.AI_FIREWORKS_MODEL
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    with open(input_file, 'r') as f:
        questions = json.load(f)
    
    logger.info(f"Loaded {len(questions)} questions from {input_file}")
    
    client = shared.get_client()
    results = []
    total_bucket_counts = {
        "DEMOGRAPHICS": 0,
        "RISK_FACTORS": 0,
        "SYMPTOMS": 0,
        "TIMING_COURSE": 0,
        "PHYSICAL_EXAM": 0,
        "LABS_IMAGING": 0,
        "NEGATIVE_FINDINGS": 0,
        "CONTEXT": 0,
        "OTHER": 0
    }

    for idx, question in enumerate(questions):
        classification_output = classify_clinical_items_per_question(question, client, model_name)

        try:
            classification_output_json = json.loads(classification_output)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error for question {idx}: {e}")
            # Retry 3 times before failing
            for retry in range(3):
                logger.info(f"Retrying classification for question {idx}, attempt {retry + 1}")
                classification_output = classify_clinical_items_per_question(question, client, model_name)
                try:
                    classification_output_json = json.loads(classification_output)
                    logger.info(f"Successfully parsed JSON on retry {retry + 1} for question {idx}")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Retry {retry + 1} failed for question {idx}: {e}")
            else:
                logger.error(f"Failed to classify question {idx} after retries. Skipping.")
                continue
        
        # Add classification results to the question
        for item, classification in zip(question['llm_extracted_info'], classification_output_json):
            try:
                # strip off numbered list prefix from condition text
                classification['condition_text'] = classification['condition_text'].split(". ", 1)[1] if ". " in classification['condition_text'] else classification['condition_text']
            except:
                logger.warning(f"Failed to strip numbered prefix for question {idx}, classification: {classification}.")
            
            try:
                assert item['condition'] == classification['condition_text']
            except AssertionError:
                logger.warning(f"Condition text mismatch for question {idx}, item: {item['condition']}, classification: {classification['condition_text']}. Skipping this item.")
                continue

            if verify_bucket_classification(classification):
                item.update(classification)
            else:
                logger.warning(f"Invalid classification format for question {idx}, classification: {classification}. Skipping this item.")
                continue
        results.append(question)

        # Update total bucket counts
        bucket_counts = get_bucket_distribution_per_question(question)
        for bucket, count in bucket_counts.items():
            total_bucket_counts[bucket] += count

        if (idx + 1) % 5 == 0:
            logger.info(f"Processed {idx + 1}/{len(questions)} questions.")
    # Save results to output file
    shared.save_json(results, output_file)
    logger.info("=" * 20 + " CLASSIFICATION SUMMARY " + "=" * 20)
    logger.info(f"Saved classified results for {len(results)} questions to {output_file}")
    # Log average bucket distribution
    logger.info("Average bucket distribution per question:")
    avg_bucket_counts = {bucket: count / len(results) for bucket, count in total_bucket_counts.items()}
    for bucket, avg_count in avg_bucket_counts.items():
        logger.info(f"{bucket}: {avg_count:.2f}")
    logger.info("=" * 60)
    # Log total bucket distribution
    logger.info("Total bucket distribution across all questions:")
    for bucket, total_count in total_bucket_counts.items():
        logger.info(f"{bucket}: {total_count}")

# python -m pipeline.classify --input_file=data/cardiology_usmle_extracted_v1.json --output_file=data/cardiology_usmle_classified_v1.json
if __name__ == "__main__":
    fire.Fire(classify_all)