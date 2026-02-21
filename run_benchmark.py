##################################
# Run benchmark 
##################################

import os
import json
import logging
import fire
import yaml
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline import shared
from typing import List, Optional

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "gpt-5.2": shared.AI_GPT_MODEL,
    "claude": shared.AI_CLAUDE_MODEL,
    "gpt-oss-120b": shared.AI_FIREWORKS_MODEL,
}

SYSTEM_PROMPT = """
IMPORTANT OUTPUT REQUIREMENTS — MUST FOLLOW EXACTLY
You MAY include reasoning, justification, or explanation earlier in your response.
However, the VERY LAST PART of your response MUST contain your final answer AND a
confidence score, following the rules below.

FINAL ANSWER RULES:
1. The final answer MUST appear at the END of your response.
2. Wrap the final answer using EXACTLY these tags <final_answer> and </final_answer>
3. Inside <final_answer>, write ONLY the selected answer option.
   - Do NOT include explanations, reasoning, or commentary inside the tags.
4. The answer inside <final_answer> MUST be an EXACT QUOTE from the question's answer choices.
   - Include the option letter in parentheses.
   - Preserve the original wording exactly as written.

CONFIDENCE SCORE RULES:
1. Immediately AFTER </final_answer>, provide a confidence reasoning block between these tags: <confidence_reasoning> and </confidence_reasoning>
2. Inside <confidence_reasoning>, write 1-3 sentences explaining WHY you chose this
   confidence level. Address:
   - What key information was present that supports your answer?
   - What critical information was missing or ambiguous (if any)?
   - Were there other plausible answer choices you considered?
3. Immediately AFTER </confidence_reasoning>, provide the confidence score between <confidence> and </confidence>
4. Inside <confidence>, write ONLY a single integer from 0 to 100.
   - 0 means you are purely guessing with no basis.
   - 100 means you are absolutely certain the answer is correct.
   - Calibrate honestly: if you think there is a 60% chance you are correct, write 60.
5. Your confidence should reflect how sufficient the PROVIDED information is to answer
   the question. If critical information is missing, your confidence should be LOW even
   if you can make an educated guess.
6. Do NOT default to high confidence. Be honest about uncertainty.

Correct example (EXAMPLE ONLY — DO NOT COPY THIS EXACT CONTENT):
<final_answer>(D) Discontinuation of lisinopril</final_answer>
<confidence_reasoning>The patient's persistent dry cough starting after initiation of lisinopril is a classic ACE inhibitor side effect. No other medication changes or respiratory conditions were mentioned. I am fairly confident but not certain, as other causes of cough were not fully excluded.</confidence_reasoning>
<confidence>82</confidence>

Incorrect examples:
- Answer not at the end of the response
- Any text after </confidence>
- Paraphrased option text
- Missing option letter
- Explanation inside <final_answer>
- Missing <confidence_reasoning> or <confidence> tags
- Non-integer or out-of-range confidence value

Failure to follow these rules is an incorrect response."""

def extract_confidence(model_response: str) -> Optional[int]:
    """Extract the confidence score (0-100) from the model response."""
    raw = shared.extract_between_tags(model_response, "<confidence>", "</confidence>")
    if raw is None:
        logger.warning("Could not extract confidence score from model response.")
        return None
    try:
        score = int(raw.strip())
        if 0 <= score <= 100:
            return score
        logger.warning(f"Confidence score {score} out of range 0-100.")
        return None
    except ValueError:
        logger.warning(f"Could not parse confidence score as integer: {raw}")
        return None


def extract_confidence_reasoning(model_response: str) -> Optional[str]:
    """Extract the confidence reasoning from the model response."""
    raw = shared.extract_between_tags(
        model_response, "<confidence_reasoning>", "</confidence_reasoning>"
    )
    if raw is None:
        logger.warning("Could not extract confidence reasoning from model response.")
        return None
    return raw.strip()


def compute_accuracy(model_response: str, correct_answer: str) -> tuple:
    """Compute if the model response matches the correct answer.

    Returns:
        (is_correct, answer_extracted): is_correct is True if the answer matches,
        answer_extracted is False if <final_answer> tags were missing.
    """
    start_tag = "<final_answer>"
    end_tag = "</final_answer>"
    extracted_answer = shared.extract_between_tags(model_response, start_tag, end_tag)
    if extracted_answer is None:
        logger.warning(f"Could not extract final answer from model response: {model_response}")
        return False, False
    return extracted_answer.strip() == correct_answer.strip(), True

def compute_average_accuracy(results: List) -> float:
    """Compute average accuracy from the results."""
    if not results:
        return 0.0
    correct_count = sum(1 for result in results if result.get("is_correct", False))
    return correct_count / len(results)

def compute_bucketed_accuracy(results: List) -> dict:
    """Compute accuracy per bucket."""
    bucket_totals = {}
    bucket_corrects = {}
    for result in results:
        bucket = result.get("bucket", "unknown")
        is_correct = result.get("is_correct", False)
        bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1
        if is_correct:
            bucket_corrects[bucket] = bucket_corrects.get(bucket, 0) + 1
    bucket_accuracies = {}
    for bucket, total in bucket_totals.items():
        correct = bucket_corrects.get(bucket, 0)
        bucket_accuracies[bucket] = correct / total
    return bucket_accuracies


def compute_average_confidence(results: List) -> Optional[float]:
    """Compute average confidence score from results (ignoring None values)."""
    scores = [r["confidence"] for r in results if r.get("confidence") is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def compute_bucketed_confidence(results: List) -> dict:
    """Compute average confidence per bucket."""
    bucket_scores: dict = {}
    for result in results:
        bucket = result.get("bucket", "unknown")
        conf = result.get("confidence")
        if conf is not None:
            bucket_scores.setdefault(bucket, []).append(conf)
    return {b: sum(s) / len(s) for b, s in bucket_scores.items()}

def load_config(config_file: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist.")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_file}")
    return config

def run_benchmark(
    benchmark_file: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    reasoning_effort: str = "high",
    include_original_questions: bool = False,
    max_workers: int = 4,
    config_file: Optional[str] = None,
) -> None:
    """Run benchmark on the specified model.

    Args:
        benchmark_file: Path to the benchmark JSON file.
        model_name: Name of the model to use.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        max_tokens: Maximum tokens in response.
        reasoning_effort: Reasoning effort level.
        include_original_questions: If True, also evaluate the model on the original
            USMLE question stems (metadata.question) alongside the patient prompts.
            Original-question results are stored with bucket="original".
        max_workers: Number of concurrent threads for LLM calls (default 4).
        config_file: Path to YAML config file. If provided, overrides other arguments.
    """
    # Load config from file if provided
    if config_file:
        config = load_config(config_file)
        benchmark_file = config.get("benchmark_file", benchmark_file)
        model_name = config.get("model_name", model_name)
        temperature = config.get("temperature", temperature)
        top_p = config.get("top_p", top_p)
        max_tokens = config.get("max_tokens", max_tokens)
        reasoning_effort = config.get("reasoning_effort", reasoning_effort)
        include_original_questions = config.get("include_original_questions", include_original_questions)
        max_workers = config.get("max_workers", max_workers)

    # Validate required arguments
    if not benchmark_file:
        raise ValueError("benchmark_file is required. Provide it via argument or config file.")
    if not model_name:
        raise ValueError("model_name is required. Provide it via argument or config file.")

    if not os.path.exists(benchmark_file):
        raise FileNotFoundError(f"Benchmark file {benchmark_file} does not exist.")
    
    with open(benchmark_file, 'r') as f:
        benchmark = json.load(f)

    logger.info(f"Loaded {len(benchmark)} benchmark samples from {benchmark_file}")

    # ---- Optionally add original-question samples ----
    # Deduplicate by question_no so each original question is evaluated once.
    if include_original_questions:
        seen_questions = set()
        original_samples = []
        for sample in benchmark:
            meta = sample.get("metadata", {})
            q_no = meta.get("question_no")
            if q_no in seen_questions:
                continue
            seen_questions.add(q_no)
            original_question = meta.get("question", "")
            if not original_question:
                continue
            original_samples.append({
                "prompt_id": f"original-{q_no}",
                "prompt": original_question,
                "bucket": "original",
                "metadata": meta,
            })
        benchmark = benchmark + original_samples
        logger.info(f"Added {len(original_samples)} original-question samples (total: {len(benchmark)})")

    results = []

    model = MODEL_MAPPING.get(model_name, None)
    if model is None:
        if "claude" in model_name:
            model = f"anthropic:{model_name}"
        elif "gpt" in model_name:
            model = f"openai:{model_name}"
        else:
            raise ValueError(f"Model name {model_name} is not recognized. Available models: {list(MODEL_MAPPING.keys())}")

    def _process_sample(idx: int, sample: dict) -> Optional[dict]:
        thread_client = shared.get_client()
        prompt = sample['prompt']
        bucket = sample['bucket']
        metadata = sample.get('metadata', {})
        prompt_id = sample.get('prompt_id', str(uuid.uuid4()))

        logger.info(f"Processing sample {idx + 1}/{len(benchmark)} (ID: {prompt_id}) in bucket '{bucket}'")

        try:
            response = shared.llm_generate(
                user_prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model_name=model,
                client=thread_client,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            correct_answer = metadata.get("correct_response", "")
            is_correct, answer_extracted = compute_accuracy(response, correct_answer)
            confidence = extract_confidence(response)
            confidence_reasoning = extract_confidence_reasoning(response)
            invalid_format = not answer_extracted or confidence is None

            return {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": response,
                "bucket": bucket,
                "model_name": model,
                "sampling_config": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "reasoning_effort": reasoning_effort,
                },
                "is_correct": is_correct,
                "answer_extracted": answer_extracted,
                "confidence": confidence,
                "confidence_reasoning": confidence_reasoning,
                "invalid_format": invalid_format,
                "metadata": metadata,
            }
        except Exception as e:
            logger.error(f"Failed to process sample {idx + 1} (ID: {prompt_id}): {e}")
            return None

    logger.info(f"Running benchmark with {max_workers} concurrent workers.")
    results_map: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, sample in enumerate(benchmark):
            future = executor.submit(_process_sample, idx, sample)
            futures[future] = idx
            time.sleep(0.5)  # Stagger submissions to avoid rate limits

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results_map[idx] = result
            except Exception as e:
                logger.error(f"Benchmark task failed for sample {idx + 1}: {e}")

    # Assemble results in original order
    for idx in range(len(benchmark)):
        if idx in results_map:
            results.append(results_map[idx])

    if not os.path.exists("results/raw_results"):
        os.makedirs("results/raw_results")
    
    output_file = f"results/raw_results/benchmark_raw_results_{model_name.replace('/', '_')}_{int(time.time())}.json"
    shared.save_json(results, output_file)
    logger.info(f"Saved raw benchmark results to {output_file}")

    # ---- Split results into rewritten (patient) prompts vs original prompts ----
    rewritten_results = [r for r in results if r.get("bucket") != "original"]
    original_results = [r for r in results if r.get("bucket") == "original"]

    def _compute_section_metrics(section_results: List) -> dict:
        invalid_count = sum(1 for r in section_results if r.get("invalid_format", False))
        return {
            "average_accuracy": compute_average_accuracy(section_results),
            "bucketed_accuracy": compute_bucketed_accuracy(section_results),
            "average_confidence": compute_average_confidence(section_results),
            "bucketed_confidence": compute_bucketed_confidence(section_results),
            "total_samples": len(section_results),
            "invalid_format_count": invalid_count,
        }

    def _log_section(label: str, metrics: dict) -> None:
        logger.info(f"--- {label} ({metrics['total_samples']} samples) ---")
        logger.info(f"  Average Accuracy: {metrics['average_accuracy']:.4f}")
        avg_conf = metrics["average_confidence"]
        if avg_conf is not None:
            logger.info(f"  Average Confidence: {avg_conf:.1f}")
        invalid = metrics.get("invalid_format_count", 0)
        if invalid > 0:
            logger.warning(f"  Invalid Format: {invalid}/{metrics['total_samples']} responses missing answer or confidence tags")
        all_buckets = sorted(set(
            list(metrics["bucketed_accuracy"].keys())
            + list(metrics["bucketed_confidence"].keys())
        ))
        for bucket in all_buckets:
            acc = metrics["bucketed_accuracy"].get(bucket)
            conf = metrics["bucketed_confidence"].get(bucket)
            parts = []
            if acc is not None:
                parts.append(f"Accuracy: {acc:.4f}")
            if conf is not None:
                parts.append(f"Avg Confidence: {conf:.1f}")
            logger.info(f"  Bucket '{bucket}' — {', '.join(parts)}")

    rewritten_metrics = _compute_section_metrics(rewritten_results)
    logger.info("=" * 20 + " RESULTS " + "=" * 20)
    _log_section("Rewritten (patient) prompts", rewritten_metrics)

    original_metrics = None
    if original_results:
        original_metrics = _compute_section_metrics(original_results)
        _log_section("Original question stems", original_metrics)

    # Save summary
    total_invalid = sum(1 for r in results if r.get("invalid_format", False))
    summary = {
        "model_name": model_name,
        "rewritten_prompts": rewritten_metrics,
        "original_prompts": original_metrics,
        "total_samples": len(results),
        "invalid_format_count": total_invalid,
        "include_original_questions": include_original_questions,
        "sampling_config": {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        },
    }
    if not os.path.exists("results/aggregated"):
        os.makedirs("results/aggregated")
    summary_file = f"results/aggregated/benchmark_summary_{model_name.replace('/', '_')}_{int(time.time())}.json"
    shared.save_json(summary, summary_file)
    logger.info(f"Saved benchmark summary to {summary_file}")

if __name__ == "__main__":
    fire.Fire(run_benchmark)