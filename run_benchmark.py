##################################
# Run benchmark 
##################################

import os
import json
import logging
import fire
from pipeline import shared
from typing import List

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
At the END of your response, you MUST include a final answer section.
Formatting rules:
1. Wrap your final answer in the following XML-style tags:
   <final_answer> and </final_answer>

2. Inside <final_answer>, write ONLY the selected answer option.
   - Do NOT add explanations, comments, or extra text inside the tags.

3. The answer inside <final_answer> MUST be an EXACT QUOTE from the question’s answer choices.
   - Include the option letter in parentheses.
   - Preserve the original wording exactly as written in the question.

Example of a correct format:
<final_answer>(D) Discontinuation of lisinopril</final_answer>

Incorrect formats (DO NOT DO THESE):
- Writing the answer without tags
- Paraphrasing the option text
- Omitting the option letter
- Adding justification inside <final_answer>

Failure to follow these rules is an incorrect response."""

def compute_accuracy(model_response: str, correct_answer: str) -> bool:
    """Compute if the model response matches the correct answer."""
    start_tag = "<final_answer>"
    end_tag = "</final_answer>"
    extracted_answer = shared.extract_between_tags(model_response, start_tag, end_tag)
    if extracted_answer is None:
        logger.warning(f"Could not extract final answer from model response: {model_response}")
        return False
    return extracted_answer.strip() == correct_answer.strip()

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

def run_benchmark(
    benchmark_file: str,
    model_name: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    reasoning_effort: str = "high",
) -> None:
    """Run benchmark on the specified model."""
    if not os.path.exists(benchmark_file):
        raise FileNotFoundError(f"Benchmark file {benchmark_file} does not exist.")
    
    with open(benchmark_file, 'r') as f:
        benchmark = json.load(f)

    logger.info(f"Loaded {len(benchmark)} benchmark samples from {benchmark_file}")

    results = []
    client = shared.get_client()

    model = MODEL_MAPPING.get(model_name, None)
    if model is None:
        raise ValueError(f"Model name {model_name} is not recognized. Available models: {list(MODEL_MAPPING.keys())}")

    for idx, sample in enumerate(benchmark):
        prompt = sample['prompt']
        bucket = sample['bucket']
        metadata = sample.get('metadata', {})

        logger.info(f"Processing sample {idx + 1}/{len(benchmark)} in bucket '{bucket}'")

        try:
            response = shared.llm_generate(
                user_prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                model_name=model,
                client=client,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            # Compute accuracies
            correct_answer = metadata.get("correct_response", "")
            is_correct = compute_accuracy(response, correct_answer)

            result = {
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
                "metadata": metadata,
            }
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process sample {idx + 1}: {e}")

    if not os.path.exists("results/raw_results"):
        os.makedirs("results/raw_results")
    
    output_file = f"results/raw_results/benchmark_raw_results_{model_name.replace('/', '_')}.json"
    shared.save_json(results, output_file)
    logger.info(f"Saved raw benchmark results to {output_file}")

    # Compute and log average accuracy
    average_accuracy = compute_average_accuracy(results)
    bucketed_accuracy = compute_bucketed_accuracy(results)
    logger.info(f"Average Accuracy for model {model_name}: {average_accuracy:.4f}")
    for bucket, acc in bucketed_accuracy.items():
        logger.info(f"Bucket '{bucket}' Accuracy: {acc:.4f}")
    # Save summary
    summary = {
        "model_name": model_name,
        "average_accuracy": average_accuracy,
        "bucketed_accuracy": bucketed_accuracy,
        "total_samples": len(results),
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
    summary_file = f"results/aggregated/benchmark_summary_{model_name.replace('/', '_')}.json"
    shared.save_json(summary, summary_file)
    logger.info(f"Saved benchmark summary to {summary_file}")


# python -m run_benchmark --benchmark_file=data/cardiology_usmle_synthetic_benchmark_v1.json --model_name=gpt-oss-120b --temperature=0.6
if __name__ == "__main__":
    fire.Fire(run_benchmark)