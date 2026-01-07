##################################################
# Create Benchmark from Sampled Data
##################################################

import os
import json
import logging
from typing import List, Dict, Any
from . import shared
import random
import fire

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_samples(question: Dict[str, Any]) -> Dict[str, Any]:
    """Create benchmark samples from a question entry."""
    samples = []
    for bucket, prompts in question['patient_prompts'].items():
        for prompt in prompts:
            # Add the last question from the original question and the choices to prompt
            prompt = prompt.strip() + "\n" + question['question_context'].split(".")[-1] + "\n" + question['options']
            sample = {
                "prompt": prompt,
                "bucket": bucket,
                "metadata": {k: v for k, v in question.items() if k != "patient_prompts"},
            }
            samples.append(sample)
    return samples
    

def create_benchmark(
    input_file: str,
    output_file: str,
    seed: int = 42,
) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    with open(input_file, 'r') as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} entries from {input_file}")

    # Filter to get question with .1 only
    data = [entry for entry in data if str(entry.get("question_no", "")).endswith(".1")]
    logger.info(f"Filtered to {len(data)} entries with question_no ending in .1")

    # Create benchmark samples
    benchmark = []
    for question in data:
        samples = create_samples(question)
        benchmark.extend(samples)
    
    logger.info(f"Created {len(benchmark)} benchmark samples")
    # Shuffle the benchmark samples
    random.seed(seed)
    random.shuffle(benchmark)
    logger.info("Shuffled benchmark samples")
    # Save benchmark samples
    shared.save_json(benchmark, output_file)
    logger.info(f"Saved benchmark samples to {output_file}")

# python -m pipeline.finalize --input_file=data/cardiology_usmle_rewritten_v1.json --output_file=data/cardiology_usmle_synthetic_benchmark_v1.json
if __name__ == "__main__":
    fire.Fire(create_benchmark)