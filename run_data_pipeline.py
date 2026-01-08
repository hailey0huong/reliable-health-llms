###########################################
# Run data generation pipeline end to end
###########################################

import os
import logging
import fire
import yaml
from typing import Optional

from pipeline import extract, classify, sample, rewrite, finalize, shared

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist.")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_file}")
    return config


def run_pipeline(
    config_file: Optional[str] = None,
    input_file: Optional[str] = None,
    output_dir: Optional[str] = "data",
    output_prefix: Optional[str] = "pipeline",
    model_name: Optional[str] = None,
    steps: Optional[str] = "all",  # "all" or comma-separated: "extract,classify,sample,rewrite,finalize"
    # Step-specific parameters
    n_samples: int = 12,
    sample_seed: int = 42,
    rewrite_max_rounds: int = 2,
    finalize_seed: int = 42,
) -> None:
    """
    Run the data generation pipeline end-to-end.
    
    Args:
        config_file: Path to YAML config file (overrides other args if provided)
        input_file: Path to input JSON file with USMLE questions
        output_dir: Directory for output files
        output_prefix: Prefix for output filenames
        model_name: LLM model name to use
        steps: Which steps to run ("all" or comma-separated list)
        n_samples: Number of samples to generate in sampling step
        sample_seed: Random seed for sampling
        rewrite_max_rounds: Max verification rounds for rewriting
        finalize_seed: Random seed for final shuffle
    """
    # Load config from file if provided
    if config_file:
        config = load_config(config_file)
        input_file = config.get("input_file", input_file)
        output_dir = config.get("output_dir", output_dir)
        output_prefix = config.get("output_prefix", output_prefix)
        model_name = config.get("model_name", model_name)
        steps = config.get("steps", steps)
        n_samples = config.get("n_samples", n_samples)
        sample_seed = config.get("sample_seed", sample_seed)
        rewrite_max_rounds = config.get("rewrite_max_rounds", rewrite_max_rounds)
        finalize_seed = config.get("finalize_seed", finalize_seed)
    
    # Validate required args
    if not input_file:
        raise ValueError("input_file is required. Provide via argument or config file.")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    # Default model
    if not model_name:
        model_name = shared.AI_FIREWORKS_MODEL
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Define intermediate file paths
    extracted_file = os.path.join(output_dir, f"{output_prefix}_extracted.json")
    classified_file = os.path.join(output_dir, f"{output_prefix}_classified.json")
    sampled_file = os.path.join(output_dir, f"{output_prefix}_sampled.json")
    rewritten_file = os.path.join(output_dir, f"{output_prefix}_rewritten.json")
    final_file = os.path.join(output_dir, f"{output_prefix}_benchmark.json")
    
    # Parse steps
    if steps == "all":
        steps_to_run = ["extract", "classify", "sample", "rewrite", "finalize"]
    else:
        steps_to_run = [s.strip().lower() for s in steps.split(",")]
    
    logger.info("=" * 60)
    logger.info("PIPELINE CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output prefix: {output_prefix}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Steps to run: {steps_to_run}")
    logger.info("=" * 60)
    
    # Track current input file (changes as pipeline progresses)
    current_input = input_file
    
    # Step 1: Extract
    if "extract" in steps_to_run:
        logger.info("=" * 20 + " STEP 1: EXTRACT " + "=" * 20)
        extract.extract_information(
            input_file=current_input,
            output_file=extracted_file,
            model_name=model_name,
        )
        current_input = extracted_file
    elif "classify" in steps_to_run or "sample" in steps_to_run or "rewrite" in steps_to_run or "finalize" in steps_to_run:
        # If skipping extract but running later steps, assume extracted file exists
        if os.path.exists(extracted_file):
            current_input = extracted_file
    
    # Step 2: Classify
    if "classify" in steps_to_run:
        logger.info("=" * 20 + " STEP 2: CLASSIFY " + "=" * 20)
        classify.classify_all(
            input_file=current_input,
            output_file=classified_file,
            model_name=model_name,
        )
        current_input = classified_file
    elif "sample" in steps_to_run or "rewrite" in steps_to_run or "finalize" in steps_to_run:
        if os.path.exists(classified_file):
            current_input = classified_file
    
    # Step 3: Sample
    if "sample" in steps_to_run:
        logger.info("=" * 20 + " STEP 3: SAMPLE " + "=" * 20)
        sample.sample_all(
            input_file=current_input,
            output_file=sampled_file,
            n_total=n_samples,
            seed=sample_seed,
        )
        current_input = sampled_file
    elif "rewrite" in steps_to_run or "finalize" in steps_to_run:
        if os.path.exists(sampled_file):
            current_input = sampled_file
    
    # Step 4: Rewrite
    if "rewrite" in steps_to_run:
        logger.info("=" * 20 + " STEP 4: REWRITE " + "=" * 20)
        rewrite.llm_rewrite_all(
            input_file=current_input,
            output_file=rewritten_file,
            model_name=model_name,
            max_rounds=rewrite_max_rounds,
        )
        current_input = rewritten_file
    elif "finalize" in steps_to_run:
        if os.path.exists(rewritten_file):
            current_input = rewritten_file
    
    # Step 5: Finalize
    if "finalize" in steps_to_run:
        logger.info("=" * 20 + " STEP 5: FINALIZE " + "=" * 20)
        finalize.create_benchmark(
            input_file=current_input,
            output_file=final_file,
            seed=finalize_seed,
        )
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final benchmark saved to: {final_file}")


# Usage examples:
# Run all steps with config file:
#   python run_data_pipeline.py --config_file=configs/pipeline_config.yaml
#
# Run all steps with command line args:
#   python run_data_pipeline.py --input_file=data/cardiology_usmle_questions.json --output_prefix=cardiology_v1
#
# Run specific steps only:
#   python run_data_pipeline.py --config_file=configs/pipeline_config.yaml --steps=rewrite,finalize

if __name__ == "__main__":
    fire.Fire(run_pipeline)
