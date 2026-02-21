# Reliable Health LLMs

A research framework for evaluating LLM reliability on clinical question-answering tasks. This project generates synthetic patient prompts from USMLE-style questions and benchmarks LLM performance across different difficulty levels.

## Overview

This framework:

1. **Extracts** clinical information from USMLE-style medical questions with discriminative weighting
2. **Classifies** extracted conditions into clinical categories
3. **Samples** contrast sets with varying difficulty levels (with must-include guards)
4. **Validates** sampled sets for clinical completeness via LLM check
5. **Rewrites** clinical facts into realistic patient-voice prompts
6. **Evaluates** LLM accuracy, confidence, and reasoning on the generated benchmark

## Project Structure

```
reliable-health-llms/
├── pipeline/                    # Data generation pipeline modules
│   ├── extract.py              # Extract clinical info with discriminative weighting
│   ├── classify.py             # Classify conditions into buckets
│   ├── sample.py               # Sample contrast sets with must-include guards
│   ├── validate.py             # LLM completeness validation of sampled sets
│   ├── rewrite.py              # Rewrite as realistic patient-voice prompts
│   ├── finalize.py             # Create final benchmark
│   └── shared.py               # Shared utilities and LLM config
├── configs/                     # Configuration files
│   ├── pipeline_config.yaml    # Data pipeline config
│   └── experiment_sample.yaml  # Benchmark evaluation config
├── data/                        # Input/output data files
├── results/                     # Benchmark results
│   ├── raw_results/            # Per-sample results
│   └── aggregated/             # Summary statistics
├── run_data_pipeline.py         # Run data generation pipeline
├── run_benchmark.py             # Run model evaluation
└── benchmark_viz.py             # Visualize results as HTML
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/reliable-health-llms.git
cd reliable-health-llms

# Install dependencies
pip install -r requirements.txt
```

### Required Environment Variables

Set up API keys for the LLM providers you plan to use:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export FIREWORKS_API_KEY="your-fireworks-key"
```

---

## Data Generation Pipeline

The pipeline transforms USMLE-style questions into patient-style prompts across three difficulty buckets:

| Bucket             | Description                                                                         |
| ------------------ | ----------------------------------------------------------------------------------- |
| **Answerable**     | Contains high-confidence, high-weight clinical facts sufficient to answer correctly |
| **Hard but Fair**  | Contains mid-weight facts; requires careful reasoning                               |
| **Boundary Tests** | Critical information dropped; tests appropriate uncertainty                         |

### Running the Full Pipeline

```bash
# Using config file (recommended)
python run_data_pipeline.py --config_file=configs/pipeline_config.yaml

# Using command line arguments
python run_data_pipeline.py \
    --input_file=data/cardiology_usmle_questions.json \
    --output_dir=data \
    --output_prefix=cardiology_v1
```

### Running Individual Steps

You can run specific pipeline steps:

```bash
# Run only extraction and classification
python run_data_pipeline.py --config_file=configs/pipeline_config.yaml --steps=extract,classify

# Resume from rewrite step (uses existing intermediate files)
python run_data_pipeline.py --config_file=configs/pipeline_config.yaml --steps=rewrite,finalize
```

### Pipeline Steps

| Step        | Command                       | Input         | Output              |
| ----------- | ----------------------------- | ------------- | ------------------- |
| 1. Extract  | `python -m pipeline.extract`  | Raw questions | `*_extracted.json`  |
| 2. Classify | `python -m pipeline.classify` | Extracted     | `*_classified.json` |
| 3. Sample   | `python -m pipeline.sample`   | Classified    | `*_sampled.json`    |
| 4. Validate | `python -m pipeline.validate` | Sampled       | `*_validated.json`  |
| 5. Rewrite  | `python -m pipeline.rewrite`  | Validated     | `*_rewritten.json`  |
| 6. Finalize | `python -m pipeline.finalize` | Rewritten     | `*_benchmark.json`  |

### Pipeline Configuration

Example `configs/pipeline_config.yaml`:

---

## Running Evaluation

### Quick Start

```bash
# Using config file
python run_benchmark.py --config_file=configs/experiment_sample.yaml

# Using command line arguments
python run_benchmark.py \
    --benchmark_file=data/cardiology_usmle_v1_benchmark.json \
    --model_name=gpt-oss-120b \
    --temperature=0.6

# Include original USMLE questions alongside patient prompts
python run_benchmark.py \
    --benchmark_file=data/cardiology_usmle_v1_benchmark.json \
    --model_name=gpt-oss-120b \
    --include_original_questions=true
```

### Evaluation Configuration

Example `configs/experiment_sample.yaml`:

### Evaluation Features

- **Confidence scoring**: Each model response includes a calibrated confidence score (0-100) and reasoning
- **Original question evaluation**: Optionally evaluate models on the original USMLE question stems alongside patient prompts (`include_original_questions: true`)
- **Split reporting**: Results are reported separately for rewritten patient prompts and original question stems

### Available Models

| Model Key      | Description                               |
| -------------- | ----------------------------------------- |
| `gpt-5.2`      | OpenAI GPT model                          |
| `claude`       | Anthropic Claude model                    |
| `gpt-oss-120b` | GPT open-source model hosted by Fireworks |

### Output Files

After running evaluation:

```
results/
├── raw_results/
│   └── benchmark_raw_results_{model_name}.json    # Per-sample results with confidence
└── aggregated/
    └── benchmark_summary_{model_name}.json        # Accuracy & confidence metrics (split by rewritten/original)
```

Each raw result includes: `response`, `is_correct`, `confidence` (0-100), `confidence_reasoning`, and `bucket`.

---

## Visualizing Results

Generate an interactive HTML visualization of benchmark results:

```bash
# Visualize raw benchmark data
python benchmark_viz.py data/cardiology_usmle_v1_benchmark.json -o visualization/benchmark_viz.html

# Visualize evaluation results (includes model responses and correctness)
python benchmark_viz.py results/raw_results/benchmark_raw_results_gpt-oss-120b.json \
    -o visualization/benchmark_viz_gpt-oss-120b.html
```

---

## Example Workflow

```bash
# 1. Generate benchmark from USMLE questions
python run_data_pipeline.py --config_file=configs/pipeline_config.yaml

# 2. Run evaluation on multiple models
python run_benchmark.py --config_file=configs/experiment_sample.yaml

# 3. Visualize results
python benchmark_viz.py results/raw_results/benchmark_raw_results_gpt-oss-120b.json \
    -o visualization/gpt-oss-120b_results.html

# 4. Open visualization in browser
open visualization/gpt-oss-120b_results.html
```

---

## License

TBD

## Citation

TBD
