##############################################################
# Compile benchmark results across models into markdown tables.
##############################################################

import os
import re
import json
import fire
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/raw_results"
BUCKETS = ["answerable", "hard_but_fair", "boundary_tests"]


def find_latest_results(results_dir: str = RESULTS_DIR) -> Dict[str, str]:
    """Find the latest result file for each model based on filename timestamp."""
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results directory {results_dir} does not exist.")

    model_files: Dict[str, List[tuple]] = defaultdict(list)

    for fname in os.listdir(results_dir):
        if not fname.startswith("benchmark_raw_results_") or not fname.endswith(".json"):
            continue

        # Extract model name and optional timestamp
        # Format: benchmark_raw_results_{model_name}_{timestamp}.json
        stem = fname[len("benchmark_raw_results_"):-len(".json")]

        # Require a trailing numeric timestamp
        ts_match = re.match(r'^(.+?)_(\d{10,})$', stem)
        if not ts_match:
            logger.info(f"Skipping {fname} (no timestamp in filename)")
            continue

        model_key = ts_match.group(1)
        timestamp = int(ts_match.group(2))
        model_files[model_key].append((timestamp, os.path.join(results_dir, fname)))

    # Pick the latest file per model
    latest = {}
    for model_key, entries in model_files.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        latest[model_key] = entries[0][1]
        logger.info(f"Model '{model_key}': using {os.path.basename(entries[0][1])}")

    return latest


def load_results(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r') as f:
        return json.load(f)


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "-"
    return f"{num / denom * 100:.1f}%"


def _conf_range(scores: List[int]) -> str:
    if not scores:
        return "-"
    return f"{min(scores)}-{max(scores)} (avg {sum(scores)/len(scores):.0f})"


def compute_model_report(results: List[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
    """Compute all metrics for a single model's results."""

    rewritten = [r for r in results if r.get("bucket") != "original"]
    original = [r for r in results if r.get("bucket") == "original"]

    # --- Detect whether newer fields exist ---
    has_confidence = any(r.get("confidence") is not None for r in results)
    has_invalid = any("invalid_format" in r for r in results)

    # --- Per-bucket stats (all samples) ---
    bucket_stats = {}
    for bucket in BUCKETS:
        items = [r for r in rewritten if r.get("bucket") == bucket]
        correct = [r for r in items if r.get("is_correct")]
        incorrect = [r for r in items if not r.get("is_correct")]

        stat = {
            "total": len(items),
            "correct": len(correct),
            "accuracy": len(correct) / len(items) if items else 0,
        }

        if has_invalid:
            stat["invalid"] = sum(1 for r in items if r.get("invalid_format", False))

        if has_confidence:
            correct_confs = [r["confidence"] for r in correct if r.get("confidence") is not None]
            incorrect_confs = [r["confidence"] for r in incorrect if r.get("confidence") is not None]
            stat["correct_conf"] = correct_confs
            stat["incorrect_conf"] = incorrect_confs

        bucket_stats[bucket] = stat

    # --- Excluding invalid samples ---
    valid_results = rewritten
    if has_invalid:
        valid_results = [r for r in rewritten if not r.get("invalid_format", False)]

    valid_bucket_stats = {}
    for bucket in BUCKETS:
        items = [r for r in valid_results if r.get("bucket") == bucket]
        correct = [r for r in items if r.get("is_correct")]
        valid_bucket_stats[bucket] = {
            "total": len(items),
            "correct": len(correct),
            "accuracy": len(correct) / len(items) if items else 0,
        }

    # --- Original question stats ---
    original_stats = None
    if original:
        orig_correct = [r for r in original if r.get("is_correct")]
        original_stats = {
            "total": len(original),
            "correct": len(orig_correct),
            "accuracy": len(orig_correct) / len(original) if original else 0,
        }
        if has_confidence:
            orig_confs = [r["confidence"] for r in original if r.get("confidence") is not None]
            original_stats["conf"] = orig_confs

    # --- Funnel: of questions the model got right on original, how many right per bucket ---
    funnel = {}
    if original:
        # Build set of question_nos the model answered correctly on original
        correct_q_nos = set()
        for r in original:
            if r.get("is_correct"):
                correct_q_nos.add(r["metadata"].get("question_no"))

        for bucket in BUCKETS:
            items = [r for r in rewritten if r.get("bucket") == bucket]
            # Filter to only questions the model got right on original
            items_from_correct = [r for r in items if r["metadata"].get("question_no") in correct_q_nos]
            correct_in_bucket = [r for r in items_from_correct if r.get("is_correct")]

            funnel_entry = {
                "total": len(items_from_correct),
                "correct": len(correct_in_bucket),
                "accuracy": len(correct_in_bucket) / len(items_from_correct) if items_from_correct else 0,
            }

            if has_confidence:
                corr_confs = [r["confidence"] for r in correct_in_bucket if r.get("confidence") is not None]
                incorr = [r for r in items_from_correct if not r.get("is_correct")]
                incorr_confs = [r["confidence"] for r in incorr if r.get("confidence") is not None]
                funnel_entry["correct_conf"] = corr_confs
                funnel_entry["incorrect_conf"] = incorr_confs

            funnel[bucket] = funnel_entry

    return {
        "model_name": model_name,
        "total_samples": len(results),
        "rewritten_total": len(rewritten),
        "original_stats": original_stats,
        "bucket_stats": bucket_stats,
        "valid_bucket_stats": valid_bucket_stats,
        "has_invalid": has_invalid,
        "has_confidence": has_confidence,
        "invalid_total": sum(1 for r in rewritten if r.get("invalid_format", False)) if has_invalid else 0,
        "funnel": funnel,
    }


def build_markdown(reports: List[Dict[str, Any]]) -> str:
    lines = []
    models = [r["model_name"] for r in reports]

    # =========================================================================
    # Section 1: Overall accuracy (all samples)
    # =========================================================================
    lines.append("# Benchmark Results Comparison\n")
    lines.append("## 1. Overall Accuracy (All Samples)\n")

    header = "| Bucket | " + " | ".join(models) + " |"
    sep = "|---|" + "|".join(["---"] * len(models)) + "|"
    lines.append(header)
    lines.append(sep)

    for bucket in BUCKETS:
        row = f"| {bucket} |"
        for report in reports:
            s = report["bucket_stats"].get(bucket, {})
            row += f" {_pct(s.get('correct', 0), s.get('total', 0))} ({s.get('correct', 0)}/{s.get('total', 0)}) |"
        lines.append(row)

    # Overall rewritten row
    row = "| **All rewritten** |"
    for report in reports:
        total = report["rewritten_total"]
        correct = sum(s["correct"] for s in report["bucket_stats"].values())
        row += f" **{_pct(correct, total)}** ({correct}/{total}) |"
    lines.append(row)

    # Original row
    any_original = any(r["original_stats"] is not None for r in reports)
    if any_original:
        row = "| **Original** |"
        for report in reports:
            os_ = report["original_stats"]
            if os_:
                row += f" **{_pct(os_['correct'], os_['total'])}** ({os_['correct']}/{os_['total']}) |"
            else:
                row += " - |"
        lines.append(row)

    # Invalid format row
    any_invalid = any(r["has_invalid"] for r in reports)
    if any_invalid:
        row = "| Invalid format |"
        for report in reports:
            if report["has_invalid"]:
                row += f" {report['invalid_total']}/{report['rewritten_total']} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")

    # =========================================================================
    # Section 2: Accuracy excluding invalid format
    # =========================================================================
    if any_invalid:
        lines.append("## 2. Accuracy (Excluding Invalid Format)\n")
        header = "| Bucket | " + " | ".join(models) + " |"
        lines.append(header)
        lines.append(sep)

        for bucket in BUCKETS:
            row = f"| {bucket} |"
            for report in reports:
                s = report["valid_bucket_stats"].get(bucket, {})
                row += f" {_pct(s.get('correct', 0), s.get('total', 0))} ({s.get('correct', 0)}/{s.get('total', 0)}) |"
            lines.append(row)

        row = "| **All valid** |"
        for report in reports:
            total = sum(s["total"] for s in report["valid_bucket_stats"].values())
            correct = sum(s["correct"] for s in report["valid_bucket_stats"].values())
            row += f" **{_pct(correct, total)}** ({correct}/{total}) |"
        lines.append(row)
        lines.append("")

    # =========================================================================
    # Section 3: Funnel — accuracy per bucket for originally-correct questions
    # =========================================================================
    any_funnel = any(r["funnel"] for r in reports)
    if any_funnel:
        section_num = 3 if any_invalid else 2
        lines.append(f"## {section_num}. Accuracy Funnel (Questions Answered Correctly on Original)\n")
        lines.append("For each bucket, only includes questions where the model answered the **original USMLE question** correctly.\n")

        header = "| Bucket | " + " | ".join(models) + " |"
        lines.append(header)
        lines.append(sep)

        # Original baseline row
        row = "| Original (baseline) |"
        for report in reports:
            os_ = report["original_stats"]
            if os_:
                row += f" {_pct(os_['correct'], os_['total'])} ({os_['correct']}/{os_['total']}) |"
            else:
                row += " - |"
        lines.append(row)

        for bucket in BUCKETS:
            row = f"| → {bucket} |"
            for report in reports:
                f = report["funnel"].get(bucket, {})
                row += f" {_pct(f.get('correct', 0), f.get('total', 0))} ({f.get('correct', 0)}/{f.get('total', 0)}) |"
            lines.append(row)

        lines.append("")

    # =========================================================================
    # Section 4: Confidence score ranges per bucket
    # =========================================================================
    any_conf = any(r["has_confidence"] for r in reports)
    if any_conf:
        section_num = (4 if any_invalid else 3) if any_funnel else (3 if any_invalid else 2)
        lines.append(f"## {section_num}. Confidence Scores by Bucket\n")
        lines.append("Range and average of confidence scores for correct vs incorrect responses.\n")

        # All samples table
        lines.append("### All Rewritten Samples\n")
        header = "| Bucket | Outcome | " + " | ".join(models) + " |"
        sep2 = "|---|---|" + "|".join(["---"] * len(models)) + "|"
        lines.append(header)
        lines.append(sep2)

        for bucket in BUCKETS:
            for outcome, key in [("Correct", "correct_conf"), ("Incorrect", "incorrect_conf")]:
                row = f"| {bucket} | {outcome} |"
                for report in reports:
                    s = report["bucket_stats"].get(bucket, {})
                    confs = s.get(key, [])
                    row += f" {_conf_range(confs)} |"
                lines.append(row)

        # Original confidence
        if any_original and any_conf:
            row = "| original | All |"
            for report in reports:
                os_ = report["original_stats"]
                if os_ and "conf" in os_:
                    row += f" {_conf_range(os_['conf'])} |"
                else:
                    row += " - |"
            lines.append(row)

        lines.append("")

        # Funnel confidence table
        if any_funnel:
            lines.append("### Funnel Samples (Originally-Correct Questions Only)\n")
            lines.append(header)
            lines.append(sep2)

            for bucket in BUCKETS:
                for outcome, key in [("Correct", "correct_conf"), ("Incorrect", "incorrect_conf")]:
                    row = f"| {bucket} | {outcome} |"
                    for report in reports:
                        f = report["funnel"].get(bucket, {})
                        confs = f.get(key, [])
                        row += f" {_conf_range(confs)} |"
                    lines.append(row)

            lines.append("")

    return "\n".join(lines)


def compile_results(
    results_dir: str = RESULTS_DIR,
    output_file: str = "results/results_comparison.md",
) -> None:
    """Compile latest results per model into a markdown comparison report."""
    latest_files = find_latest_results(results_dir)

    if not latest_files:
        logger.error("No result files found.")
        return

    reports = []
    for model_key in sorted(latest_files.keys()):
        filepath = latest_files[model_key]
        logger.info(f"Loading {filepath}")
        data = load_results(filepath)
        report = compute_model_report(data, model_key)
        reports.append(report)

    md = build_markdown(reports)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(md)
    logger.info(f"Saved comparison report to {output_file}")
    print(md)


# python compile_results.py
# python compile_results.py --results_dir=results/raw_results --output_file=results/comparison.md
if __name__ == "__main__":
    fire.Fire(compile_results)
