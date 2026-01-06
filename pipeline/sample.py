#####################################################
# Sample clinical conditions to create new prompts
#####################################################

from __future__ import annotations

import random
import logging
import os
import json
import fire
from typing import Any, Dict, List, Optional, Tuple
from . import shared

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Item = Dict[str, Any]  # expects keys: condition(str), weight(int/float 0-10), bucket(str), confidence(float 0-1)


CORE_BUCKETS = {"SYMPTOMS", "TIMING_COURSE", "PHYSICAL_EXAM", "LABS_IMAGING"}
CONTEXT_BUCKETS = {"DEMOGRAPHICS", "RISK_FACTORS", "CONTEXT"}
NEG_BUCKET = "NEGATIVE_FINDINGS"


def _score(item: Item) -> float:
    # Primary usefulness score: weight scaled by confidence.
    w = float(item.get("weight", 0.0))
    c = float(item.get("confidence", 0.0))
    return max(0.0, w) * max(0.0, min(1.0, c))


def _jaccard(a: List[Item], b: List[Item]) -> float:
    """Jaccard similarity based on condition text."""
    sa = {x["condition"] for x in a}
    sb = {x["condition"] for x in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _weighted_sample_without_replacement(
    rng: random.Random,
    pool: List[Item],
    k: int,
    weight_fn,
) -> List[Item]:
    """Efraimidis-Spirakis style: assign random keys and take top-k."""
    if k <= 0 or not pool:
        return []
    items = []
    for it in pool:
        w = float(weight_fn(it))
        if w <= 0:
            continue
        # key = U^(1/w); larger is better
        u = rng.random()
        key = u ** (1.0 / w)
        items.append((key, it))
    items.sort(key=lambda t: t[0], reverse=True)
    return [it for _, it in items[:k]]


def _ensure_core_coverage(
    rng: random.Random,
    selected: List[Item],
    candidates: List[Item],
    min_core_buckets: int = 1,
) -> List[Item]:
    """Try to ensure at least `min_core_buckets` distinct core buckets in selected."""
    core_present = {i["bucket"] for i in selected if i.get("bucket") in CORE_BUCKETS}
    if len(core_present) >= min_core_buckets:
        return selected

    # Add one best core item not already selected, if available.
    sel_text = {i["condition"] for i in selected}
    core_pool = [i for i in candidates if i.get("bucket") in CORE_BUCKETS and i["condition"] not in sel_text]
    if core_pool:
        best = max(core_pool, key=_score)
        selected.append(best)
    return selected


def _pick_size(rng: random.Random, avg_size: int = 3, mode: str = "answerable") -> int:
    """
    Average set size around 3. We keep it tight:
      - Answerable: usually 3, sometimes 4
      - Hard: usually 3, sometimes 2
      - Boundary base: usually 3 (drop makes it 2)
    """
    if mode == "answerable":
        return 4 if rng.random() < 0.25 else 3
    if mode == "hard":
        return 2 if rng.random() < 0.30 else 3
    if mode == "boundary":
        return 3
    return max(1, avg_size)


def _sample_answerable(rng: random.Random, items: List[Item]) -> List[Item]:
    clean = [i for i in items if float(i.get("confidence", 0.0)) >= 0.70]
    anchors = [i for i in clean if float(i.get("weight", 0.0)) >= 8.0 and i.get("bucket") in CORE_BUCKETS]
    if not anchors:
        anchors = [i for i in clean if float(i.get("weight", 0.0)) >= 8.0]

    target = _pick_size(rng, mode="answerable")
    selected: List[Item] = []

    # 1 anchor if possible
    if anchors:
        selected += _weighted_sample_without_replacement(rng, anchors, 1, _score)

    # Fill remaining using score, preferring >=3 weight
    sel_text = {i["condition"] for i in selected}
    pool = [i for i in clean if i["condition"] not in sel_text and float(i.get("weight", 0.0)) >= 3.0]
    need = max(0, target - len(selected))

    # Mild bucket-balancing: downweight buckets already present
    bucket_counts = {}
    for it in selected:
        bucket_counts[it.get("bucket")] = bucket_counts.get(it.get("bucket"), 0) + 1

    def balanced_weight(it: Item) -> float:
        base = _score(it)
        b = it.get("bucket")
        penalty = 1.0 / (1.0 + bucket_counts.get(b, 0))
        return base * penalty

    selected += _weighted_sample_without_replacement(rng, pool, need, balanced_weight)

    # Ensure at least 1 core bucket represented (with avg size 3, 2-core is often too strict)
    selected = _ensure_core_coverage(rng, selected, clean, min_core_buckets=1)

    # Optional single negative finding (rare at size ~3)
    if len(selected) < target and rng.random() < 0.10:
        neg_pool = [i for i in clean if i.get("bucket") == NEG_BUCKET and i["condition"] not in {x["condition"] for x in selected}]
        if neg_pool:
            selected += _weighted_sample_without_replacement(rng, neg_pool, 1, _score)

    # If we exceeded target due to ensure_core_coverage, trim lowest-score non-core first
    if len(selected) > target:
        selected_sorted = sorted(selected, key=_score, reverse=True)
        selected = selected_sorted[:target]

    return selected


def _sample_hard_but_fair(rng: random.Random, items: List[Item]) -> List[Item]:
    clean = [i for i in items if float(i.get("confidence", 0.0)) >= 0.60]
    ambiguous = [i for i in items if 0.45 <= float(i.get("confidence", 0.0)) < 0.60]

    target = _pick_size(rng, mode="hard")
    selected: List[Item] = []

    # Usually avoid slam-dunk anchors; if include, make it contextual
    if rng.random() < 0.20:
        contextual_anchors = [i for i in clean if float(i.get("weight", 0.0)) >= 8.0 and i.get("bucket") in CONTEXT_BUCKETS]
        if contextual_anchors:
            selected += _weighted_sample_without_replacement(rng, contextual_anchors, 1, _score)

    # Prefer mid-weight items (3–7), avoid >7 when possible
    sel_text = {i["condition"] for i in selected}
    pool = [
        i for i in clean
        if i["condition"] not in sel_text
        and 3.0 <= float(i.get("weight", 0.0)) <= 7.0
    ]
    need = max(0, target - len(selected))

    # Add novelty: slight boost for underrepresented buckets
    bucket_counts = {}
    for it in selected:
        bucket_counts[it.get("bucket")] = bucket_counts.get(it.get("bucket"), 0) + 1

    def hard_weight(it: Item) -> float:
        base = _score(it)
        b = it.get("bucket")
        novelty = 1.0 / (1.0 + bucket_counts.get(b, 0))
        return base * (0.85 + 0.15 * novelty)

    selected += _weighted_sample_without_replacement(rng, pool, need, hard_weight)

    # Optionally add one ambiguity OR negative (but keep size small)
    if len(selected) < target and rng.random() < 0.35:
        cand = []
        cand += [i for i in ambiguous if float(i.get("weight", 0.0)) >= 3.0]
        cand += [i for i in clean if i.get("bucket") == NEG_BUCKET and float(i.get("weight", 0.0)) >= 3.0]
        cand = [i for i in cand if i["condition"] not in {x["condition"] for x in selected}]
        if cand:
            selected += _weighted_sample_without_replacement(rng, cand, 1, _score)

    selected = _ensure_core_coverage(rng, selected, clean, min_core_buckets=1)

    # Trim if needed
    if len(selected) > target:
        selected = sorted(selected, key=_score, reverse=True)[:target]

    return selected


def _make_variant_drop(base: List[Item]) -> List[Item]:
    """Drop the highest-impact item (prefer CORE)."""
    if len(base) <= 1:
        return []
    def impact(it: Item) -> float:
        mult = 1.2 if it.get("bucket") in CORE_BUCKETS else 1.0
        return _score(it) * mult
    ranked = sorted(base, key=impact, reverse=True)
    # Prefer dropping a core item if present
    core_first = next((i for i in ranked if i.get("bucket") in CORE_BUCKETS), ranked[0])
    return [i for i in base if i["condition"] != core_first["condition"]]


def generate_contrast_sets(
    items: List[Item],
    n_total: int = 50,
    seed: Optional[int] = 0,
    avg_set_size: int = 3,
    jaccard_max: float = 0.80,
) -> Dict[str, Any]:
    """
    Generates:
      - 30% Answerable
      - 50% Hard-but-fair
      - 20% Boundary tests (base + variant_drop)

    Returns dict with:
      {
        "answerable": [ [Item, ...], ... ],
        "hard_but_fair": [ [Item, ...], ... ],
        "boundary_tests": [ { "base": [...], "variant_drop": [...] }, ... ]
      }
    """
    rng = random.Random(seed)

    n_a = round(0.30 * n_total)
    n_h = round(0.50 * n_total)
    n_b = max(0, n_total - n_a - n_h)  # boundary families

    out_answerable: List[List[Item]] = []
    out_hard: List[List[Item]] = []
    out_boundary: List[Dict[str, List[Item]]] = []

    # Dedup stores sets across answerable+hard (boundary bases deduped separately)
    accepted_sets: List[List[Item]] = []

    def accept_set(candidate: List[Item], against: List[List[Item]], threshold: float) -> bool:
        for s in against:
            if _jaccard(candidate, s) > threshold:
                return False
        return True

    # Generate Answerable
    attempts = 0
    while len(out_answerable) < n_a and attempts < n_a * 50:
        attempts += 1
        s = _sample_answerable(rng, items)
        if not s:
            continue
        if accept_set(s, accepted_sets, jaccard_max):
            out_answerable.append(s)
            accepted_sets.append(s)

    # Generate Hard-but-fair
    attempts = 0
    while len(out_hard) < n_h and attempts < n_h * 60:
        attempts += 1
        s = _sample_hard_but_fair(rng, items)
        if not s:
            continue
        if accept_set(s, accepted_sets, jaccard_max):
            out_hard.append(s)
            accepted_sets.append(s)

    # Generate Boundary families (base + drop). Bases can overlap more with global sets,
    # but keep them somewhat diverse from other bases.
    boundary_bases: List[List[Item]] = []
    attempts = 0
    while len(out_boundary) < n_b and attempts < n_b * 80:
        attempts += 1

        # Choose base mode (slight preference to hard to make boundaries interesting)
        base = _sample_hard_but_fair(rng, items) if rng.random() < 0.60 else _sample_answerable(rng, items)
        if len(base) < 2:
            continue

        # Keep boundary bases somewhat unique among themselves
        if not accept_set(base, boundary_bases, threshold=0.85):
            continue

        variant_drop = _make_variant_drop(base)
        if len(variant_drop) < 1:
            continue

        out_boundary.append({"base": base, "variant_drop": variant_drop})
        boundary_bases.append(base)

    return {
        "answerable": out_answerable,
        "hard_but_fair": out_hard,
        "boundary_tests": out_boundary,
        "meta": {
            "n_total_requested": n_total,
            "counts": {
                "answerable": len(out_answerable),
                "hard_but_fair": len(out_hard),
                "boundary_tests": len(out_boundary),
            },
            "seed": seed,
            "avg_set_size_target": avg_set_size,
            "jaccard_max": jaccard_max,
        },
    }

def sample_all(
    input_file: str,
    output_file: str,
    n_total: int = 12,
    seed: Optional[int] = 42,
    avg_set_size: int = 3,
    jaccard_max: float = 0.80,
) -> None:
    """Sample contrast sets from input items and save to output file."""
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    
    with open(input_file, 'r') as f:
        items = json.load(f)
    
    logger.info(f"Loaded {len(items)} items from {input_file}")
    
    results = []

    for idx, item in enumerate(items):
        result = generate_contrast_sets(
            item['llm_extracted_info'],
            n_total=n_total,
            seed=seed,
            avg_set_size=avg_set_size,
            jaccard_max=jaccard_max,
        )
        item['sampled_sets'] = result
        results.append(item)
        if (idx + 1) % 5 == 0:
            logger.info(f"Processed {idx + 1}/{len(items)} items.")
    
    # Save results
    shared.save_json(results, output_file)
    logger.info(f"Saved sampled contrast sets to {output_file}")

# python -m pipeline.sample --input_file=data/cardiology_usmle_classified_v1.json -output_file=data/cardiology_usmle_sampled_v1.json
if __name__ == "__main__":
    fire.Fire(sample_all)