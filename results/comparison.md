# Benchmark Results Comparison

## 1. Overall Accuracy (All Samples)

| Bucket | claude-opus-4-5-20251101 | gpt-5.2 | gpt-oss-120b |
|---|---|---|---|
| answerable | 85.5% (53/62) | 91.9% (57/62) | 87.1% (54/62) |
| hard_but_fair | 56.0% (56/100) | 64.0% (64/100) | 61.0% (61/100) |
| boundary_tests | 71.4% (25/35) | 71.4% (25/35) | 60.0% (21/35) |
| **All rewritten** | **68.0%** (134/197) | **74.1%** (146/197) | **69.0%** (136/197) |
| **Original** | **86.7%** (13/15) | **100.0%** (15/15) | **93.3%** (14/15) |
| Invalid format | 20/197 | 1/197 | 0/197 |

## 2. Accuracy (Excluding Invalid Format)

| Bucket | claude-opus-4-5-20251101 | gpt-5.2 | gpt-oss-120b |
|---|---|---|---|
| answerable | 94.6% (53/56) | 91.9% (57/62) | 87.1% (54/62) |
| hard_but_fair | 62.9% (56/89) | 64.6% (64/99) | 61.0% (61/100) |
| boundary_tests | 78.1% (25/32) | 71.4% (25/35) | 60.0% (21/35) |
| **All valid** | **75.7%** (134/177) | **74.5%** (146/196) | **69.0%** (136/197) |

## 3. Accuracy Funnel (Questions Answered Correctly on Original)

For each bucket, only includes questions where the model answered the **original USMLE question** correctly.

| Bucket | claude-opus-4-5-20251101 | gpt-5.2 | gpt-oss-120b |
|---|---|---|---|
| Original (baseline) | 86.7% (13/15) | 100.0% (15/15) | 93.3% (14/15) |
| → answerable | 85.2% (46/54) | 91.9% (57/62) | 86.9% (53/61) |
| → hard_but_fair | 55.7% (49/88) | 64.0% (64/100) | 60.6% (57/94) |
| → boundary_tests | 71.0% (22/31) | 71.4% (25/35) | 60.6% (20/33) |

## 4. Confidence Scores by Bucket

Range and average of confidence scores for correct vs incorrect responses.

### All Rewritten Samples

| Bucket | Outcome | claude-opus-4-5-20251101 | gpt-5.2 | gpt-oss-120b |
|---|---|---|---|---|
| answerable | Correct | 72-98 (avg 91) | 67-97 (avg 85) | 68-96 (avg 90) |
| answerable | Incorrect | 72-78 (avg 75) | 64-78 (avg 72) | 70-88 (avg 83) |
| hard_but_fair | Correct | 35-97 (avg 86) | 43-92 (avg 79) | 68-95 (avg 84) |
| hard_but_fair | Incorrect | 25-92 (avg 68) | 46-84 (avg 67) | 45-95 (avg 79) |
| boundary_tests | Correct | 35-95 (avg 81) | 46-92 (avg 78) | 78-95 (avg 85) |
| boundary_tests | Incorrect | 35-92 (avg 59) | 38-85 (avg 64) | 70-90 (avg 79) |
| original | All | 75-95 (avg 93) | 68-92 (avg 86) | 73-95 (avg 86) |

### Funnel Samples (Originally-Correct Questions Only)

| Bucket | Outcome | claude-opus-4-5-20251101 | gpt-5.2 | gpt-oss-120b |
|---|---|---|---|---|
| answerable | Correct | 72-98 (avg 90) | 67-97 (avg 85) | 68-96 (avg 90) |
| answerable | Incorrect | 72-78 (avg 75) | 64-78 (avg 72) | 70-88 (avg 83) |
| hard_but_fair | Correct | 35-97 (avg 85) | 43-92 (avg 79) | 68-95 (avg 85) |
| hard_but_fair | Incorrect | 25-92 (avg 68) | 46-84 (avg 67) | 45-95 (avg 79) |
| boundary_tests | Correct | 35-95 (avg 82) | 46-92 (avg 78) | 78-95 (avg 85) |
| boundary_tests | Incorrect | 35-92 (avg 63) | 38-85 (avg 64) | 70-90 (avg 79) |
