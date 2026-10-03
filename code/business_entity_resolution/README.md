# Business Entity Resolution Pipeline

## Team: ML Navigators

## Overview
This repository contains the complete end to end machine learning pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. The pipeline identifies and links business records across three independent data sources using blocking based candidate generation, multi feature similarity scoring, and threshold optimized matching rules.

## Repository Structure

```
code/business_entity_resolution/
├── src/
│   ├── validation_pipeline.ipynb    # Threshold optimization on training data
│   ├── inference_pipeline.ipynb     # Chunked inference on test data
│   ├── ml_navigators.ipynb          # Core pipeline notebook
│   └── test_inference.py            # Standalone inference script
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

## Prerequisites

### Hardware
* Minimum 16 GB RAM (recommended 32 GB+ for full dataset processing)
* The pipeline was developed and tested on AWS SageMaker

### Software
* Python 3.8 or higher
* All dependencies listed in `requirements.txt`

## Installation

```bash
pip install pandas numpy tqdm rapidfuzz polars matplotlib scikit_learn pyarrow
```

Or install from the requirements file:
```bash
pip install -r requirements.txt
```

## Dependencies

### Core Libraries
| Package | Purpose |
| :--- | :--- |
| `pandas` | Tabular data manipulation, TSV file I/O |
| `numpy` | Numerical computation, metric averaging |
| `tqdm` | Progress bar visualization during inference loops |

### Similarity and Matching
| Package | Purpose |
| :--- | :--- |
| `rapidfuzz` | High performance fuzzy string matching (token_set_ratio) |

### Optional Libraries
| Package | Purpose |
| :--- | :--- |
| `polars` | Optional high performance DataFrame backend (graceful fallback if absent) |
| `matplotlib` | EDA visualization (country distributions, text length histograms) |
| `scikit_learn` | Utility functions for evaluation |
| `pyarrow` | High throughput Arrow based CSV engine for accelerated data loading (`engine="pyarrow"`) |

### Standard Library Modules Used
`os`, `re`, `gc`, `time`, `math`, `warnings`, `pathlib`, `collections` (defaultdict, Counter)

## Data Placement

Place the challenge dataset files in the following structure relative to your home directory:

```
~/student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── output/               # Created automatically by the pipeline
```

## How to Reproduce Results

### Phase 1: Threshold Validation

1. Open `src/validation_pipeline.ipynb` in Jupyter or SageMaker.
2. Execute all cells sequentially from top to bottom.
3. The notebook will:
   * Load and normalize training data from all three sources.
   * Parse the ground truth labels and split comma separated match IDs.
   * Build inverted blocking indexes on S2 and S3 training data.
   * Compute pairwise similarity features (exact match, Jaccard token similarity, RapidFuzz fuzzy similarity).
   * Run a grid search over name thresholds [0.92, 0.94, 0.96] and address thresholds [0.84, 0.88, 0.92].
   * Output the optimal threshold combination that maximizes the macro F_0.5 score.
4. Record the optimal thresholds. Our validation achieved approximately F_0.5 = 0.67.

### Phase 2: Full Test Inference

1. Open `src/inference_pipeline.ipynb` in Jupyter or SageMaker.
2. Set the optimized thresholds in the `is_match()` function cell.
3. The pipeline processes the 1,732,544 test S1 entities in **18 chunks** of 100,000 rows each.
4. For each chunk:
   * Set `CURRENT_CHUNK` to the chunk number (0 through 17).
   * Execute all cells.
   * The notebook saves `matching_results_chunk_{i}.tsv` and `candidate_pairs_chunk_{i}.tsv` to the output directory.
5. After all 18 chunks are complete:
   * Restart the kernel to free memory.
   * Run only the final stitching cell, which concatenates all chunk files into:
     * `output/matching_results.tsv` (final submission file)
     * `output/candidate_pairs.tsv` (blocking analysis file)

### Phase 3: Validation of Output

Run the provided validation script from the `student_resource/` directory:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│ 1. DATA PREPROCESSING                              │
│    Normalize → Clean Legal Suffixes → Compact       │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│ 2. BLOCKING / CANDIDATE GENERATION                  │
│    Build 3 Inverted Indexes on S2+S3                │
│    Query with S1 keys → Union of candidates         │
│    ~10 trillion comparisons → ~40M comparisons      │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│ 3. FEATURE ENGINEERING                              │
│    Exact Match + Jaccard Tokens + RapidFuzz Fuzzy   │
│    6 features per candidate pair                    │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│ 4. THRESHOLD BASED CLASSIFICATION                   │
│    4 tier disjunctive rule cascade                  │
│    Optimized via grid search on F_0.5               │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│ 5. OUTPUT                                           │
│    matching_results.tsv + candidate_pairs.tsv       │
└─────────────────────────────────────────────────────┘
```

## Final Output Statistics

| Metric | Value |
| :--- | :--- |
| Total S1 Entities Processed | 1,732,544 |
| Total Singletons (No Match) | 444,097 (25.6%) |
| Total Linked Pairs | 2,580,875 |
| Total Candidates Evaluated | 40,583,712 |
| Average Candidates per Query | 23.4 |
