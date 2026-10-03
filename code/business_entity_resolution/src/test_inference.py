import os
import re
import gc
import time
import math
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import polars as pl
    HAS_POLARS = True
except Exception:
    HAS_POLARS = False

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

print(" [ML Navigators] Environment ready")
print("Polars:", HAS_POLARS)
print("RapidFuzz:", HAS_RAPIDFUZZ)

import os
from pathlib import Path

DATA_ROOT = Path(r"D:\aws_ml_challenge\6ab10eb3b23ba_student_resource\student_resource\dataset")

print("\nSelected DATA_ROOT:", DATA_ROOT)

all_files = sorted(DATA_ROOT.rglob("*"))
for f in all_files:
    if f.is_file():
        print(" -", f.relative_to(DATA_ROOT))

def find_file(filename):
    hits = list(DATA_ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename}")
    return hits[0]

FILES = {
    "train_s1": find_file("train_source1.tsv"),
    "train_s2": find_file("train_source2.tsv"),
    "train_s3": find_file("train_source3.tsv"),
    "ground_truth": find_file("train_ground_truth.tsv"),
    "test_s1": find_file("test_source1.tsv"),
    "test_s2": find_file("test_source2.tsv"),
    "test_s3": find_file("test_source3.tsv"),
}

for k, v in FILES.items():
    print(f" [ML Navigators] {k:14s} -> {v}")

for k, path in FILES.items():
    size_mb = path.stat().st_size / (1024**2)
    print(f" [ML Navigators] {k:14s} {size_mb:10.1f} MB")

def preview_tsv(path, n=5):
    return pd.read_csv(path, sep="\t", nrows=n)

print(" [ML Navigators] \nTRAIN SOURCE 1 PREVIEW")
print(preview_tsv(FILES["train_s1"]))

print(" [ML Navigators] \nGROUND TRUTH PREVIEW")
print(preview_tsv(FILES["ground_truth"]))

CORE_COLS = ["entity_id", "business_name", "business_address", "country"]

for name in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
    df = pd.read_csv(FILES[name], sep="\t", nrows=3)
    print(f" [ML Navigators] \n{name}")
    print(df.dtypes)
    print("columns:", list(df.columns))

SAMPLE_N = 50_000
RANDOM_STATE = 42

def sample_tsv(path, n=SAMPLE_N, seed=RANDOM_STATE):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=CORE_COLS,
        nrows=n
    )

eda_s1 = sample_tsv(FILES["train_s1"])
eda_s2 = sample_tsv(FILES["train_s2"])
eda_s3 = sample_tsv(FILES["train_s3"])

for label, df in [("S1", eda_s1), ("S2", eda_s2), ("S3", eda_s3)]:
    print(f" [ML Navigators] \n===== {label} =====")
    print("rows:", len(df))
    print(" [ML Navigators] missing:")
    print(df.isna().mean().mul(100).round(2).to_frame("missing_%"))
    print(" [ML Navigators] countries:")
    print(df["country"].value_counts(dropna=False).head(15).to_frame("count"))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(2, 2, 1)
eda_s1["country"].value_counts().head(10).plot(kind="bar", ax=ax1)
ax1.set_title("S1 Sample — Country Distribution")
ax1.set_ylabel("Records")
ax1.tick_params(axis="x", rotation=45)

ax2 = fig.add_subplot(2, 2, 2)
eda_s1["business_name"].fillna("").str.len().clip(upper=100).plot(kind="hist", bins=40, ax=ax2)
ax2.set_title("Business Name Length")
ax2.set_xlabel("Characters")

ax3 = fig.add_subplot(2, 2, 3)
eda_s1["business_address"].fillna("").str.len().clip(upper=250).plot(kind="hist", bins=40, ax=ax3)
ax3.set_title("Business Address Length")
ax3.set_xlabel("Characters")

ax4 = fig.add_subplot(2, 2, 4)
miss = eda_s1.isna().mean().mul(100)
miss.plot(kind="bar", ax=ax4)
ax4.set_title("Missingness — S1 Sample")
ax4.set_ylabel("%")

plt.tight_layout()
plt.show()

LEGAL_SUFFIXES = {
    "incorporated", "inc", "corporation", "corp",
    "limited", "ltd", "llc", "llp",
    "private", "pvt", "company", "co",
    "limitedliabilitycompany"
}

def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_name(x):
    s = normalize_text(x)
    tokens = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)

def normalize_address(x):
    return normalize_text(x)

def compact(s):
    return re.sub(r"\s+", "", s)

def token_signature(s):
    return " ".join(sorted(set(s.split())))

for df in [eda_s1]:
    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_address"] = df["business_address"].map(normalize_address)
    df["compact_name"] = df["norm_name"].map(compact)
    df["compact_address"] = df["norm_address"].map(compact)

print(
    eda_s1[
        ["business_name", "norm_name", "business_address", "norm_address"]
    ].head(15)
)

def prepare_small(df):
    out = df[CORE_COLS].copy()
    out = out.fillna("")
    out["norm_name"] = out["business_name"].map(normalize_name)
    out["norm_address"] = out["business_address"].map(normalize_address)
    out["compact_name"] = out["norm_name"].map(compact)
    out["compact_address"] = out["norm_address"].map(compact)
    out["name_addr_key"] = (
        out["country"].astype(str) + "|" +
        out["compact_name"] + "|" +
        out["compact_address"]
    )
    return out

small_s1 = prepare_small(pd.read_csv(FILES["train_s1"], sep="\t", nrows=20_000, dtype=str))
small_s2 = prepare_small(pd.read_csv(FILES["train_s2"], sep="\t", nrows=50_000, dtype=str))
small_s3 = prepare_small(pd.read_csv(FILES["train_s3"], sep="\t", nrows=50_000, dtype=str))

print(small_s1.head())

def build_index(df, column):
    idx = defaultdict(list)
    for row in df.itertuples(index=False):
        key = getattr(row, column)
        if key:
            idx[key].append(row.entity_id)
    return idx

def add_block_keys(df):
    df = df.copy()
    df["country_name_key"] = (
        df["country"].astype(str) + "|" + df["compact_name"]
    )
    df["country_addr_key"] = (
        df["country"].astype(str) + "|" + df["compact_address"]
    )
    df["country_name_addr_key"] = (
        df["country"].astype(str) + "|" +
        df["compact_name"] + "|" + df["compact_address"]
    )
    return df

small_s1 = add_block_keys(small_s1)
small_s2 = add_block_keys(small_s2)
small_s3 = add_block_keys(small_s3)

s2_name_idx = build_index(small_s2, "country_name_key")
s2_addr_idx = build_index(small_s2, "country_addr_key")
s3_name_idx = build_index(small_s3, "country_name_key")
s3_addr_idx = build_index(small_s3, "country_addr_key")

print("S2 name index keys:", len(s2_name_idx))
print("S3 name index keys:", len(s3_name_idx))

def candidates_for_row(row, indexes):
    candidates = set()

    for key_col, idx in [
        ("country_name_key", indexes["name"]),
        ("country_addr_key", indexes["addr"]),
        ("country_name_addr_key", indexes["name_addr"]),
    ]:
        key = row.get(key_col) if isinstance(row, dict) else getattr(row, key_col, None)
        if key:
            candidates.update(idx.get(key, []))

    return candidates

s2_name_addr_idx = build_index(small_s2, "country_name_addr_key")
s3_name_addr_idx = build_index(small_s3, "country_name_addr_key")

INDEXES_S2 = {
    "name": s2_name_idx,
    "addr": s2_addr_idx,
    "name_addr": s2_name_addr_idx,
}
INDEXES_S3 = {
    "name": s3_name_idx,
    "addr": s3_addr_idx,
    "name_addr": s3_name_addr_idx,
}

example = small_s1.iloc[0]
print("Example S1:", example["entity_id"])
print("S2 candidates:", candidates_for_row(example, INDEXES_S2))
print("S3 candidates:", candidates_for_row(example, INDEXES_S3))

def jaccard_tokens(a, b):
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def pair_features(a, b):
    name_exact = int(a["compact_name"] != "" and a["compact_name"] == b["compact_name"])
    addr_exact = int(a["compact_address"] != "" and a["compact_address"] == b["compact_address"])
    name_token = jaccard_tokens(a["norm_name"], b["norm_name"])
    addr_token = jaccard_tokens(a["norm_address"], b["norm_address"])

    if HAS_RAPIDFUZZ:
        name_fuzzy = fuzz.token_set_ratio(a["norm_name"], b["norm_name"]) / 100
        addr_fuzzy = fuzz.token_set_ratio(a["norm_address"], b["norm_address"]) / 100
    else:
        name_fuzzy = float(name_exact)
        addr_fuzzy = float(addr_exact)

    return {
        "name_exact": name_exact,
        "addr_exact": addr_exact,
        "name_token": name_token,
        "addr_token": addr_token,
        "name_fuzzy": name_fuzzy,
        "addr_fuzzy": addr_fuzzy,
    }

def match_score(f):
    return (
        4.0 * f["name_exact"] +
        4.0 * f["addr_exact"] +
        2.0 * f["name_fuzzy"] +
        2.0 * f["addr_fuzzy"] +
        1.0 * f["name_token"] +
        1.0 * f["addr_token"]
    )

def inspect_pair(a, b):
    f = pair_features(a, b)
    result = pd.DataFrame([{
        "S1": a["entity_id"],
        "candidate": b["entity_id"],
        "country": a["country"],
        "S1_name": a["business_name"],
        "candidate_name": b["business_name"],
        "S1_address": a["business_address"],
        "candidate_address": b["business_address"],
        **f,
        "score": match_score(f)
    }])
    print(result.T)

if len(small_s1) and len(small_s2):
    inspect_pair(small_s1.iloc[0], small_s2.iloc[0])

def make_indexes(df):
    return {
        "name": build_index(df, "country_name_key"),
        "addr": build_index(df, "country_addr_key"),
        "name_addr": build_index(df, "country_name_addr_key"),
    }

val_idx_s2 = make_indexes(train_s2_val)
val_idx_s3 = make_indexes(train_s3_val)

s2_by_id = train_s2_val.set_index("entity_id").to_dict("index")
s3_by_id = train_s3_val.set_index("entity_id").to_dict("index")

def is_match(f):
    if f["name_exact"] and f["addr_exact"]:
        return True
    if f["name_exact"] and f["addr_fuzzy"] >= 0.84:
        return True
    if f["addr_exact"] and f["name_fuzzy"] >= 0.92:
        return True
    if f["name_fuzzy"] >= 0.92 and f["addr_fuzzy"] >= 0.84:
        return True
    return False

def predict_for_s1(row, idx2, idx3, by_id2, by_id3):
    candidates = set()
    candidates.update(candidates_for_row(row, idx2))
    candidates.update(candidates_for_row(row, idx3))

    predictions = []

    for cid in candidates:
        if cid.startswith("S2-"):
            c = by_id2.get(cid)
        else:
            c = by_id3.get(cid)

        if c is None:
            continue

        f = pair_features(row, c)
        if is_match(f):
            predictions.append(cid)



import time

def load_core(path, name, limit=None):
    print(f"[{time.strftime('%H:%M:%S')}] Started loading {name} from disk...")
    t0 = time.time()
    kwargs = {
        "sep": "\t",
        "usecols": CORE_COLS,
        "dtype": str,
    }
    if limit is not None:
        kwargs["nrows"] = limit

    df = pd.read_csv(path, **kwargs).fillna("")
    print(f"[{time.strftime('%H:%M:%S')}] {name} loaded ({len(df):,} rows). Now normalizing text (this may take a few minutes)...")

    df = prepare_small(df)
    print(f"[{time.strftime('%H:%M:%S')}] {name} text normalization complete. Building block keys...")

    df = add_block_keys(df)
    print(f"[{time.strftime('%H:%M:%S')}] {name} completely finished in {time.time()-t0:.1f} seconds!\n")
    return df

FAST_DEV_MODE = True

if FAST_DEV_MODE:
    test_s1 = load_core(FILES["test_s1"], "Test S1", DEV_S1_LIMIT)
    test_s2 = load_core(FILES["test_s2"], "Test S2", DEV_S2_LIMIT)
    test_s3 = load_core(FILES["test_s3"], "Test S3", DEV_S3_LIMIT)
else:
    test_s1 = load_core(FILES["test_s1"], "Test S1")
    test_s2 = load_core(FILES["test_s2"], "Test S2")
    test_s3 = load_core(FILES["test_s3"], "Test S3")

print("Test S1:", len(test_s1))
print("Test S2:", len(test_s2))
print("Test S3:", len(test_s3))

# --- ML NAVIGATORS CHUNKING LOGIC ---
CHUNK_SIZE = 50
NUM_CHUNKS = int(len(test_s1) / CHUNK_SIZE) + 1
CURRENT_CHUNK = 0  # Set from 0 to (NUM_CHUNKS - 1) to run different chunks

start_idx = CURRENT_CHUNK * CHUNK_SIZE
end_idx = min((CURRENT_CHUNK + 1) * CHUNK_SIZE, len(test_s1))
test_s1 = test_s1.iloc[start_idx:end_idx] if hasattr(test_s1, 'iloc') else test_s1[start_idx:end_idx]
print(f"Processing chunk {CURRENT_CHUNK+1} of {NUM_CHUNKS} (Rows {start_idx} to {end_idx})")


test_idx_s2 = make_indexes(test_s2)
test_idx_s3 = make_indexes(test_s3)

test_s2_by_id = test_s2.set_index("entity_id").to_dict("index")
test_s3_by_id = test_s3.set_index("entity_id").to_dict("index")

print("S2 name blocks:", len(test_idx_s2["name"]))
print("S3 name blocks:", len(test_idx_s3["name"]))

def candidate_ids_for_test_row(row):
    out = set()
    out.update(candidates_for_row(row, test_idx_s2))
    out.update(candidates_for_row(row, test_idx_s3))
    return sorted(out)

candidate_rows = []
t0 = time.time()

for row in test_s1.itertuples(index=False):
    cands = candidate_ids_for_test_row(row)
    candidate_rows.append({
        "source1_entity_id": row.entity_id,
        "candidate_entity_ids": ",".join(cands)
    })

candidate_df = pd.DataFrame(candidate_rows)

print(" [ML Navigators] Candidate generation time:", round(time.time() - t0, 2), "seconds")
print("S1 entities:", len(candidate_df))
print("Average candidates:", candidate_df["candidate_entity_ids"].map(
    lambda x: len(x.split(",")) if x else 0
).mean())
print(candidate_df.head(10))

def final_predictions_for_row(row):
    cands = candidate_ids_for_test_row(row)
    out = []

    row_dict = row._asdict() if hasattr(row, "_asdict") else row

    for cid in cands:
        if cid.startswith("S2-"):
            c = test_s2_by_id.get(cid)
        elif cid.startswith("S3-"):
            c = test_s3_by_id.get(cid)
        else:
            c = None

        if c is None:
            continue

        c_dict = c._asdict() if hasattr(c, "_asdict") else c

        f = pair_features(row_dict, c_dict)
        if is_match(f):
            out.append(cid)

    return sorted(set(out))

match_rows = []
t0 = time.time()

for row in test_s1.itertuples(index=False):
    matches = final_predictions_for_row(row)
    match_rows.append({
        "source1_entity_id": row.entity_id,
        "matched_entity_ids": ",".join(matches)
    })

matching_df = pd.DataFrame(match_rows)

print(" [ML Navigators] Matching time:", round(time.time() - t0, 2), "seconds")
print(matching_df.head(10))

def validate_outputs(s1_df, matching_df, candidate_df, s2_df, s3_df):
    problems = []

    expected = set(s1_df["entity_id"])
    predicted = set(matching_df["source1_entity_id"])

    if expected != predicted:
        problems.append(
            f"S1 coverage mismatch: expected {len(expected)}, got {len(predicted)}"
        )

    if matching_df["source1_entity_id"].duplicated().any():
        problems.append("Duplicate source1_entity_id in matching output.")

    valid_s2s3 = set(s2_df["entity_id"]) | set(s3_df["entity_id"])

    candidate_map = dict(zip(
        candidate_df["source1_entity_id"],
        candidate_df["candidate_entity_ids"].map(
            lambda x: set(x.split(",")) if x else set()
        )
    ))

    for row in matching_df.itertuples(index=False):
        mids = set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()

        if len(mids) != len(row.matched_entity_ids.split(",")) if row.matched_entity_ids else False:
            problems.append(f"Duplicate IDs for {row.source1_entity_id}")

        bad_ids = mids - valid_s2s3
        if bad_ids:
            problems.append(f"Invalid IDs for {row.source1_entity_id}: {list(bad_ids)[:3]}")

        if not mids.issubset(candidate_map.get(row.source1_entity_id, set())):
            problems.append(f"Match outside candidate set: {row.source1_entity_id}")

    if problems:
        print(" [ML Navigators] ❌ Validation found issues:")
        for p in problems[:25]:
            print("-", p)
        return False

    print(" [ML Navigators] ✅ Notebook validation passed.")
    return True

validate_outputs(
    test_s1,
    matching_df,
    candidate_df,
    test_s2,
    test_s3
)

OUTPUT_DIR = Path(r"D:\aws_ml_challenge\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

matching_path = OUTPUT_DIR / f"matching_results_chunk_{CURRENT_CHUNK}.tsv"
candidate_path = OUTPUT_DIR / f"candidate_pairs_chunk_{CURRENT_CHUNK}.tsv"

matching_df.to_csv(matching_path, sep="\t", index=False)
candidate_df.to_csv(candidate_path, sep="\t", index=False)

print(" [ML Navigators] Saved:")
print(matching_path)
print(candidate_path)

print(" [ML Navigators] \nFile sizes:")
print(" [ML Navigators] matching_results.tsv:", round(matching_path.stat().st_size / 1024**2, 2), "MB")
print(" [ML Navigators] candidate_pairs.tsv:", round(candidate_path.stat().st_size / 1024**2, 2), "MB")

print("=" * 70)
print(" [ML Navigators] AMAZON ML CHALLENGE 2026 — FINAL CHECK")
print("=" * 70)
print("matching_results.tsv :", matching_path.exists())
print("candidate_pairs.tsv :", candidate_path.exists())
print("S1 rows              :", len(matching_df))
print("Predicted links      :", int(match_counts.sum()))
print("Predicted singletons :", int((match_counts == 0).sum()))
print("Avg candidates       :", round(candidate_counts.mean(), 3))
print("=" * 70)

print(f" [ML Navigators] Output directory: {OUTPUT_DIR}")



# ============================================================
# 30. STITCH CHUNKS TOGETHER (Run after all chunks are done)
# ============================================================
"""
import pandas as pd
from pathlib import Path
import os

OUTPUT_DIR = Path(r"D:\aws_ml_challenge\output")

match_chunks = []
for i in range(NUM_CHUNKS):
    try:
        df = pd.read_csv(OUTPUT_DIR / f"matching_results_chunk_{i}.tsv", sep="\t", dtype=str)
        match_chunks.append(df)
    except FileNotFoundError:
        print(f"Warning: matching_results_chunk_{i}.tsv not found.")
if match_chunks:
    final_matches = pd.concat(match_chunks, ignore_index=True)
    final_matches.to_csv(OUTPUT_DIR / "matching_results.tsv", sep="\t", index=False)
    print(f"Stitched {len(match_chunks)} chunks for matching_results.tsv")

cand_chunks = []
for i in range(NUM_CHUNKS):
    try:
        df = pd.read_csv(OUTPUT_DIR / f"candidate_pairs_chunk_{i}.tsv", sep="\t", dtype=str)
        cand_chunks.append(df)
    except FileNotFoundError:
        print(f"Warning: candidate_pairs_chunk_{i}.tsv not found.")
if cand_chunks:
    final_cands = pd.concat(cand_chunks, ignore_index=True)
    final_cands.to_csv(OUTPUT_DIR / "candidate_pairs.tsv", sep="\t", index=False)
    print(f"Stitched {len(cand_chunks)} chunks for candidate_pairs.tsv")
"""


