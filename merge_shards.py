"""
merge_shards.py — Merge all shard_*.jsonl files from test_llama/ into output.csv
Usage:
    python merge_shards.py                  # uses hardcoded paths below
    python merge_shards.py --out my.csv     # custom output path
    python merge_shards.py --shards /other/dir --out /other/output.csv
"""

import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Hardcoded paths (match llama-cpp.py) ──────────────────────────────────────
SHARD_DIR  = Path("/home/aza/workspace/textai-reason/test_llama")
OUTPUT_CSV = "/home/aza/workspace/textai-reason/test_llama/output.csv"
VALID_LABELS = {"depression", "non-depression"}


def merge(shard_dir: Path, output_csv: str) -> None:
    shards = sorted(shard_dir.glob("shard_*.jsonl"))

    if not shards:
        print(f"[ERROR] No shard_*.jsonl files found in {shard_dir}")
        return

    print(f"Found {len(shards):,} shard files in {shard_dir}")

    records = []
    bad_lines = 0

    for shard in tqdm(shards, desc="Reading shards", unit="shard"):
        try:
            text = shard.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[WARN] Could not read {shard.name}: {e}")
            continue

        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"[WARN] {shard.name}:{lineno} — bad JSON: {e}")

    if not records:
        print("[ERROR] No records found — nothing to write.")
        return

    df = pd.DataFrame(records)

    # ── Basic stats ────────────────────────────────────────────────────────────
    total      = len(df)
    valid_mask = df["label"].isin(VALID_LABELS)
    invalid    = (~valid_mask).sum()

    print(f"\n── Merge summary ────────────────────────────────────────")
    print(f"  Total rows       : {total:,}")
    print(f"  Bad JSON lines   : {bad_lines:,}")
    print(f"  Invalid labels   : {invalid:,}")
    if invalid:
        print(f"  Invalid examples : {df.loc[~valid_mask, 'label'].unique().tolist()}")
    print(f"\n── Label distribution ───────────────────────────────────")
    print(df["label"].value_counts().to_string())
    print(f"────────────────────────────────────────────────────────\n")

    # Ensure column order: text, label, thinking (others appended after)
    cols = ["text", "label", "thinking"]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved {total:,} rows → {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL shards into a single CSV")
    parser.add_argument("--shards", default=str(SHARD_DIR),
                        help=f"Directory containing shard_*.jsonl files (default: {SHARD_DIR})")
    parser.add_argument("--out", default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    merge(Path(args.shards), args.out)


if __name__ == "__main__":
    main()
