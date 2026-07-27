"""
merge_shards.py — Merge all shard_*.jsonl files from test_llama/ into output.csv or output.json
Usage:
    python merge_shards.py                  # uses hardcoded paths below, writes CSV
    python merge_shards.py --json            # write JSON instead of CSV
    python merge_shards.py --csv --out my.csv
    python merge_shards.py --json --out my.json
    python merge_shards.py --shards /other/dir --out /other/output.csv
"""

import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── Hardcoded paths (match llama-cpp.py) ──────────────────────────────────────
SHARD_DIR   = Path("/home/aza/workspace/textai-reason/test_llama")
OUTPUT_BASE = "/home/aza/workspace/textai-reason/test_llama/output"
VALID_LABELS = {"depression", "non-depression"}


def merge(shard_dir: Path, output_path: str, fmt: str) -> None:
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

    # ── Split valid vs. invalid-label rows ────────────────────────────────────
    valid_mask = df["label"].isin(VALID_LABELS)
    good_df = df[valid_mask]
    bad_df  = df[~valid_mask]

    def _save(frame: pd.DataFrame, path: str) -> None:
        if fmt == "json":
            frame.to_json(path, orient="records", indent=2, force_ascii=False)
        else:
            frame.to_csv(path, index=False, encoding="utf-8")

    _save(good_df, output_path)
    print(f"Saved {len(good_df):,} rows → {output_path}  ({fmt.upper()})")

    if not bad_df.empty:
        out_dir = Path(output_path).parent
        error_path = out_dir / f"error.{fmt}"
        _save(bad_df, str(error_path))
        print(f"Saved {len(bad_df):,} invalid-label rows → {error_path}  ({fmt.upper()})")


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL shards into a single CSV or JSON file")
    parser.add_argument("--shards", default=str(SHARD_DIR),
                        help=f"Directory containing shard_*.jsonl files (default: {SHARD_DIR})")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: output.csv or output.json depending on format)")

    fmt_group = parser.add_mutually_exclusive_group()
    fmt_group.add_argument("--csv", action="store_true", help="Write output as CSV (default)")
    fmt_group.add_argument("--json", action="store_true", help="Write output as JSON")

    args = parser.parse_args()

    fmt = "json" if args.json else "csv"  # csv is the default when neither flag is given
    out_path = args.out or f"{OUTPUT_BASE}.{fmt}"

    merge(Path(args.shards), out_path, fmt)


if __name__ == "__main__":
    main()