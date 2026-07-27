import os
import asyncio
import json
import re
import pandas as pd
import httpx
from tqdm import tqdm
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
LLAMA_BASE_URL  = os.environ.get("LLAMA_BASE_URL",  "http://localhost:8099")
INPUT_CSV       = "/home/aza/workspace/textai-reason/Suicide_Detection.csv"
OUTPUT_DIR      = Path("/home/aza/workspace/textai-reason/test_llama")
OUTPUT_CSV      = "/home/aza/workspace/textai-reason/test_llama/output.csv"

# How many requests to fire concurrently — tune to match --parallel on server
CONCURRENCY     = int(os.environ.get("CONCURRENCY",  "4"))
MAX_TOKENS      = int(os.environ.get("MAX_TOKENS",   "8192"))
TEMPERATURE     = float(os.environ.get("TEMPERATURE","0.7"))
MAX_RETRIES     = 3                              # retries when label is invalid
VALID_LABELS    = {"depression", "non-depression"}

CHAT_URL        = f"{LLAMA_BASE_URL}/v1/chat/completions"

SYSTEM_PROMPT = """You are a clinical text classifier. Classify input text as exactly one of two labels: "depression" or "non-depression".

Classify as "depression" if the text shows:
- Suicidal ideation, self-harm urges, or wishes to die
- Hopelessness, worthlessness, or pervasive emptiness
- Anhedonia — loss of interest in previously enjoyed activities
- Inability to function, extreme fatigue, or emotional numbness
- Expressions of being trapped, a burden, or without a future

Classify as "non-depression" if the text shows:
- Neutral, factual, or informational content
- Situational stress without persistent clinical markers
- Positive, stable, or curious emotional states

STRICT OUTPUT RULE: Output exactly one of these two strings and nothing else:
depression
non-depression"""

USER_PROMPT = """<input_text>
{input_text}
</input_text>

Classification:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_PROMPT.format(input_text=text)},
    ]


def parse_output(raw: str) -> tuple[str, str]:
    """Extract (thinking, label) from model output."""
    raw = raw.strip() if raw else ""

    if "<|channel>" in raw and "<channel|>" in raw:
        start      = raw.index("<|channel>") + len("<|channel>")
        end        = raw.index("<channel|>")
        think_body = raw[start:end]
        if think_body.startswith("thought"):
            think_body = think_body[len("thought"):].lstrip("\n")
        thinking = think_body.strip()
        label    = raw[end + len("<channel|>"):].strip()

    elif "<think>" in raw and "</think>" in raw:
        start    = raw.index("<think>") + len("<think>")
        end      = raw.index("</think>")
        thinking = raw[start:end].strip()
        label    = raw[end + len("</think>"):].strip()

    else:
        thinking = ""
        label    = raw

    # Normalise label — keep only first token
    label = re.split(r"[\s\n]+", label.strip())[0].lower()
    return thinking, label


def get_resume_index(output_dir: Path) -> int:
    """Count already-completed rows from JSONL shards in output_dir."""
    if not output_dir.exists():
        return 0
    total = 0
    for fpath in sorted(output_dir.glob("shard_*.jsonl")):
        try:
            lines = [l for l in fpath.read_text().splitlines() if l.strip()]
            total += len(lines)
        except Exception:
            pass
    if total > 0:
        print(f"Resuming from row {total:,} (found existing shards).")
    return total


def next_shard_path(output_dir: Path) -> Path:
    existing = list(output_dir.glob("shard_*.jsonl"))
    idx = len(existing)
    return output_dir / f"shard_{idx:06d}.jsonl"


# ── Async request ─────────────────────────────────────────────────────────────

async def classify_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    text: str,           # only the raw text — no label/class column
    row_index: int,
) -> tuple[str, str, str]:
    """Send one chat-completion request; retry until label is valid.
    Returns (text, thinking, label).
    """
    payload = {
        "messages":    build_messages(text),   # prompt uses text only
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream":      False,
    }

    thinking = ""
    label    = ""

    for attempt in range(1, MAX_RETRIES + 1):
        async with sem:
            resp = await client.post(CHAT_URL, json=payload, timeout=300.0)
            resp.raise_for_status()

        message  = resp.json()["choices"][0]["message"]

        # llama.cpp with --reasoning on exposes separate fields
        thinking = (message.get("reasoning_content") or "").strip()
        raw      = (message.get("content")           or "").strip()

        # Fallback: parse <think> tags when reasoning_content is absent
        if not thinking:
            thinking, raw = parse_output(raw)

        # Normalise: keep only the first token, lowercase
        label = re.split(r"[\s\n]+", raw)[0].lower() if raw else ""

        if label in VALID_LABELS:
            break   # valid — no retry needed

        print(f"\n[WARN] row {row_index}: invalid label {label!r} "
              f"(attempt {attempt}/{MAX_RETRIES}) — retrying")

    if label not in VALID_LABELS:
        print(f"\n[ERROR] row {row_index}: still invalid label {label!r} "
              f"after {MAX_RETRIES} attempts — storing as-is")

    return text, thinking, label


# ── Main ──────────────────────────────────────────────────────────────────────

async def main_async():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {INPUT_CSV} ...")
    df    = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Total rows: {total:,}")
    print(f"Llama server : {CHAT_URL}")
    print(f"Concurrency  : {CONCURRENCY}")
    print(f"Output dir   : {OUTPUT_DIR}")

    start_row = get_resume_index(OUTPUT_DIR)
    df_todo   = df.iloc[start_row:].reset_index(drop=True)

    if len(df_todo) == 0:
        print("All rows already processed.")
        _merge_and_save(OUTPUT_DIR, total)
        return

    sem    = asyncio.Semaphore(CONCURRENCY)
    pbar   = tqdm(total=total, initial=start_row, unit="row",
                  dynamic_ncols=True, desc="Classifying")

    # Fire requests in sliding-window batches so we can checkpoint often
    BATCH_SIZE  = CONCURRENCY * 4   # checkpoint every N rows
    # Keep only the columns we need; class/label col stays out of the prompt
    text_col    = "text"
    all_texts   = df_todo[text_col].tolist()      # pure text — no class
    all_indices = list(range(start_row, start_row + len(df_todo)))

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(all_texts), BATCH_SIZE):
            batch_texts   = all_texts[batch_start:batch_start + BATCH_SIZE]
            batch_indices = all_indices[batch_start:batch_start + BATCH_SIZE]

            tasks   = [
                classify_one(client, sem, txt, idx)
                for txt, idx in zip(batch_texts, batch_indices)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            shard_rows = []
            for res in results:
                if isinstance(res, Exception):
                    print(f"\n[WARN] Request failed: {res}")
                    continue
                txt, thinking, label = res
                # Output row: text (exact CSV value) + thinking + label only
                shard_rows.append({
                    "text":     txt,
                    "thinking": thinking,
                    "label":    label,
                })

            # Write shard JSONL checkpoint
            if shard_rows:
                shard_path = next_shard_path(OUTPUT_DIR)
                with open(shard_path, "w", encoding="utf-8") as f:
                    for r in shard_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            pbar.update(len(shard_rows))

    pbar.close()
    print(f"\nAll rows processed. Merging shards → {OUTPUT_CSV}")
    _merge_and_save(OUTPUT_DIR, total)


def _merge_and_save(output_dir: Path, total: int):
    """Merge all JSONL shards into a single CSV."""
    shards = sorted(output_dir.glob("shard_*.jsonl"))
    if not shards:
        print("No shard files found — nothing to merge.")
        return

    frames = []
    for s in shards:
        lines = [l for l in s.read_text().splitlines() if l.strip()]
        if lines:
            frames.append(pd.DataFrame([json.loads(l) for l in lines]))

    if not frames:
        print("Shards are empty — nothing to merge.")
        return

    final = pd.concat(frames, ignore_index=True)
    final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {len(final):,} / {total:,} rows → {OUTPUT_CSV}")
    print("\nLabel distribution:")
    print(final["label"].value_counts())


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()