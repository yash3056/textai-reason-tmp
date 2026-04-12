"""
vLLM offline batch inference — depression classification
Model: Gemma 4 (chat template with enable_thinking=True)

Features:
  - Gemma 4 <|channel>thought...<channel|> thinking format (+ DeepSeek/Qwen fallback)
  - Uses llm.chat() so the chat template + enable_thinking are applied correctly
  - Writes results to CSV after every batch (no data loss on crash)
  - Resumes from last completed batch if OUTPUT_CSV already exists
  - tqdm progress bar: rows done / total, rows/sec, ETA
  - TENSOR_PARALLEL_SIZE env var to switch 2-GPU (test) vs 4-GPU (prod)
"""

import os
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "/model"
INPUT_CSV       = "/data/Suicide_Detection.csv"
OUTPUT_CSV      = "/data/output.csv"
BATCH_SIZE      = 256                # smaller = more frequent saves
MAX_TOKENS      = 16384

def get_tensor_parallel_size() -> int:
    """Auto-detect available GPUs, fall back to env var, then 1."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"Auto-detected {n} GPU(s), setting tensor_parallel_size={n}")
        return n
    return 1

TENSOR_PARALLEL = get_tensor_parallel_size()

SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,             # greedy — deterministic labels
    max_tokens=MAX_TOKENS,
)

SYSTEM_PROMPT = (
    "You are an expert mental health research assistant specializing in the "
    "binary classification of text into \"depression\" or \"non-depression.\""
)

USER_TEMPLATE = """\
**Objective:** Analyze the provided `<Input_text>` and determine if it exhibits markers of clinical depression or situational distress.

**Process:**
You must perform a detailed analysis within your thinking process before providing the final answer. Your analysis must follow these steps:
1. **Emotional Content:** Identify the primary emotions present (e.g., sadness, anger, curiosity, neutrality).
2. **Symptom Check:** Look for clinical markers (e.g., anhedonia, hopelessness, sleep/appetite disturbances, persistent low mood, suicidal ideation).
3. **Behavioral Patterns:** Evaluate the behavior described (e.g., social withdrawal, fatigue, inability to function).
4. **Conclusion:** State why the text does or does not meet the criteria for the "depression" label.

**Output Constraints:**
- Your thinking process must be thorough and detailed.
- After the thinking process is complete, you must output **only one word**: either "depression" or "non-depression".
- Do not include any conversational filler, introductory phrases, or explanations after the thinking block.

**Input Variable:**
<Input_text>{input_text}</Input_text>\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_messages(text: str) -> list[dict]:
    """Build the message list for llm.chat() — system + user turn."""
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": USER_TEMPLATE.format(input_text=text)},
    ]


def parse_output(raw: str) -> tuple[str, str]:
    """
    Extract (thinking, label) from raw model output.

    Handles three formats:
      1. Gemma 4:       <|channel>thought\\n...\\n<channel|>label
      2. DeepSeek/Qwen: <think>...</think>label
      3. No thinking:   label only
    """
    raw = raw.strip()

    # ── Gemma 4 format ────────────────────────────────────────────────────────
    if "<|channel>" in raw and "<channel|>" in raw:
        chan_open  = "<|channel>"
        chan_close = "<channel|>"
        start      = raw.index(chan_open) + len(chan_open)
        end        = raw.index(chan_close)
        think_body = raw[start:end]
        # Gemma emits "thought\n" right after the opening tag — strip it
        if think_body.startswith("thought"):
            think_body = think_body[len("thought"):].lstrip("\n")
        thinking = think_body.strip()
        label    = raw[end + len(chan_close):].strip().lower()

    # ── DeepSeek / Qwen format ────────────────────────────────────────────────
    elif "<think>" in raw and "</think>" in raw:
        start    = raw.index("<think>") + len("<think>")
        end      = raw.index("</think>")
        thinking = raw[start:end].strip()
        label    = raw[end + len("</think>"):].strip().lower()

    # ── No thinking block ─────────────────────────────────────────────────────
    else:
        thinking = ""
        label    = raw.lower()

    # Normalise label to one of three values
    if "non-depression" in label:
        label = "non-depression"
    elif "depression" in label:
        label = "depression"
    else:
        label = "unknown"

    return thinking, label


def get_resume_index(output_csv: str) -> int:
    """Return number of already-completed rows (enables crash resume)."""
    if not os.path.exists(output_csv):
        return 0
    try:
        done      = pd.read_csv(output_csv)
        completed = done[done["label"].notna() & (done["label"] != "")]
        n         = len(completed)
        if n > 0:
            print(f"Resuming from row {n:,} (found existing output).")
        return n
    except Exception:
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model (tensor_parallel_size={TENSOR_PARALLEL}) ...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        enable_chunked_prefill=True,
        max_model_len=16384,
    )

    print(f"Reading {INPUT_CSV} ...")
    df    = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Total rows: {total:,}")

    # ── Resume support ────────────────────────────────────────────────────────
    start_row = get_resume_index(OUTPUT_CSV)
    df_todo   = df.iloc[start_row:].reset_index(drop=True)
    remaining = len(df_todo)

    if remaining == 0:
        print("All rows already processed.")
        return

    # ── Progress bar ──────────────────────────────────────────────────────────
    pbar = tqdm(
        total=total,
        initial=start_row,
        unit="row",
        dynamic_ncols=True,
        desc="Classifying",
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} rows "
            "[{elapsed}<{remaining}, {rate_fmt}]"
        ),
    )

    # ── Batch loop ────────────────────────────────────────────────────────────
    first_write = not os.path.exists(OUTPUT_CSV)

    for batch_start in range(0, remaining, BATCH_SIZE):
        batch_end  = min(batch_start + BATCH_SIZE, remaining)
        batch_df   = df_todo.iloc[batch_start:batch_end]

        # Build conversation messages for every row in the batch
        conversations = [build_messages(t) for t in batch_df["text"].tolist()]

        # llm.chat() applies the Gemma 4 chat template correctly and passes
        # enable_thinking so the <|think|> token is injected by the template
        outputs = llm.chat(
            conversations,
            sampling_params=SAMPLING_PARAMS,
            chat_template_kwargs={"enable_thinking": True},
        )

        rows       = []
        last_label = ""
        for i, out in enumerate(outputs):
            raw             = out.outputs[0].text
            thinking, label = parse_output(raw)
            row             = batch_df.iloc[i].to_dict()
            row["thinking"] = thinking
            row["label"]    = label
            rows.append(row)
            last_label      = label

        # Write this batch immediately — safe even if job is killed after this
        pd.DataFrame(rows).to_csv(
            OUTPUT_CSV,
            mode="a",            # append each batch
            header=first_write,  # header only on very first write
            index=False,
        )
        first_write = False

        rows_done = start_row + batch_end
        pbar.update(len(rows))
        pbar.set_postfix({
            "last":  last_label,
            "done":  f"{rows_done:,}",
            "left":  f"{total - rows_done:,}",
        })

    pbar.close()
    print(f"\nDone. Results saved to {OUTPUT_CSV}")

    final = pd.read_csv(OUTPUT_CSV)
    print(final["label"].value_counts())


if __name__ == "__main__":
    main()