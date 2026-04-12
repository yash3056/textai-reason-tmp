import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = os.environ.get("INPUT_CSV",  "/data/test.csv")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "/data/output.csv")
BATCH_SIZE = 256
MODEL_NAME = "google/gemma-4-31B-it"

client = OpenAI(
    base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key="EMPTY"
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

<Input_text>{input_text}</Input_text>\
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_text(text: str) -> tuple[str, str]:
    """Call vLLM server, return (thinking, label)."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(input_text=text)},
        ],
        max_tokens=4096,
        temperature=0.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True}
        }
    )
    message = response.choices[0].message
    thinking = message.reasoning if hasattr(message, "reasoning") and message.reasoning else ""
    label    = (message.content or "").strip().lower()

    if "non-depression" in label:
        label = "non-depression"
    elif "depression" in label:
        label = "depression"
    else:
        label = "unknown"

    return thinking, label


def get_resume_index(output_csv: str) -> int:
    if not os.path.exists(output_csv):
        return 0
    try:
        done = pd.read_csv(output_csv)
        completed = done[done["label"].notna() & (done["label"] != "")]
        n = len(completed)
        if n > 0:
            print(f"Resuming from row {n:,}.")
        return n
    except Exception:
        return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading {INPUT_CSV} ...")
    df    = pd.read_csv(INPUT_CSV)
    total = len(df)
    print(f"Total rows: {total:,}")

    start_row = get_resume_index(OUTPUT_CSV)
    df_todo   = df.iloc[start_row:].reset_index(drop=True)

    if len(df_todo) == 0:
        print("All rows already processed.")
        return

    first_write = not os.path.exists(OUTPUT_CSV)

    pbar = tqdm(total=total, initial=start_row, unit="row",
                dynamic_ncols=True, desc="Classifying")

    for batch_start in range(0, len(df_todo), BATCH_SIZE):
        batch_df = df_todo.iloc[batch_start:batch_start + BATCH_SIZE]
        rows = []

        for i, row_data in batch_df.iterrows():
            thinking, label = classify_text(row_data["text"])
            row = row_data.to_dict()
            row["thinking"] = thinking
            row["label"]    = label
            rows.append(row)
            pbar.update(1)

        pd.DataFrame(rows).to_csv(
            OUTPUT_CSV, mode="a", header=first_write, index=False
        )
        first_write = False

    pbar.close()
    print(f"\nDone. Results saved to {OUTPUT_CSV}")

    final = pd.read_csv(OUTPUT_CSV)
    print(final["label"].value_counts())


if __name__ == "__main__":
    main()