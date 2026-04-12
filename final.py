import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.environ.get("MODEL_PATH",  "/model")
INPUT_CSV      = os.environ.get("INPUT_CSV",   "/data/test.csv")
OUTPUT_JSON    = os.environ.get("OUTPUT_JSON",  "/data/output.json")
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", "8"))
MAX_NEW_TOKENS = 8192

PROMPT_TEMPLATE = """**Role:** You are an expert mental health research assistant specializing in the binary classification of text into "depression" or "non-depression."

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
<Input_text>{input_text}</Input_text>"""

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(text: str) -> str:
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(input_text=text)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def parse_output(raw: str) -> tuple[str, str]:
    raw = raw.strip() if raw else ""

    # if not hasattr(parse_output, "_printed"):
    #     print("\n=== FIRST RAW OUTPUT (debug) ===")
    #     print(repr(raw[:500]))
    #     print("================================\n")
    #     parse_output._printed = True

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

    return thinking, label


def get_resume_index(output_json: str) -> int:
    if not os.path.exists(output_json):
        return 0
    try:
        done      = pd.read_json(output_json, orient="records", lines=True)
        completed = done[done["label"].notna() & (done["label"] != "")]
        n         = len(completed)
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

    start_row   = get_resume_index(OUTPUT_JSON)
    df_todo     = df.iloc[start_row:].reset_index(drop=True)

    if len(df_todo) == 0:
        print("All rows already processed.")
        return

    first_write = not os.path.exists(OUTPUT_JSON)

    pbar = tqdm(total=total, initial=start_row, unit="row",
                dynamic_ncols=True, desc="Classifying")

    for batch_start in range(0, len(df_todo), BATCH_SIZE):
        batch_df = df_todo.iloc[batch_start:batch_start + BATCH_SIZE]
        texts    = batch_df["text"].tolist()

        prompts = [build_prompt(t) for t in texts]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        rows = []
        for i, output_ids in enumerate(outputs):
            new_tokens      = output_ids[prompt_len:]
            raw             = tokenizer.decode(new_tokens, skip_special_tokens=False)
            thinking, label = parse_output(raw)

            row             = batch_df.iloc[i].to_dict()
            row["thinking"] = thinking
            row["label"]    = label
            rows.append(row)

        # JSONL — one JSON object per line, safe to append per batch
        pd.DataFrame(rows).to_json(
            OUTPUT_JSON,
            orient="records",
            lines=True,
            mode="a" if not first_write else "w",
            force_ascii=False,
        )
        first_write = False
        pbar.update(len(rows))

    pbar.close()
    print(f"\nDone. Results saved to {OUTPUT_JSON}")

    final = pd.read_json(OUTPUT_JSON, orient="records", lines=True)
    print(final["label"].value_counts())


if __name__ == "__main__":
    main()