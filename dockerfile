# ── Base: PyTorch + CUDA (no vLLM server needed) ─────────────────────────────
# docker build -t depression_classify .
# docker run --gpus all \
#   -v /home/aza/workspace/textai-reason/gemma-5b-model:/model \
#   -v /home/aza/workspace/textai-reason:/data \
#   depression_classify

#   -e INPUT_CSV=/data/my_other_file.csv \
#   -e OUTPUT_JSON=/data/my_other_output.json \

FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime
# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir pandas tqdm transformers accelerate -U --break-system-packages

# ── Copy script ───────────────────────────────────────────────────────────────
WORKDIR /app
COPY final.py /app/final.py

# ── Defaults (all overridable with -e at runtime) ─────────────────────────────
ENV MODEL_PATH=/model \
    INPUT_CSV=/data/test.csv \
    OUTPUT_JSON=/data/output.json \
    BATCH_SIZE=8

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python3", "/app/final.py"]