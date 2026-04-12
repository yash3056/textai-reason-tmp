# ── Base: official vLLM image (CUDA 12.4, H100-ready) ────────────────────────
# Pull this on an internet machine, then transfer as a .tar or convert to .sif
# docker pull vllm/vllm-openai:latest
#  docker build -t depression_vllm .
# docker run --gpus all -it --entrypoint /bin/bash depression_vllm 
FROM vllm/vllm-openai:gemma4

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir pandas tqdm transformers -U

# ── Copy inference script ─────────────────────────────────────────────────────
WORKDIR /app
COPY classify_vllm.py /app/classify_vllm.py

# ── Path config — host paths mounted into container ───────────────────────────
# Host:    /home/aza/workspace/textai-reason/gemma-5b-model  →  /model
# Host:    /home/aza/workspace/textai-reason/                →  /data
#   input:  /data/test.csv
#   output: /data/output.csv
ENV MODEL_PATH=/model \
    INPUT_CSV=/data/test.csv \
    OUTPUT_CSV=/data/output.csv

# ── Run with Docker ───────────────────────────────────────────────────────────
# docker run --gpus all \
#   -v /home/aza/workspace/textai-reason/gemma-5b-model:/model \
#   -v /home/aza/workspace/textai-reason:/data \
#   depression_vllm

# ── Run with Singularity (SIF on HPC) ────────────────────────────────────────
# singularity exec --nv \
#   --bind /home/aza/workspace/textai-reason/gemma-5b-model:/model \
#   --bind /home/aza/workspace/textai-reason:/data \
#   depression_vllm.sif python3 /app/classify_vllm.py

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["python3", "/app/classify_vllm.py"]