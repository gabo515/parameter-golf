# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenAI's Parameter Golf challenge: train the best language model that fits in a 16MB artifact (code bytes + int8+zlib compressed model) and trains in under 10 minutes on 8xH100s, scored by bits-per-byte (BPB) on FineWeb validation.

## Key Commands

### Download data (sp1024 tokenizer, 80 train shards by default)
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10  # small subset
python3 data/cached_challenge_fineweb.py --variant sp1024                    # full 8B tokens
```

### Train on GPU (CUDA required for train_gpt.py)
```bash
# Single GPU
RUN_ID=my_run torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (competition target)
RUN_ID=my_run torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Train on Apple Silicon (MLX)
```bash
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

### Key env vars for overriding defaults
- `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE` - dataset/tokenizer selection
- `ITERATIONS`, `MAX_WALLCLOCK_SECONDS` (default 600s), `TRAIN_BATCH_TOKENS`
- `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`
- `VAL_LOSS_EVERY` (set >0 for periodic val), `VAL_BATCH_SIZE`
- `TIE_EMBEDDINGS` (1=tied, default), `LOGIT_SOFTCAP`

## Architecture

### train_gpt.py (CUDA/PyTorch, ~900 lines)
Single self-contained training script. Hard cap of 1500 lines. Contains:

- **Hyperparameters** class: all config via env vars with defaults
- **Muon optimizer**: Newton-Schulz orthogonalization for matrix params (from modded-nanogpt)
- **BPB evaluation**: tokenizer-agnostic bits-per-byte via SentencePiece lookup tables (not raw cross-entropy loss)
- **Int8 quantization + zlib compression**: post-training export pipeline that quantizes to int8 (per-row scales for 2D, per-tensor for vectors), then zlib compresses. Small/control tensors kept as fp16.
- **GPT model**: U-Net style skip connections (encoder half stores activations, decoder half reuses them in reverse), GQA attention with RoPE, ReLU-squared MLP, RMSNorm, logit softcapping
- **Optimizer split**: embedding (Adam), matrix params (Muon), scalars (Adam), optional separate head (Adam) -- each with different LRs
- **Distributed**: DDP via torchrun, grad_accum_steps = 8 / world_size

### train_gpt_mlx.py (Apple Silicon/MLX, ~900 lines)
MLX port of the same architecture for local iteration on Mac.

### data/
- `cached_challenge_fineweb.py` - manifest-driven downloader from HuggingFace (`willdepueoai/parameter-golf`)
- `download_hf_docs_and_tokenize.py` - rebuild tokenizers/shards from raw docs
- `tokenizer_specs.json` - tokenizer family definitions
- Data stored in `data/datasets/fineweb10B_<variant>/` as binary shards (uint16 tokens with 256-int header)

### records/
Submissions organized as `records/track_<constraint>/<date_name>/` containing `train_gpt.py`, `submission.json`, `train.log`, and `README.md`.

## Submission Rules
- Must beat SOTA by >= 0.005 nats with p < 0.01
- Artifact = code bytes + compressed model bytes, cap is 16,000,000 bytes (decimal, not MiB)
- No external downloads or network calls during evaluation
- PRs add a new folder under `/records/` only
