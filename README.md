# Litesearch

*Fork of [autoresearch](https://github.com/karpathy/autoresearch) optimized for consumer GPUs (2GB–32GB+ VRAM) with a GUI.*

## What is this?

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

**Litesearch** adds:
- **Consumer GPU support**: Works on 2GB–32GB+ VRAM GPUs (GTX 1060 through RTX 4090+)
- **Auto VRAM scaling**: Model size, batch size, and sequence length automatically fit your GPU
- **GUI dashboard**: CustomTkinter interface with live training log, VRAM bar, and config sliders
- **Pascal support**: Automatic fp32 fallback for GTX 10-series GPUs
- **Gradient checkpointing**: Always enabled to minimize memory usage

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+.

### Option A: pip (standard)

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data and train tokenizer (one-time, ~2 min)
python prepare.py

# 4a. Launch the GUI
python gui.py

# 4b. Or run headless (original autoresearch style)
python train.py
```

### Option B: uv (faster)

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4a. Launch the GUI
uv run gui.py

# 4b. Or run headless
uv run train.py
```

## The GUI

```
+-------------------------------------------------------+
|  Litesearch — Autonomous Research for Consumer GPUs    |
+-------------------------------------------------------+
|  VRAM Budget:      [========slider========] 4.0 GB    |
|  Learning Rate:    [========slider========] 0.040      |
|  GPU: NVIDIA RTX 3060 (5.7 GB) | bfloat16             |
|                                                       |
|  Config: depth=6 d=384 B=4 T=512 ~32M params ~1.2GB  |
|                                                       |
|  [ Start Research ]            [ Stop ]               |
+-------------------------------------------------------+
|  VRAM: [████████░░░░░░░░] 2.1 / 5.7 GB               |
+-------------------------------------------------------+
|  > step 00042 (14.2%) | loss: 3.421 | mfu: 23.1%      |
|  (scrolling terminal log)                             |
+-------------------------------------------------------+
```

**Controls:**
- **VRAM slider**: Set your GPU's VRAM budget. The model auto-scales to fit, leaving 800MB for the OS.
- **Matrix LR slider**: Override the Muon optimizer learning rate (default 0.04).
- **Start Research**: Begins a training experiment with auto-computed config.
- **Stop**: Gracefully stops training after the current step.

## VRAM Auto-Scaling

The model automatically adapts to your GPU. Set the slider to your available VRAM:

| VRAM Budget | Depth | n_embd | Batch | Seq Len | ~Params | Est. VRAM |
|-------------|-------|--------|-------|---------|---------|-----------|
| 2 GB | 4 | 256 | 1 | 512 | ~8M | ~600 MB |
| 4 GB | 6 | 384 | 2 | 512 | ~17M | ~1.2 GB |
| 8 GB | 8 | 512 | 4 | 1024 | ~33M | ~2.8 GB |
| 16 GB | 12 | 768 | 8 | 2048 | ~86M | ~7.5 GB |
| 32 GB | 16 | 1024 | 16 | 2048 | ~183M | ~15 GB |

Always leaves an 800MB safety buffer. Gradient checkpointing keeps activation memory low.

## Running the agent (headless)

You can still use the original autoresearch agent workflow:

```
Hi have a look at program.md and let's kick off a new experiment!
```

The `program.md` file provides agent instructions. Point your agent here and let it go.

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop, VRAM auto-config (agent modifies this)
gui.py          — CustomTkinter dashboard (new)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Changes from autoresearch

- **FlashAttention-3 replaced** with `torch.nn.functional.scaled_dot_product_attention` (built-in, no extra dependencies)
- **Sliding window attention removed** — standard causal attention for maximum compatibility
- **Gradient checkpointing** always enabled — trades compute for memory
- **Pascal GPU support** — automatic fp32 fallback on GTX 10-series (pre-Turing)
- **Peak FLOPs lookup table** — accurate MFU calculation for any GPU
- **`kernels` dependency removed** — no custom CUDA kernel dependencies
- **`customtkinter` added** — for the GUI dashboard
- **`run_training()` function** — clean API for GUI integration
- **`compute_optimal_config()`** — automatic model sizing based on VRAM budget

## Platform support

| GPU | Status | Notes |
|-----|--------|-------|
| RTX 4090 / 4080 | ✅ | bfloat16, best performance |
| RTX 3090 / 3080 | ✅ | bfloat16 |
| RTX 2080 Ti | ✅ | bfloat16 |
| GTX 1080 Ti / 1080 | ✅ | fp32 fallback (~2x memory) |
| GTX 1060 6GB | ⚠️ | Works but very limited (small models) |
| AMD / Intel | ❌ | Not supported (would need ROCm/XPU backend) |

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
