# Running Megatron-LM on Polaris (ALCF)

A practical, end-to-end log of getting **Megatron-LM** to run a real training job on
[Polaris](https://docs.alcf.anl.gov/polaris/) at the Argonne Leadership Computing
Facility (ALCF) — starting from a fresh checkout and ending with a single-GPU GPT
pretraining run that actually steps and reduces loss.

This is written as a **troubleshooting guide** rather than polished documentation. The
goal is to save the next person the half-day of environment debugging that this took,
especially the parts where the obvious approach silently fails.

> **Scope note.** Everything below was verified on Polaris in May 2026 with the
> `conda/2025-09-25` module and a recent Megatron-LM checkout (`megatron-core 0.18.0`).
> Software stacks on shared HPC systems change; treat versions and paths as things to
> re-check, not as guarantees. Where a claim is "this worked once for me" rather than
> "this is documented behavior," it's flagged as such.

---

## TL;DR

If you just want the working recipe:

```bash
# 1. Get a compute node (PBS, not Slurm) — replace <PROJECT> with your allocation
qsub -I -A <PROJECT> -q debug -l select=1:system=polaris -l walltime=00:30:00 -l filesystems=home:grand

# 2. On the compute node, set up the environment
source ~/setup_env.sh           # contents below

# 3. Verify the env is sane (this is where most problems hide)
python3 -c "import torch, transformer_engine, flash_attn; print('all ok')"
python3 -c "import megatron.core; print(megatron.core.__file__)"   # must point at YOUR repo

# 4. Run the minimal single-GPU training
cd /path/to/Megatron-LM
bash run_single_gpu.sh
```

The two non-obvious fixes that make this work are
[`NVTE_CUDA_INCLUDE_DIR`](#fix-1-transformer-engine-cant-find-cuda-headers) and
[an empty top-level `megatron/__init__.py`](#fix-2-a-stale-conda-megatron-hijacks-the-import).
Both are explained below.

---

## Environment

Polaris compute nodes (verified May 2026): **4× NVIDIA A100-SXM4-40GB** per node, PBS
Pro scheduler, environment provided via `module`. The two facts that shape everything:

- **You build/run in an ALCF-provided conda environment**, not a hand-rolled `pip`
  environment. Trying to `pip install torch` / `transformer-engine` / `flash-attn`
  yourself leads straight into CUDA-version dependency hell. The conda module already
  has them, compiled to match.
- **Login nodes have no GPU.** `import torch` may succeed there, but anything that
  touches CUDA only proves out on a compute node. Do real verification on a compute
  node.

### `setup_env.sh`

```bash
module use /soft/modulefiles
module load conda
conda activate base
export NVTE_CUDA_INCLUDE_DIR=$CUDA_HOME/include
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
```

Source this in **every** new shell / every new compute node — a fresh node does not
inherit the login node's environment.

> Replace `/path/to/Megatron-LM` with your actual checkout path. On Polaris, put the
> repo and data on a persistent filesystem like **Grand** (`/grand/projects/...`), not
> on `home` (small quota) and never on a node's local NVMe (wiped when the job ends).

---

## The two fixes that matter

Most of the setup is unremarkable. These two are the ones that cost real time, because
in both cases the naive approach *appears* to work and then fails later.

### Fix 1: Transformer Engine can't find CUDA headers

**Symptom.** `import transformer_engine` crashes with:

```
TypeError: argument should be a str or an os.PathLike object where __fspath__
returns a str, not 'NoneType'
```

...originating in `transformer_engine/common/__init__.py` at `_nvidia_cudart_include_dir()`.

**Cause.** TE tries to locate CUDA headers via a pip-installed `nvidia` package
(`nvidia.__file__`). On Polaris, CUDA comes from a module, not a pip package, so that
lookup returns `None` and `Path(None)` throws.

**Fix.** Tell TE where CUDA is explicitly. TE checks the `NVTE_CUDA_INCLUDE_DIR`
environment variable first and skips the auto-detection:

```bash
export NVTE_CUDA_INCLUDE_DIR=$CUDA_HOME/include
```

(Already included in `setup_env.sh` above.)

### Fix 2: A stale conda `megatron` hijacks the import

This is the subtle one.

**Symptom.** Running `tools/preprocess_data.py` (or importing anything under
`megatron.core`) fails with:

```
ModuleNotFoundError: No module named 'megatron.core.tokenizers'
```

or, when you look closer, a traceback that runs through
`/soft/applications/conda/.../site-packages/megatron/__init__.py` and dies on
`from tools.retro.utils import ...` → `No module named 'tools.retro'`.

**Cause.** Two different `megatron` packages are on `sys.path`:

1. Your checkout's `megatron/` — the modern layout, where `megatron.core.*` lives.
   But in recent Megatron-LM, the **top-level `megatron/` directory has no
   `__init__.py`** (it's a namespace package).
2. The conda environment's `site-packages/megatron/` — an old, broken copy that
   *does* have an `__init__.py`.

When Python resolves the top-level `megatron` package, a namespace package (no
`__init__.py`) loses to a regular package (has `__init__.py`) even if the namespace
package appears earlier on `sys.path`. So Python picks the conda copy's
`__init__.py`, which then fails to import. Your `megatron.core` never gets a chance.

You can confirm which copy wins:

```bash
python3 -c "import megatron.core; print(megatron.core.__file__)"
# BAD:  /soft/applications/conda/.../site-packages/megatron/core/__init__.py
# GOOD: /path/to/Megatron-LM/megatron/core/__init__.py
```

**Fix.** Give your checkout a top-level `__init__.py`. This promotes your `megatron/`
from a (low-priority) namespace package to a (high-priority) regular package; since it
sits earlier on `sys.path`, it now wins outright and the conda copy is never consulted:

```bash
touch /path/to/Megatron-LM/megatron/__init__.py
```

Re-run the check above — `megatron.core.__file__` should now point at your repo.

> **Why not just uninstall the conda copy?** It lives in a shared, read-only
> ALCF environment (`/soft/...`); you don't own it and shouldn't modify it. Adding a
> file to *your* checkout is the clean, permission-safe fix.
>
> **Note:** this `__init__.py` is not part of upstream Megatron-LM, so `git status`
> will show it as untracked, and `git pull` / `git checkout` may interact with it. Keep
> a note that you added it deliberately.

A useful general lesson here: when a Python import goes to the wrong place, `pip show
<package>` (look at the `Location:` field) tells you where a package actually lives far
faster than repeatedly poking at `import`.

---

## Preparing data

Megatron doesn't read raw text — it reads a preprocessed binary format
(`.bin` + `.idx`). For a smoke test, any text works.

```bash
# Tokenizer files (GPT-2 BPE)
cd /path/to/data
wget https://huggingface.co/gpt2/resolve/main/vocab.json -O gpt2-vocab.json
wget https://huggingface.co/gpt2/resolve/main/merges.txt  -O gpt2-merges.txt

# A throwaway dataset, JSON Lines with a "text" field
python3 -c "
import json
with open('sample.jsonl','w') as f:
    for i in range(1000):
        f.write(json.dumps({'text': f'This is sample sentence number {i}. '
                                     'The quick brown fox jumps over the lazy dog.'})+'\n')
"
```

> In my testing, Polaris **compute** nodes had outbound network access, so the `wget`
> calls worked from a compute node. I would not rely on this being true for every node
> at every time — if downloads hang, do them from a login node instead (your data on
> Grand is visible from both).

Convert to Megatron's binary format:

```bash
cd /path/to/Megatron-LM
python3 tools/preprocess_data.py \
    --input ../data/sample.jsonl \
    --output-prefix ../data/sample \
    --vocab-file ../data/gpt2-vocab.json \
    --merge-file ../data/gpt2-merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --workers 4 \
    --append-eod
```

Success looks like `Processed 1000 documents (...)` and produces
`sample_text_document.bin` and `sample_text_document.idx`.

---

## Minimal single-GPU training

This is deliberately a **toy** model (4 layers, hidden size 512) with **no
parallelism** — the point is to prove the infrastructure works end to end, not to train
anything useful. It's adapted from `examples/gpt3/train_gpt3_175b_distributed.sh` by
shrinking the model, setting all parallel sizes to 1, and using a single GPU.

### `run_single_gpu.sh`

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

DATA_PATH=../data/sample_text_document
VOCAB_FILE=../data/gpt2-vocab.json
MERGE_FILE=../data/gpt2-merges.txt

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --master_addr localhost
    --master_port 6000
)
GPT_MODEL_ARGS=(
    --num-layers 4
    --hidden-size 512
    --num-attention-heads 8
    --seq-length 1024
    --max-position-embeddings 1024
    --attention-backend auto
)
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters 20
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 6.0e-4
    --lr-decay-style cosine
    --min-lr 6.0e-5
    --lr-warmup-fraction .01
    --lr-decay-iters 20
)
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)
DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
)
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 100
    --eval-iters 5
)
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
```

### What success looks like

Loss should decrease monotonically over the 20 steps, with zero NaN iterations:

```
iteration  1/20 | lm loss: 1.084000E+01 | ...
iteration  5/20 | lm loss: 9.733748E+00 | ...
iteration 10/20 | lm loss: 8.469354E+00 | ...
iteration 15/20 | lm loss: 7.088836E+00 | ...
iteration 20/20 | lm loss: 6.578415E+00 | ...
```

The first couple of iterations are slow (CUDA warmup/JIT); steady state on the toy
model was ~150 ms/iter and ~1.2 GB peak GPU memory — i.e. there's enormous headroom on
a 40 GB A100 for bigger models and parallelism.

Two log lines worth noting for what comes next:

```
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
```

These are the knobs you turn to start exploring parallelism (tensor/pipeline/expert).

---

## Quick reference

| Thing | Value |
|---|---|
| Scheduler | PBS Pro (`qsub`, not `sbatch`) |
| GPUs per node | 4× A100-SXM4-40GB |
| Conda module | `conda` from `/soft/modulefiles` (`2025-09-25` at time of writing) |
| Persistent storage | Grand (`/grand/projects/...`) |
| Required env var | `NVTE_CUDA_INCLUDE_DIR=$CUDA_HOME/include` |
| Required repo tweak | `touch megatron/__init__.py` |
| Sanity check | `import torch, transformer_engine, flash_attn` |
| Right-repo check | `megatron.core.__file__` points at your checkout |

---

## Things I haven't verified

To be honest about the edges of this guide:

- Multi-GPU and multi-node runs (tensor/pipeline/expert parallelism). The single-GPU
  run is confirmed; scaling up is the obvious next step and has its own set of issues
  (distributed launch, inter-node networking).
- Whether the `flash-attn 2.8.3` version warning (`Supported flash-attn versions are
  >= 2.1.1, <= 2.8.1`) ever causes real problems. It was only a warning here.
- Long-term reproducibility across ALCF software refreshes.

PRs / issues with corrections welcome.
