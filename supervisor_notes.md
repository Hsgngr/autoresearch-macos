# Supervisor Notes

## Last updated
2026-03-11 — 9 experiments run, best val_bpb=1.347712

---

## ⚠️ CRITICAL: Always run on power adapter
Battery mode throttles M2 Max severely — step time degrades from ~0.34s to ~0.55s, cutting steps from ~890 to ~546. Any result with num_steps < 800 should be treated as invalid. Always plug in before running experiments.

## Hardware context (M2 Max 64GB, MPS)
- No `torch.compile` on MPS — every op is interpreted PyTorch, no kernel fusion
- MPS JIT-compiles Metal kernels on first use — adds ~100-200s overhead per experiment
- **MPS pre-warmup** added to train.py: runs a dummy forward+backward before the loop to trigger Metal compilation up front. Saves ~100-200s per run.
- `peak_vram_mb` always reports 0.0 on MPS — memory is shared with system RAM
- 64GB unified memory is not a bottleneck at current model sizes

---

## What has been tried (full history)

| commit  | val_bpb  | status  | description |
|---------|----------|---------|-------------|
| 21b86c4 | 1.379616 | keep    | baseline: DEPTH=4, dim=256, HEAD_DIM=128, TOTAL_BATCH=2^16, DEVICE_BATCH=16 → ~240 steps |
| 6e4ab46 | 1.764969 | discard | DEPTH=8 dim=512 50M params → only 69 steps, 14x slower per step on MPS |
| 352cab4 | 1.475373 | discard | HEAD_DIM=64 → 4 heads at dim=256 → 205 steps, slower on MPS for unknown reason |
| 5068279 | 1.360532 | keep    | TOTAL_BATCH=2^15 (32K), DEVICE_BATCH=16 → 457 steps, improved |
| a20e8a2 | crash    | crash   | TOTAL_BATCH=2^14 with DEVICE_BATCH=16 → assert fail (batch too small for device batch) |
| 097c37a | 1.347712 | keep    | TOTAL_BATCH=2^14 (16K), DEVICE_BATCH=8 → 890 steps, improved — **CURRENT BEST** |
| 515fa11 | 1.366126 | discard | TOTAL_BATCH=2^13 (8K), DEVICE_BATCH=4 → 1610 steps but batch too small, noise hurts |
| 709e870 | 1.350212 | discard | MATRIX_LR=0.06 → marginal, no improvement |
| f2a77ac | 1.352478 | discard | WARMDOWN_RATIO=0.3 → worse than 0.5 |
| 4b95143 | 1.349630 | discard | WINDOW_PATTERN=SSSL → marginally worse |
| e31ada2 | 1.350980 | discard | WEIGHT_DECAY=0.0 → no improvement |

---

## Key lessons learned

### Model scaling on MPS
- **Scaling depth/width is counterproductive on MPS** without torch.compile
- DEPTH=8 (50M params) runs 14x slower per step → only 69 steps in 5 min vs 240 for baseline
- HEAD_DIM=64 (4 heads at same dim) also slowed down to 205 steps — MPS inefficient for this shape
- **Stay at DEPTH=4, dim=256** until torch.compile is stable on MPS

### Batch size (highest-impact finding)
- Smaller batch → more gradient steps → better val_bpb (up to a limit)
- TOTAL_BATCH=2^16 (baseline): ~240 steps, val_bpb=1.3796
- TOTAL_BATCH=2^15: ~457 steps, val_bpb=1.3605 ✓ improved
- TOTAL_BATCH=2^14: ~890 steps, val_bpb=1.3477 ✓ improved again — **sweet spot**
- TOTAL_BATCH=2^13: ~1610 steps, val_bpb=1.3661 ✗ too small, gradient noise hurts
- **Sweet spot: TOTAL_BATCH=2^14 with DEVICE_BATCH=8**

### LR and schedule
- MATRIX_LR=0.04 is already near-optimal (0.06 didn't help)
- WARMDOWN_RATIO=0.5 is important — reducing to 0.3 made things worse
- WEIGHT_DECAY=0.2 is good — removing it didn't help
- WINDOW_PATTERN="L" (full context) is at least as good as "SSSL"

### Overhead
- Each experiment takes ~9-10 min total (300s training + ~240s startup overhead)
- The MPS pre-warmup addition should reduce overhead by ~100-200s per run
- After pre-warmup, target ~6-7 min per experiment

---

## Current best configuration
```
DEPTH = 4
ASPECT_RATIO = 64        # → dim=256
HEAD_DIM = 128           # → 2 heads
WINDOW_PATTERN = "L"
TOTAL_BATCH_SIZE = 2**14
DEVICE_BATCH_SIZE = 8
MATRIX_LR = 0.04
EMBEDDING_LR = 0.6
WARMDOWN_RATIO = 0.5
WEIGHT_DECAY = 0.2
```
val_bpb = **1.347712**

---

## Suggested next directions

1. **[NOW] Verify MPS pre-warmup saves time** — run one experiment and compare total_seconds vs before (~540s → ~360s target). If it works, all future experiments benefit.

2. **Try ASPECT_RATIO=32 with DEPTH=8** — gives dim=256 (same!) but 8 layers. Same per-layer compute as baseline but double the depth. Step time should be ~2x (since we doubled layers at same dim) → ~450 steps. More representational capacity, worth testing.

3. **Try WARMUP_RATIO=0.05** — small LR warmup at start. Only 45 steps of warmup (5% of 890), might stabilize early training.

4. **Try FINAL_LR_FRAC=0.05** — instead of decaying to 0, keep a small final LR. May prevent over-aggressive compression at end of training.

5. **Try EMBEDDING_LR=0.8** — embeddings have a lot of parameters (value_embeds = 4.2M of 11.5M total). Higher LR might help them converge faster.

6. **Try ADAM_BETAS=(0.9, 0.95)** — slightly higher beta1. The default (0.8) is tuned for Muon+AdamW combos but worth testing.

7. **Try GQA (n_kv_head=1)** — with DEPTH=4 and only 2 heads, using 1 KV head (GQA) reduces attention memory at same speed. Might allow DEVICE_BATCH_SIZE=16 at 2^15, combining the best of both. Less parameter sharing overhead.

---

## Warnings
- **Do NOT try DEPTH > 4** without ASPECT_RATIO reduction to keep dim=256
- **Do NOT go below TOTAL_BATCH=2^14** — sweet spot found, smaller batches hurt
- **Do NOT reduce WARMDOWN_RATIO below 0.5** — tested and confirmed worse
- **Do NOT try MATRIX_LR > 0.06** — not helpful at this config
- **Do NOT add new packages** — only what's in pyproject.toml
