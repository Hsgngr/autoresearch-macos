## Last updated
2026-03-11, 0 experiments logged (baseline run in progress, step ~209/~240)

## Assessment
The baseline is running cleanly — loss tracking well (~3.97 at 86% budget, entering cooldown), no crashes. However, the model is dramatically undersized for the hardware: depth=4, dim=256, only 2 attention heads, 11.5M params on a machine with 64GB unified memory. This is the dominant bottleneck — scaling the model is the single highest-leverage move available.

## What's been tried
- Baseline: DEPTH=4, ASPECT_RATIO=64 → dim=256, HEAD_DIM=128 → 2 heads, 11.5M params, WINDOW_PATTERN="L", WARMDOWN_RATIO=0.5

## Suggested next directions

1. **[HIGHEST PRIORITY] Scale up: DEPTH=8** — Set `DEPTH=8`, keep `ASPECT_RATIO=64`. This gives dim=512, HEAD_DIM=128 → 4 heads, ~50M params. The M2 Max 64GB will handle this easily at DEVICE_BATCH_SIZE=16. More params and more heads should substantially improve val_bpb.

2. **Scale further: DEPTH=12** — dim=768, HEAD_DIM=128 → 6 heads, ~124M params. Still fits comfortably. Try this if DEPTH=8 is a clear win.

3. **Reduce HEAD_DIM to 64** — At current DEPTH=4, dim=256: HEAD_DIM=64 → 4 heads. More attention diversity without changing depth. A simpler change than scaling depth, worth a quick comparison against DEPTH=8 once we have a baseline result.

4. **Try WINDOW_PATTERN="SSSL"** — After finding the right model size, try mixing short-window and long-window attention layers. May help the model learn both local and global patterns with the same compute.

5. **Increase DEVICE_BATCH_SIZE to 32** — On a larger model, if step time allows it, fewer grad_accum steps could improve throughput.

## Warnings
- **Do not stay at DEPTH=4** — two attention heads is severely limiting regardless of other hyperparameters.
- **MPS has no torch.compile** — no fusion benefit. Don't spend time on optimizer tricks that rely on kernel fusion.
- **Do not reduce WARMDOWN_RATIO below 0.4** — the cooldown phase is helping. Not worth changing until model size is right.
- **Do not try GeLU or SiLU** — relu² is fast on MPS and comparable in quality; activation swaps are low-priority.
