## Last updated
2026-03-11, 2 experiments logged (baseline + depth=8)

## Assessment
⚠️ DEPTH=8 got val_bpb=1.7649 — WORSE than baseline (1.3796). Root cause: the bigger model runs ~4x slower (~5s/step vs ~1.3s), so in 5 minutes it only completed 69 steps and 4.5M tokens instead of ~240 steps. The model is severely undertrained. Scaling depth without addressing step budget is counterproductive. The priority now is to find a model size that fits more gradient steps within the 300s budget.

## What's been tried
- `baseline`: DEPTH=4, dim=256, 11.5M params → val_bpb=1.3796 (keep) — ~240 steps, 15.8M tokens
- `depth=8`: DEPTH=8, dim=512, 50M params → val_bpb=1.7649 (discard) — only 69 steps, 4.5M tokens — too slow

## Suggested next directions

1. **[HIGHEST PRIORITY] Revert to DEPTH=4 and increase DEVICE_BATCH_SIZE** — The baseline is actually a reasonable model. The bottleneck is tokens-per-run, not model capacity. Try `DEVICE_BATCH_SIZE=32` (doubles tokens per step, same step count → 2x tokens). This is the safest next move.

2. **Try DEPTH=6** — dim=384, HEAD_DIM=128 → 3 heads, ~25M params. Step time ~2-2.5s → ~120 steps, ~8M tokens. A middle ground — more capacity than depth=4 but more steps than depth=8.

3. **Reduce ASPECT_RATIO with DEPTH=8** — e.g., `ASPECT_RATIO=32` → dim=256 at depth=8, keeping dim small but more layers. Same param count as baseline but 8 layers. Step time stays near baseline.

4. **Try TOTAL_BATCH_SIZE=2^15 (32K)** — Half the current batch size means 2x more steps in the same time. More gradient updates may compensate for smaller batches.

5. **WINDOW_PATTERN tuning** — Once a good model size/speed tradeoff is found, try "SSSL" pattern.

## Warnings
- **DEPTH=8 with default ASPECT_RATIO is too slow** — only 69 steps in 5 min. Do not repeat without reducing dim or batch size.
- **Tokens per run matters as much as model size** — with a 5-min fixed budget, step speed is the primary constraint on MPS.
- **Do not try DEPTH=12** without fixing the step speed problem first.
- **MPS has no torch.compile** — no fusion benefit.


When Researcher check these notes, they should add their findings.

---
Researcher Findings:
