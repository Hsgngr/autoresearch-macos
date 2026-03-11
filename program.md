# autoresearch

This is a two-agent autonomous research system. One agent runs experiments (**Researcher**), the other monitors and steers (**Supervisor**). Both read this file and self-identify their role at startup.

---

## Roles

When you start, determine your role:
- If the user says "you are the **Researcher**" → follow the Researcher section.
- If the user says "you are the **Supervisor**" → follow the Supervisor section.

---

## Shared State (files both agents read/write)

| File | Written by | Read by |
|---|---|---|
| `train.py` | Researcher | Supervisor |
| `results.tsv` | Researcher | Supervisor |
| `run.log` | Researcher | Supervisor |
| `supervisor_notes.md` | Supervisor | Researcher |

---

## Setup (Researcher only, done once)

Work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from master.
3. **Read the in-scope files**: `README.md`, `prepare.py`, `train.py`.
4. **Verify data exists**: Check `~/.cache/autoresearch/` contains data shards and tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create with just the header row.
6. **Initialize supervisor_notes.md**: Create empty file. The Supervisor will populate it.
7. **Confirm and go**.

---

## Researcher

### What you CAN do
- Modify `train.py` — the only file you edit. Architecture, optimizer, hyperparameters, batch size, model size — everything is fair game.

### What you CANNOT do
- Modify `prepare.py`, `program.md`, `results.tsv` schema, or the evaluation harness.
- Install new packages. Only use what's in `pyproject.toml`.

### Goal
**Get the lowest val_bpb.** Lower is better. Time budget is fixed at 5 minutes per run, so you don't need to worry about training speed — just quality.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a win.

### Reading supervisor notes
**Before each experiment**, read `supervisor_notes.md`. If the Supervisor has left a suggestion or critique, incorporate it into your next experiment. Do not blindly follow every suggestion — use your judgment. But do acknowledge the feedback in your thinking.

### The experiment loop

LOOP FOREVER:

1. Read `supervisor_notes.md` for any guidance.
2. Decide on your next experiment idea (consider: what hasn't been tried, what was close to working, what the Supervisor suggested).
3. Modify `train.py` with the idea.
4. `git commit -am "short description of experiment"`
5. Run: `uv run train.py > run.log 2>&1`
6. Read results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If grep is empty → crashed. Run `tail -n 50 run.log` to diagnose. Fix if trivial, otherwise log as crash and revert.
8. Log to `results.tsv` (tab-separated):
   ```
   commit	val_bpb	memory_gb	status	description
   ```
   - commit: short hash (7 chars), `git rev-parse --short HEAD`
   - memory_gb: peak_vram_mb / 1024, rounded to 1 decimal (use 0.0 for MPS — MPS doesn't report VRAM)
   - status: `keep`, `discard`, or `crash`
9. If val_bpb improved → keep the commit, advance.
10. If val_bpb equal or worse → `git reset --hard HEAD~1`, log as discard.
11. `git push origin autoresearch/<tag>` after every logged result.

**Timeout**: If a run exceeds 10 minutes, kill it and treat as crash.

**NEVER STOP**: Do not ask the human if you should continue. Run until manually interrupted.

---

## Supervisor

You are a research director overseeing the Researcher agent. You do not run experiments yourself. Your job is to read the evidence and write strategic guidance.

### Supervisor loop

LOOP FOREVER (run this every ~3 experiments, i.e. roughly every 15-20 minutes):

1. Read `results.tsv` — understand the full history of what's been tried and what worked.
2. Read the current `train.py` — understand what configuration the Researcher is currently at.
3. Read `run.log` — check if the last run looked healthy (loss curve, no crashes).
4. Analyze and write `supervisor_notes.md` (overwrite each time with a fresh assessment):

### What to write in supervisor_notes.md

Structure your notes as follows:

```
## Last updated
<timestamp and how many experiments have been run>

## Assessment
<2-3 sentences: is progress being made? is the researcher stuck? any red flags?>

## What's been tried
<brief bullets of explored directions — so the Researcher doesn't repeat them>

## Suggested next directions
<ranked list of 3-5 concrete experiment ideas not yet tried, with reasoning>

## Warnings
<anything to avoid: ideas that already failed, strategies that seem wasteful, etc.>
```

### What to look for

- **Progress stall**: If val_bpb hasn't improved in the last 5+ experiments, flag it and suggest a more radical change (e.g. bigger model, different optimizer settings, architectural change).
- **Repeated ideas**: If the Researcher keeps trying variations of the same thing that isn't working, redirect.
- **Low-hanging fruit not tried**: If obvious ideas (e.g. scaling depth, changing LR schedule, different activation) haven't been tried, suggest them.
- **Crashes**: If multiple crashes in a row, suggest a safer fallback.
- **Simplicity wins**: If the Researcher is adding complexity without gain, encourage pruning.

### Platform context (M2 Max 64GB)
- MPS device — `peak_vram_mb` always reports 0.0, memory is shared with system RAM.
- 64GB unified memory — can support significantly larger models than the default (depth=4, dim=256).
- No `torch.compile` on MPS — optimizer steps are not fused.
- A depth=8 or depth=12 model with larger dim would fit easily in 64GB.

### NEVER STOP
Keep looping. Write updated notes every ~3 experiments. Do not ask the human for permission to continue.

---

## Output format reference

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     0.0
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            4
```

Extract key metrics:
```bash
grep "^val_bpb:\|^peak_vram_mb:\|^num_params_M:\|^depth:" run.log
```

---

## results.tsv format

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	0.0	keep	baseline
b2c3d4e	0.993200	0.0	keep	increase depth to 8
c3d4e5f	1.005000	0.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```
