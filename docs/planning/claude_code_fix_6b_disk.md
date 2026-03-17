# Fix kalavai_pythia_6b_experiment.py — Disk Space Crash

## The Problem

The script crashed at `torch.save(model.state_dict(), ckpt)` because the 6.9B model state dict is ~14GB and RunPod's disk ran out of space. The error:

```
RuntimeError: [enforce fail at inline_container.cc:664] . unexpected pos 1353023680 vs 1353023568
```

## The Fix

Three changes needed:

### 1. Remove all full checkpoint saves

The `save_specialist_checkpoint` function saves the entire 6.9B state dict (~14GB per checkpoint). With 3 specialists × 3 seeds = 9 checkpoints = 126GB. That won't fit.

**Replace `save_specialist_checkpoint` with a function that saves ONLY the unfrozen layer weights + the router.** The unfrozen layers for 6.9B with freeze=6 are layers 6-31 (26 out of 32 layers), which is still ~11GB. Even that's too much for 9 saves.

**Simplest fix: don't save checkpoints at all.** Keep specialists in GPU memory during each seed's run (train specialist A → keep in memory → train specialist B → keep in memory → train specialist C → keep in memory → fuse → evaluate → save results JSON → delete all models → next seed). The results JSON is tiny (<1KB). That's all we need for the paper.

Find every call to `save_specialist_checkpoint` or `torch.save` that saves model weights and either:
- Comment it out, or
- Replace with `print(f"Skipping checkpoint save for {domain} (disk space)")`

Do NOT remove `json.dump` calls or `git commit` calls — those save results, not weights.

### 2. Add explicit memory cleanup between seeds

After each seed completes (all 3 specialists trained, fused, evaluated, results saved):

```python
# After saving results JSON for this seed
del model_code, model_science, model_fiction, fused_model, router
torch.cuda.empty_cache()
import gc
gc.collect()
print(f"Memory cleared after seed {seed}")
```

This ensures the 6.9B models from seed=42 are freed before loading fresh copies for seed=137.

### 3. Add resume logic

The script should detect which steps already completed (by checking for results JSON files) and skip them:

```python
import os

def step_done(path):
    return os.path.exists(path)

# Before each major step:
result_path = f"results/pythia_6b/step6_fusion_seed{seed}.json"
if step_done(result_path):
    print(f"Seed {seed} already complete, skipping")
    continue
```

This way if the script crashes mid-run, restarting picks up where it left off without repeating finished work.

### 4. Clear HuggingFace cache between model loads

The HF cache stores full model downloads. Loading `pythia-6.9b` at different revisions can eat disk:

```python
import shutil

def clear_hf_cache():
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if "pythia-6.9b" in d:
                path = os.path.join(cache_dir, d)
                size = sum(f.stat().st_size for f in pathlib.Path(path).rglob('*') if f.is_file()) / 1e9
                print(f"Clearing cache: {d} ({size:.1f}GB)")
                shutil.rmtree(path)
```

Call this after the step10000 experiments finish and before loading step143000 for the maturity check.

## Verification After Fix

Before running the full experiment, do a quick sanity check:

```python
# Add this at the top of main():
import shutil
disk = shutil.disk_usage("/workspace")
print(f"Disk: {disk.free/1e9:.1f}GB free / {disk.total/1e9:.1f}GB total")
if disk.free < 20e9:
    print("WARNING: Less than 20GB free. Clear caches before proceeding.")
```

## Expected Memory Flow (per seed)

```
1. Load base model (~14GB GPU, ~14GB disk cache)        → disk: ~14GB used
2. Eval base model                                       → no disk change
3. Train code specialist (in GPU memory)                 → no disk change
4. Train science specialist (load fresh base → GPU)      → no disk change  
5. Train fiction specialist (load fresh base → GPU)      → no disk change
6. Fuse (all 3 specialists + router in GPU memory)       → no disk change
7. Evaluate fused model                                  → no disk change
8. Save results JSON                                     → ~1KB disk
9. Git commit + push                                     → ~1KB
10. Delete all models from GPU memory                    → GPU freed
11. Next seed: go to step 1
```

Peak GPU memory: ~28-35GB (base model + specialist + gradients + optimizer during training). A100 80GB handles this easily.

Peak disk usage: ~14GB (HF cache for one revision) + datasets (~5GB) + repo (~1GB) = ~20GB. Well within the 80-100GB volume.

## IMPORTANT: Keep the training config unchanged

Do NOT modify any of these while fixing the disk issue:
- model_id, revision, freeze_layers, learning_rate, max_steps
- batch_size, gradient_accumulation, precision, max_length
- domains, datasets, text_fn
- router architecture, router training steps
- evaluation method, held-out data splits
- random seeds

The ONLY changes should be: removing torch.save calls, adding memory cleanup, adding resume logic, adding disk space checks.

## After fixing, test with a dry run

Before running the full experiment, verify the fix works:

```bash
# Check disk space
df -h /workspace

# Run just the base eval (Phase 1, Step 1) to verify model loads without OOM
# Add a flag or modify to run only step 1, then check disk space again
```

If base eval completes and disk space is still >50GB free, proceed with the full run.
