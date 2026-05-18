# Comparison and Evaluation Experiment Code

This directory contains the concrete code layer for the LoRA / VeRA / LoRA-XS / TinyLoRA blueprint.

## What is included

- `../colab_glue_experiment.ipynb`: Colab-first notebook entrypoint using public GLUE datasets.
- `../colab_run_glue.py`: Colab CLI entrypoint using the same real GLUE runner.
- `models/lora.py`: LoRA adapter wrapper for `nn.Linear`.
- `models/vera.py`: VeRA adapter wrapper for `nn.Linear`.
- `models/loraxs.py`: LoRA-XS adapter wrapper for `nn.Linear`.
- `models/tinylora.py`: TinyLoRA adapter wrapper for `nn.Linear`.
- `models/common.py` and `models/basis.py`: shared adapter base class and basis builders.
- `models/adapters.py`: compatibility exports for all adapter wrappers.
- `models/inject.py`: helper to replace target linear modules in a Hugging Face/PyTorch model.
- `models/subspace.py`: projection and gradient-energy utilities.
- `model_settings/*.py`: method-level settings, one file per adapter family.
- `experiment_settings/*.py`: experiment-level settings composed from model settings.
- `experiments/parameter_tradeoff.py`: trainable-parameter accounting for module/layer trade-offs.
- `experiments/list_settings.py`: prints model-level and experiment-level settings.
- `experiments/projection_formula_check.py`: verifies the projection formula condition.
- `experiments/toy_convergence.py`: compares fixed random TinyLoRA projection vs trainable random projection.
- `experiments/tinylora_tying_sweep.py`: compares zero update, full-model tying, per-layer tying, no tying, and direct LoRA-XS R.
- `experiments/smoke_test.py`: local sanity test without downloading datasets.

## Quick commands

Colab entrypoint:

```text
colab_glue_experiment.ipynb
```

Colab CLI entrypoint:

```bash
!python colab_run_glue.py --task sst2 --base-model roberta-base --model-setting lora_qv_r8
```

TinyLoRA same-target tying sweep:

```bash
!python colab_run_glue.py --task sst2 --base-model roberta-base --model-setting tinylora_qv_full_tie_u1_r2 --max-train-samples -1 --max-eval-samples -1 --epochs 3
!python colab_run_glue.py --task sst2 --base-model roberta-base --model-setting tinylora_qv_per_layer_tie_u1_r2 --max-train-samples -1 --max-eval-samples -1 --epochs 3
!python colab_run_glue.py --task sst2 --base-model roberta-base --model-setting tinylora_qv_no_tie_u1_r2 --max-train-samples -1 --max-eval-samples -1 --epochs 3
```

TinyLoRA paper-style coverage:

```bash
!python colab_run_glue.py --task sst2 --base-model roberta-base --model-setting tinylora_qvo_fc1_full_tie_u1_r2 --max-train-samples -1 --max-eval-samples -1 --epochs 3
!python colab_run_glue.py --task sst2 --base-model roberta-base --model-setting tinylora_qvo_fc1_per_layer_tie_u1_r2 --max-train-samples -1 --max-eval-samples -1 --epochs 3
!python colab_run_glue.py --task sst2 --base-model roberta-base --model-setting tinylora_qvo_fc1_no_tie_u1_r2 --max-train-samples -1 --max-eval-samples -1 --epochs 3
```

Use full public splits instead of quick subsets:

```bash
!python colab_run_glue.py --task sst2 --model-setting loraxs_qv_r8 --max-train-samples -1 --max-eval-samples -1
```

Local development utilities:

```powershell
python ".\comparsion and evaluation experinment\experiments\list_settings.py" --kind all
python ".\comparsion and evaluation experinment\experiments\projection_formula_check.py"
python ".\comparsion and evaluation experinment\experiments\parameter_tradeoff.py" --setting paper-style
python ".\comparsion and evaluation experinment\experiments\toy_convergence.py"
python ".\comparsion and evaluation experinment\experiments\tinylora_tying_sweep.py"
```

## Notes

The notebook uses real public GLUE data through Hugging Face `datasets`.
The small `smoke_test.py` file is development-only and is not the final experiment interface.

The code answers the current design questions first:

- how to inspect method-level settings independently from experiment-level settings;
- whether LoRA-XS gains are confounded by adapting more modules;
- when `A(A^T G B)B^T` is a valid projection;
- how trainable parameters scale with `layers * modules`;
- how fixed random projection differs from making the random projection trainable.
- whether reducing TinyLoRA trainable parameters through stronger tying causes a sharp final-loss change.
