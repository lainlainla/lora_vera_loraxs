# Comparison and Evaluation Experiment Code

This directory contains the concrete code layer for the LoRA / VeRA / LoRA-XS / TinyLoRA blueprint.

## What is included

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

```powershell
python ".\comparsion and evaluation experinment\experiments\smoke_test.py"
python ".\comparsion and evaluation experinment\experiments\list_settings.py" --kind all
python ".\comparsion and evaluation experinment\experiments\projection_formula_check.py"
python ".\comparsion and evaluation experinment\experiments\parameter_tradeoff.py" --setting paper-style
python ".\comparsion and evaluation experinment\experiments\toy_convergence.py"
python ".\comparsion and evaluation experinment\experiments\tinylora_tying_sweep.py"
```

## Notes

The code is intentionally independent of GLUE downloads and RL frameworks. It answers the current design questions first:

- how to inspect method-level settings independently from experiment-level settings;
- whether LoRA-XS gains are confounded by adapting more modules;
- when `A(A^T G B)B^T` is a valid projection;
- how trainable parameters scale with `layers * modules`;
- how fixed random projection differs from making the random projection trainable.
- whether reducing TinyLoRA trainable parameters through stronger tying causes a sharp final-loss change.
