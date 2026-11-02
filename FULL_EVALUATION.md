# Complete AML evaluation

Protocol `complete_forward_seeds_v1` evaluates every requested transaction once.
It appends missing forward seed edges to the sampled graph before message
passing, with their original attributes and parallel-edge group IDs. Reverse
neighbors stay as sampled. Labels are used only for metrics. Seed endpoints are
mapped using integer node IDs. This is a changed graph-evaluation protocol, not
an estimate of full-test metrics from the previous 75% subset, and not a causal
temporal evaluation protocol.

## Existing checkpoints

Run from the project root in the existing environment. Choose the allocated
device explicitly. For runs saved by `train_multi_gpu.py`:

```bash
python -u reevaluate.py \
  --run_dir checkpoints/S2_single_gpu_seed42_20261027 \
  --device cuda:2 --batch_size 1024 --num_neighs 20 20 --eval_seed 42
```

`--run_dir` reads `run_config.json` and all immediate `epoch_*.tar` files.
Model size and original feature switches come from the run configuration.
`--data_config data_config.json` can override paths when a run has moved.
Use the same evaluation batch size, neighbor fanouts, and evaluation seed for
all comparisons. Validation uses `eval_seed`; test uses `eval_seed + 1`.

For legacy B2/S1 runs without `run_config.json`, specify the original feature
and architecture flags and checkpoint files explicitly. Model dimensions come
from the current `model_settings.json`, which must match the original run:

```bash
python -u reevaluate.py --checkpoints checkpoints/B2/epoch_28.tar \
  --data Small_HI --model gin --task edge_class \
  --emlps --reverse_mp --ego --flatten_edges --edge_agg_type gin \
  --aml_semantic_features --aml_categorical_features \
  --device cuda:2 --batch_size 1024 --num_neighs 20 20 --seed 42 --eval_seed 42
```

For S1 add `--aml_dual_endpoint_update`. For S2 add
`--aml_conditioned_edge_agg`. For S3 add both. Architecture/feature checkpoint
metadata is checked when present, and state dict loading is strict. Checkpoints
without feature metadata require manual verification of the original flags.

Every candidate gets a complete-validation pass and a newly selected threshold.
Highest validation F1 chooses the checkpoint; ties retain the first candidate.
Only that checkpoint is evaluated on test, using its validation threshold.
Selection is limited to saved checkpoints, not all historical training epochs.
Supply one file with `--checkpoints` to reevaluate only a fixed checkpoint.

## Outputs and training

Results go into a fresh timestamped `evaluations/complete_*` directory, or a new
`--output_dir`. Existing output directories are rejected. Original checkpoints,
training logs, and stored thresholds are never overwritten.

- `evaluation_config.json`: resolved arguments, protocol and candidate paths.
- `validation_candidates.json`: candidate validation metrics and checkpoint hashes.
- `results.json`: validation-selected checkpoint, threshold and final test metrics.
- `validation_predictions.npz`, `test_predictions.npz`: sorted IDs, logits, labels
  and optional group metadata; the validation file belongs to the chosen model.
- `evaluation.log`: coverage counts, metrics and selected checkpoint.

Coverage checks compare exact ID sets and reject duplicates, rather than merely
comparing counts. The ID hash is included in metrics to help compare runs on the
same dataset; predictions also retain labels for checking dataset consistency.

AML validation/test calls in `train_util.py` and `distributed_training.py` now
use complete seeds. `inference.py` recalibrates on complete validation first.
The training sampler/loss and existing model weights are unchanged. Consequently
these results assess the old trained weights under the new evaluation protocol;
they do not claim to repair incomplete seed supervision in earlier training.
