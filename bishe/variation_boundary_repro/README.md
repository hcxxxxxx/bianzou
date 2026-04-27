# SA-CNFolk Reproduction

This directory is a self-contained reproduction scaffold for the paper's
Chinese folk song variation boundary detection experiment. It only depends on:

- `../../paper.pdf`
- `../../model_st.py` as architectural reference
- `../../songs_dataset.json`
- `../../wavs/*.wav`

The implementation follows the paper/reference hints:

- audio -> 128-bin log Mel spectrogram (`sr=44100`, `hop_length=512`, `fmax=8000`)
- grouped train/val/test split at 8:1:1 by song prefix, so different versions of
  the same folk song never cross splits
- start and end of a song are not used as variation boundaries
- 1-second feature aggregation by default
- CNN embedding + Bi-LSTM + linear boundary classifier
- `Adam(lr=1e-3)`, `BCEWithLogitsLoss`, `ReduceLROnPlateau(mode="max")`
- gradient accumulation with `accum_steps=2`
- model selection by validation `HR3F`

## Quick Start

Run from the repository root:

```bash
cd /Users/hcx/Desktop/毕业论文/bianzou
python3 bishe/variation_boundary_repro/prepare_splits.py
python3 bishe/variation_boundary_repro/extract_mels.py
python3 bishe/variation_boundary_repro/train.py
```

After training, sweep post-processing choices without retraining:

```bash
python3 bishe/variation_boundary_repro/evaluate_checkpoint.py --split val
python3 bishe/variation_boundary_repro/evaluate_checkpoint.py --split test
```

Artifacts are written under:

```text
bishe/variation_boundary_repro/artifacts/
```

Important outputs:

- `artifacts/splits.json`: grouped 8:1:1 split
- `artifacts/mels/*.npy`: extracted log Mel features
- `artifacts/checkpoints/best.pt`: best validation checkpoint
- `artifacts/runs/<run_name>/metrics.json`: training history and final test scores
- `artifacts/runs/<run_name>/test_predictions.json`: predicted boundaries

## Common Options

```bash
python3 bishe/variation_boundary_repro/train.py \
  --batch-size 4 \
  --epochs 100 \
  --fold-seconds 1.0 \
  --embed-dim 24 \
  --hidden-size 128 \
  --lstm-layers 2 \
  --label-tolerance 3.0 \
  --threshold 0.0001
```

For closer alignment with the author's feature-saving snippet, disable per-song
Mel normalization:

```bash
python3 bishe/variation_boundary_repro/train.py --no-normalize-mel --tune-threshold --max-predictions-per-song 4
```

For the author's post-processing setting, do not pass
`--max-predictions-per-song` or `--min-predictions-per-song`; use pure local
maxima filtering:

```bash
python3 bishe/variation_boundary_repro/train.py --threshold 0.0001
python3 bishe/variation_boundary_repro/train.py --threshold 0.001
```

Batch-run focused ablations with `batch_size=1` and `filter_size=9`:

```bash
python3 bishe/variation_boundary_repro/run_experiment_matrix.py --preset focused --device cuda
```

The paper reports the best hyperparameter set as:

```text
w=1.0s, sigma=24, hidden=128, LSTM layers=2
```

## Notes

`mir_eval` is not required. The boundary matching metric here is implemented
directly as one-to-one boundary matching within the requested tolerance window,
which corresponds to the HR.5 and HR3 boundary metrics described in the paper.
