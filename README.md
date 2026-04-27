# CSPF-Net AI-generated Text Detection

This repository now includes a lightweight text-side implementation aligned with the proposal:

- `cspf_text/features/prob_features.py`: GPT-2 style log-likelihood features
- `cspf_text/features/style_features.py`: NLTK-first stylometric features
- `cspf_text/features/cohesion_features.py`: TOCSIN-inspired random-deletion cohesion features
- `cspf_text/modeling/stacking_model.py`: `Scikit-learn` stacking baseline
- `cspf_text/modeling/mlp_model.py`: simple `PyTorch` MLP classifier
- `cspf_text/data.py`: dataset loading helpers for local CSV/JSONL/Parquet files and Hugging Face datasets
- `cspf_text/pipeline.py`: end-to-end feature fusion and document aggregation

## Quick start

```python
from cspf_text import CSPFTextPipeline, load_text_dataset

bundle = load_text_dataset(
    "hc3_reborn",
    dataset_id="/path/to/hc3_human_ai_reborn.csv",
    split="train",
    text_column="text",
    label_column="label",
    sample_size=256,
)
pipeline = CSPFTextPipeline()
pipeline.fit(bundle.texts, bundle.labels)

result = pipeline.predict_document("This is a short test document. It has two sentences.")
print(result.document_probability, result.ai_contribution_ratio)
```

## Train from CLI

```bash
.venv/bin/python train_text_detector.py \
  --dataset hc3_reborn \
  --dataset-id /path/to/hc3_human_ai_reborn.csv \
  --train-split train \
  --eval-split test \
  --text-column text \
  --label-column label \
  --sample-size 2000 \
  --eval-sample-size 500 \
  --model-type stacking \
  --prob-model-name gpt2 \
  --device cuda \
  --prob-batch-size 16 \
  --hf-cache-dir /workspace/huggingface \
  --output-dir artifacts/hc3_reborn_stacking
```

For `HC3 Human AI Reborn`, use binary labels:

- `human` -> `0`
- `ai` -> `1`
- `reborn` -> `1`

The loader already maps `reborn` into the AI class.

If your local file does not already provide `train` / `test` splits, split it yourself first, or place it in a local folder like:

```text
data/hc3_reborn/
  train.csv
  test.csv
```

and pass:

```bash
--dataset-id data/hc3_reborn
```

Run `.venv/bin/python check_env.py` to verify the environment before moving to Kaggle.

If internet is disabled, prefer using a local dataset path and pre-download the model into `--hf-cache-dir`, then add `--local-files-only`.

If you do not pass `--hf-cache-dir`, the scripts default to `HF_HOME` when it is set, otherwise `/workspace/huggingface`.

For GPU runs on CUDA machines such as RunPod, pass `--device cuda`. If you use `--model-type mlp`, both the GPT-style feature extractor and the MLP will move to that device.

## Notes

- `ASAP2` on Hugging Face may require access approval depending on the dataset mirror you use.
- The recommended primary dataset for the current binary setup is `HC3 Human AI Reborn`, with `human=0` and `ai/reborn=1`.
- The TOCSIN part here follows the paper's high-level idea rather than reproducing the full original codebase.
- The current repo does not have the required Python dependencies installed yet; see `requirements.txt`.
