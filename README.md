# CSPF-Net Text Prototype

This repository now includes a lightweight text-side implementation aligned with the proposal:

- `cspf_text/features/prob_features.py`: GPT-2 style log-likelihood features
- `cspf_text/features/style_features.py`: NLTK-first stylometric features
- `cspf_text/features/cohesion_features.py`: TOCSIN-inspired random-deletion cohesion features
- `cspf_text/modeling/stacking_model.py`: `Scikit-learn` stacking baseline
- `cspf_text/modeling/mlp_model.py`: simple `PyTorch` MLP classifier
- `cspf_text/data.py`: Hugging Face dataset loading helpers for RAID / ASAP2
- `cspf_text/pipeline.py`: end-to-end feature fusion and document aggregation

## Quick start

```python
from cspf_text import CSPFTextPipeline, load_text_dataset

bundle = load_text_dataset("raid", split="train", sample_size=256)
pipeline = CSPFTextPipeline()
pipeline.fit(bundle.texts, bundle.labels)

result = pipeline.predict_document("This is a short test document. It has two sentences.")
print(result.document_probability, result.ai_contribution_ratio)
```

## Train from CLI

```bash
.venv/bin/python train_text_detector.py \
  --dataset raid \
  --train-split train \
  --eval-split test \
  --text-column generation \
  --label-column model \
  --sample-size 2000 \
  --eval-sample-size 500 \
  --model-type stacking \
  --prob-model-name gpt2 \
  --device cuda \
  --prob-batch-size 16 \
  --hf-cache-dir /kaggle/working/hf_cache \
  --output-dir artifacts/raid_stacking
```

For `RAID`, the common field mapping is:

- text: `generation`
- label: `model` where `human` maps to `0` and model names map to `1`

Run `.venv/bin/python check_env.py` to verify the environment before moving to Kaggle.

If Kaggle internet is disabled, pre-download the dataset/model into `--hf-cache-dir` and add `--local-files-only`.

For GPU runs on CUDA machines such as RunPod, pass `--device cuda`. If you use `--model-type mlp`, both the GPT-style feature extractor and the MLP will move to that device.

## Notes

- `ASAP2` on Hugging Face may require access approval depending on the dataset mirror you use.
- The TOCSIN part here follows the paper's high-level idea rather than reproducing the full original codebase.
- The current repo does not have the required Python dependencies installed yet; see `requirements.txt`.
