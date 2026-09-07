<<<<<<< HEAD
# CSAF-Net: Gated Cross-Scale Attention Fusion for Wall Crack Detection

A deep learning system for binary wall-crack classification that fuses **DenseNet-121** (local texture detail) and a **Vision Transformer** (global structural context) through a novel **gated bidirectional cross-attention fusion module**. The gate learns, per input, how much to trust each backbone — improving recall on real cracks over simpler fusion strategies, which matters for a safety-relevant inspection task.

Built for SWE3004 (VIT), with a target of IEEE ICCCNT / INDICON (IEEE Access as a journal fallback).

## Why gated fusion?

CNNs (DenseNet-121) excel at local texture patterns — the fine, jagged lines that define a crack. Vision Transformers excel at global context — recognizing uniform wall regions vs. irregular ones. Simply concatenating their features treats both as equally reliable for every image. **CSAF-Net's gate learns to weight them adaptively per input**, and empirically shifts toward the CNN branch specifically on crack-positive images (see gate activation analysis below).

## Results

Five variants were trained and evaluated on the same SDNET2018 wall-crack split (4,000 images, 70/15/15) to isolate exactly what each architectural component contributes:

| Variant | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| CNN Only (DenseNet-121) | 75.33% | 73.48% | 79.46% | 68.33% |
| ViT Only | 82.83% | 82.09% | 85.82% | 78.67% |
| Concat Fusion (no attention) | 84.17% | 84.09% | 84.51% | 83.67% |
| Cross-Attention (no gate) | 85.83% | 85.22% | 89.09% | 81.67% |
| **CSAF-Net (full, gated)** | **86.00%** | **86.14%** | 85.29% | **87.00%** |

**Key finding:** compared to the best non-gated alternative (cross-attention without a gate), CSAF-Net reduces missed crack detections by **29%** (55 → 39 false negatives on the test set), at the cost of a small increase in false alarms — a favorable trade-off for a safety-critical inspection system, where a missed crack is worse than a false alarm.

The gate's mean activation also separates cleanly by class — averaging ~0.53 on crack-negative images vs. ~0.61 on crack-positive images — indicating the model has learned to lean on the CNN branch specifically when a crack is present, rather than applying a fixed fusion weight.

All five variants share the same frozen backbones and classification head; CSAF-Net's gate adds only ~0.2M trainable parameters over the ungated cross-attention baseline (1.22M vs. 1.02M), against a shared 94M-parameter backbone.

## Architecture

```
Input Image (224×224×3)
        │
   ┌────┴────┐
   │         │
DenseNet-121  ViT
(local tokens) (global tokens)
   │         │
   └────┬────┘
        │
Bidirectional Cross-Attention
 (CNN queries ViT, ViT queries CNN)
        │
   Learned Sigmoid Gate
 (per-token, per-channel, conditioned
  on both attended branches)
        │
   Gated Fusion
        │
 Classification Head
        │
   Crack / No Crack
```

## Repository structure

| File | Purpose |
|---|---|
| `csaf_module.py` | Core fusion modules: `GatedCrossAttentionFusion` (full novelty), `ConcatFusion` and `UngatedCrossAttentionFusion` (ablation baselines) |
| `csaf_net.py` | Full model assembly (`CSAFNet`, `SingleBackboneNet`) and `build_model()` factory for all 5 variants |
| `backbones.py` | DenseNet-121 and ViT backbone wrappers |
| `train.py` | Unified training script — trains any variant via `--variant` |
| `evaluate.py` | Test-set evaluation, confusion matrices, ROC curves |
| `predict_csaf.py` | Single-image inference + Canny/Hough crack analysis |
| `app.py` | Streamlit demo app — live prediction, training analytics, ablation comparison |
| `prepare_dataset.py` | Dataset preparation from SDNET2018 |
| `figures/` | Generated result figures (ablation comparison, confusion matrices, ROC curves, gate visualizations, training curves) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Prepare the dataset** (expects SDNET2018 wall subset):
```bash
python prepare_dataset.py
```

**Train a variant:**
```bash
python train.py --variant full_csaf --epochs 5
```
Variant options: `cnn_only`, `vit_only`, `concat`, `ungated_cross_attn`, `full_csaf`

**Evaluate:**
```bash
python evaluate.py --variant full_csaf
```

**Run the interactive demo:**
```bash
streamlit run app.py
```

**Single-image prediction:**
```bash
python predict_csaf.py --image path/to/image.jpg --variant full_csaf
```

## Dataset

[SDNET2018](https://digitalcommons.usu.edu/all_datasets/48/) — wall subset, 4,000 labeled images (crack / no-crack), split 70/15/15 for train/val/test.

## Status

Core architecture, all five ablation variants, and full evaluation pipeline (metrics, confusion matrices, ROC curves, gate activation analysis) are complete. Cross-domain generalization to other crack datasets is explicitly scoped as future work.
=======

>>>>>>> 0fc3d789dd1522b4309c10f1fd8e1ff25016bee2
