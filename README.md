# CIUM-Net

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%2B-red)

CIUM-Net is a PyTorch framework for low-light image enhancement in HVI color space.
It combines bidirectional chromaticity–intensity update units (BiCIU), local
contrast-conditioned update modulation (LCUM), and training-only intensity-error-aware
chromatic consistency regularization (IECC) to separate complete interaction-unit
update generation from update injection.

> ## 📄 Paper

## 🔧 Installation

### Requirements

- Python 3.9 or later.
- PyTorch 1.13.1 or later; CUDA is optional for inference and recommended for training.
- A matching torchvision version for AlexNet LPIPS evaluation.

The manuscript environment uses PyTorch 1.13.1 and an NVIDIA RTX A6000. Install the
appropriate PyTorch/torchvision build for your machine before installing the remaining
dependencies. Known version pairs are torch 1.13.1 / torchvision 0.14.1 and
torch 2.0.1 / torchvision 0.15.2.

### Set up the project

Clone the repository and enter the project directory:

```bash
git clone https://github.com/liemeicheng/CIUM-Net.git
cd CIUM-Net
```

```bash
conda create -n cium-net python=3.9
conda activate cium-net
python -m pip install -r requirements.txt
```

LPIPS uses pretrained AlexNet features and may download its evaluator weights on first
use. `niqe_pris_params.npz` contains the reference statistics required by NIQE; it is
not a CIUM-Net checkpoint.

## 📁 Project Structure

```text
CIUM-Net/
├── model.py                  # HVI, BiCIU, LCUM and the CIUM-Net backbone
├── Myloss.py                 # IECC, L1/SSIM/edge restoration objective
├── loss_weights.py           # Paper loss weights and training settings
├── dataloader.py             # Paired images, random crops and flips
├── lowlight_train.py         # Training, validation and checkpoint resume
├── lowlight_test.py          # Image enhancement from a checkpoint
├── metrics.py                # Paired and unpaired evaluation commands
├── metrics_core.py           # PSNR, SSIM, LPIPS, NIQE and BRISQUE
├── niqe_pris_params.npz       # Required NIQE reference statistics
├── model_stats.py            # Parameter counting
├── utils.py                  # Device, random seed, learning rate and checkpoints
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
├── .gitignore
└── README.md
```

## 📁 Dataset Preparation

Keep the official training/test splits and store datasets outside the source files:

```text
datasets/
├── LOLdataset/
│   ├── our485/{low,high}/
│   └── eval15/{low,high}/
├── LOLv2/
│   ├── Real_captured/{Train,Test}/{Low,Normal}/
│   └── Synthetic/{Train,Test}/{Low,Normal}/
└── unpaired/
    ├── DICM/
    ├── LIME/
    ├── MEF/
    ├── NPE/
    └── VV/
```

| Dataset | Training pairs | Test pairs | `--dataset` |
| --- | ---: | ---: | --- |
| LOLv1 | 485 | 15 | `lolv1` |
| LOLv2-Real | 689 | 100 | `lolv2-real` |
| LOLv2-Synthetic | 900 | 100 | `lolv2-synthetic` |

Pairs are matched by filename stem. For LOLv2-Real, `low...` and `normal...` prefixes
are matched explicitly. Missing or ambiguous pairs produce an error. Training requires
paired low-/normal-light images; unpaired folders are used only for inference/evaluation.

## 🚀 Train / Test

### Training

Run from the project root:

```bash
python lowlight_train.py --dataset lolv1 --train-low datasets/LOLdataset/our485/low --train-normal datasets/LOLdataset/our485/high --val-low datasets/LOLdataset/eval15/low --val-normal datasets/LOLdataset/eval15/high --output runs/lolv1 --device cuda
```

For LOLv2, select `lolv2-real` or `lolv2-synthetic` and pass the corresponding directories.
Training settings are centralized in `loss_weights.py`:

| Setting | Value |
| --- | --- |
| Epochs / batch size / patch size | 1,000 / 8 / 256 × 256 |
| Optimizer | Adam, betas (0.9, 0.999), epsilon 1e-8, weight decay 0 |
| Learning rate | 1e-4, three-epoch linear warm-up, cosine decay to 1e-7 |
| Augmentation | Independent horizontal and vertical flips, each with probability 0.5 |
| RGB and HVI loss | L1 + 0.5 × (1 − SSIM) + 50 × edge MSE |
| Total loss | RGB loss + HVI loss + 0.005 × IECC |
| IECC coefficients | lambda_mu = 1, lambda_var = 0.5, lambda_cov = 0.01 |

The seed is controlled by `--seed` (software default: 42). Validation is performed every
10 epochs by default; these two choices are implementation defaults, not values specified
by the manuscript. Choose validation directories appropriate to your evaluation protocol.
Run files contain the resolved settings, epoch history, `last.pt` and best-per-metric
checkpoints. To resume, use the same command with `--resume runs/lolv1/last.pt`.
`--workers 0` is the portable default; `--device cpu` selects CPU execution.

### Testing

```bash
python lowlight_test.py --checkpoint runs/lolv1/best_psnr.pt --input datasets/LOLdataset/eval15/low --output results/lolv1 --device cuda
```

Outputs are lossless PNG images. Inputs are padded to a multiple of eight when needed
and cropped back to the original dimensions. No additional brightness, saturation,
gamma adjustment or ground-truth-derived correction is applied.

### Paired evaluation

```bash
python metrics.py --predictions results/lolv1 --targets datasets/LOLdataset/eval15/high --output results/lolv1_metrics.json --device cuda
```

For LOLv2-Real, add `--pairing lolv2-real`. Metrics are RGB PSNR over [0,255], channel-mean
SSIM using an 11×11 Gaussian window (sigma 1.5), and AlexNet LPIPS. Scores are computed
per image and then averaged. There is no GT-mean correction or reference-driven resizing.
SSIM uses valid window centers; no additional border crop is applied. `--skip-lpips`
explicitly requests partial PSNR/SSIM evaluation.

### Unpaired evaluation

Enhance each dataset with `lowlight_test.py`, then evaluate the resulting folders:

```bash
python metrics.py --dataset DICM=results/DICM --dataset LIME=results/LIME --dataset MEF=results/MEF --dataset NPE=results/NPE --dataset VV=results/VV --output results/unpaired_metrics.json
```

NIQE and BRISQUE are averaged per dataset, then equally across datasets. The manuscript
uses 69 DICM, 10 LIME, 17 MEF, 8 NPE and 24 VV images. The exact selected subset filenames
are not distributed here. `--niqe-only` requests partial evaluation without BRISQUE.
For unpaired command help, use `python metrics.py --dataset NAME=FOLDER --help`.

### Component ablations

| Configuration | Additional training arguments |
| --- | --- |
| BiCIU only | `--no-lcum --no-iecc` |
| BiCIU + IECC | `--no-lcum` |
| BiCIU + LCUM | `--no-iecc` |
| Full CIUM-Net | None |
| IECC E0–E4, Table 9 | `--iecc-variant E0` through `--iecc-variant E4` |

E0 retains the auxiliary chromatic L1 term; it differs from `--no-iecc`.
The E0–E4 settings disable LCUM. The model's `interaction_alpha3` and
`interaction_alpha4` forward arguments support complete-update enable/disable interventions.
This release contains the core method and these component ablations; it does not bundle
the separate Table 8 comparison models or the full mechanism-analysis plotting pipeline.

### Model parameters

```bash
python model_stats.py
```

## 📜 Acknowledgements

The method builds on HVI/CIDNet representations and restoration operators; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for implementation provenance.
The compact repository layout follows [RemDiNet](https://github.com/xenbaloch/RemDiNet).
Licensing and upstream attribution are described in the provenance notice.

## 📌 Citation

```bibtex
@unpublished{li_cium_net,
  title = {CIUM-Net: Contrast-Conditioned Interaction-Unit Update Modulation for Low-Light Image Enhancement},
  author = {Li, Meicheng and Zhang, Guangjian and Wang, Yong and Gong, Maoguo and Dai, Jiahui},
  note = {Manuscript}
}
```
