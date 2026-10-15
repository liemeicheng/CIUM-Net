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
- PyTorch 1.13.1 or later and a matching torchvision version.
- CUDA-enabled GPU recommended for training.

### Step 1: Clone the Repository

```bash
git clone https://github.com/liemeicheng/CIUM-Net.git
cd CIUM-Net
```

### Step 2: Create Virtual Environment (Recommended)

```bash
conda create -n cium-net python=3.9
conda activate cium-net
```

### Step 3: Install Dependencies

Install the appropriate PyTorch build for your machine, then:

```bash
python -m pip install -r requirements.txt
```

Core dependencies: PyTorch, NumPy, Pillow and OpenCV. Full evaluation additionally
uses torchvision, LPIPS, SciPy, scikit-image, image-quality and LIBSVM.
LPIPS may download AlexNet evaluator weights on first use. The included
`niqe_pris_params.npz` file contains the reference statistics required by NIQE.

---

## 📁 Dataset Preparation

### Directory Structure

Keep the official dataset splits:

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

Paired low-/normal-light images are required for training. Validation directories
are optional and must be supplied together. No automatic split is performed.

### Supported Datasets

- LOLv1: 485 training pairs and 15 test pairs (`--dataset lolv1`).
- LOLv2-Real: 689 training pairs and 100 test pairs (`--dataset lolv2-real`).
- LOLv2-Synthetic: 900 training pairs and 100 test pairs (`--dataset lolv2-synthetic`).
- DICM, LIME, MEF, NPE and VV: unpaired inference and evaluation.

### Data Pairing Strategies

- LOLv1 and LOLv2-Synthetic: matching filename stems.
- LOLv2-Real: corresponding `low...` and `normal...` filename prefixes.

Missing or ambiguous image pairs raise an error. The training `--dataset` argument
selects the pairing convention.

---

## Train/ Test

### Default train run (lowlight_train.py)

```bash
python lowlight_train.py --dataset lolv1 --train-low datasets/LOLdataset/our485/low --train-normal datasets/LOLdataset/our485/high --val-low datasets/LOLdataset/eval15/low --val-normal datasets/LOLdataset/eval15/high --output runs/lolv1 --device cuda
```

Defaults in `loss_weights.py`: 1,000 epochs, batch size 8, 256×256 random crops,
independent horizontal/vertical flips with probability 0.5, and Adam with
betas (0.9, 0.999), epsilon 1e-8 and zero weight decay.
The objective is RGB loss + HVI loss + 0.005 × IECC; each restoration domain uses
L1 + 0.5 × (1 − SSIM) + 50 × edge MSE.

For LOLv2, change `--dataset` and the image directories. Resume using the same
command with `--resume runs/lolv1/last.pt`.

---

### Default test run (lowlight_test.py)

```bash
python lowlight_test.py --checkpoint runs/lolv1/best_psnr.pt --input datasets/LOLdataset/eval15/low --output results/lolv1 --device cuda
```

The output is saved as PNG at the original image size. No GT-mean brightness
correction is applied. CIUM-Net enhancement checkpoints are not bundled.

To compute PSNR, SSIM and AlexNet LPIPS:

```bash
python metrics.py --predictions results/lolv1 --targets datasets/LOLdataset/eval15/high --output results/lolv1_metrics.json --device cuda
```

For LOLv2-Real add `--pairing lolv2-real`. For an unpaired output folder, use
`python metrics.py --dataset DICM=results/DICM` to compute NIQE and BRISQUE.

---

### Training stages

1. Linear learning-rate warm-up over the first three epochs to 1e-4.
2. Cosine learning-rate decay to 1e-7 over the remaining epochs.

The same RGB/HVI restoration objective and IECC regularizer are used throughout.

---

## 📜 License

---

## 📧 Contact

GitHub: [liemeicheng](https://github.com/liemeicheng)

---

## 🔄 Updates

- Initial CIUM-Net source release with training, inference and evaluation code.

---

## 📌 Citation

```bibtex
@unpublished{li_cium_net,
  title = {CIUM-Net: Contrast-Conditioned Interaction-Unit Update Modulation for Low-Light Image Enhancement},
  author = {Li, Meicheng and Zhang, Guangjian and Wang, Yong and Gong, Maoguo and Dai, Jiahui},
  note = {Manuscript}
}
```
