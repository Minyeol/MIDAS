# Training-Free Coverless Multi-Image Steganography with Access Control

## 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n MIDAS python=3.12.9 -y
conda activate MIDAS

# Install PyTorch with CUDA 12.1 support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

---

## 2. Pretrained Models

Download the IP-Adapter and encoder models and place them in the `./models/` directory.

- IP-Adapter:
  - https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-plus_sd15.bin
- Image Encoder:
  - https://huggingface.co/h94/IP-Adapter/blob/main/models/image_encoder/pytorch_model.bin

---

## 3. Dataset Preparation

Download the Stego260 dataset from:

- https://github.com/yujiwen/CRoSS

The expected directory structure is as follows:

```text
MIDAS/
├── dataset
│   └── steganodataset_v1
│       ├── data
│       └── config.yaml
├── metrics
│   ├── metrics_degrad.py
│   ├── metrics_multi.py
│   └── metrics_single.py
├── models
│   ├── config.json
│   ├── ip-adapter-plus_sd15.bin
│   └── pytorch_model.bin
├── results
├── cover_generation.py
├── MIDAS_degrad.py
├── MIDAS_multi.py
├── MIDAS_single.py
├── README.md
├── requirements.txt
├── scheduling.py
└── ...
```

---

## 4. Execution Guide

### Note on File Paths

Before running the scripts, ensure that the following arguments match your local environment:

- `--yaml_path`:
  Path to your dataset configuration file  
  (default: `./dataset/steganodataset_v1/config.yaml`)

- `--save_path`:
  Directory where the generated results will be stored.  
  Example:
  `./results/MIDAS/stego260/2/seed9000/`

---

### A. Single Image Hiding (1 Image)

#### MIDAS Mode

```bash
python ./MIDAS_single.py \
    --pw 9000 \
    --mode 'MIDAS' \
    --key_type 'random_basis' \
    --personal_key_strength 0.5 \
    --private_strength 0.4 \
    --edit_strength 0.7 \
    --ref_strength 0.4 \
    --alpha 0.95
```

#### DiffStega* Mode (Baseline)

```bash
python ./MIDAS_single.py \
    --pw 9000 \
    --mode 'DiffStega*' \
    --key_type 'noise_flip' \
    --personal_key_strength 0.05 \
    --private_strength 0.6 \
    --edit_strength 0.6
```

---

### B. Multi-Image Hiding (2 Images)

#### MIDAS Mode

```bash
python ./MIDAS_multi.py \
    --pw 9000 \
    --num_secret_image 2 \
    --mode 'MIDAS' \
    --key_type 'random_basis' \
    --personal_key_strength 0.4 \
    --key_strength 0.5 \
    --private_strength 0.4 \
    --edit_strength 0.7 \
    --ref_strength 0.4 \
    --alpha 0.95
```

#### DiffStega* Mode (Baseline)

```bash
python ./MIDAS_multi.py \
    --pw 9000 \
    --num_secret_image 2 \
    --mode 'DiffStega*' \
    --key_type 'noise_flip' \
    --personal_key_strength 0.05 \
    --key_strength 0.05 \
    --private_strength 0.6 \
    --edit_strength 0.6
```

> **Tip:**  
> You can evaluate performance across various scales by varying
> `--num_secret_image` to `2`, `4`, or `8`.

---

### C. Degradation Test

> **Note:**  
> The degradation test is currently implemented specifically for the
> 2-image hiding scenario. Ensure that you have completed the experiments
> in Section B before running this script.

#### MIDAS Mode

```bash
python ./MIDAS_degrad.py \
    --pw 9000 \
    --mode 'MIDAS' \
    --key_type 'random_basis' \
    --personal_key_strength 0.4 \
    --key_strength 0.5 \
    --private_strength 0.5 \
    --edit_strength 0.7 \
    --ref_strength 0.4 \
    --alpha 0.95
```

#### DiffStega* Mode (Baseline)

```bash
python ./MIDAS_degrad.py \
    --pw 9000 \
    --mode 'DiffStega*' \
    --key_type 'noise_flip' \
    --personal_key_strength 0.05 \
    --key_strength 0.05 \
    --private_strength 0.7 \
    --edit_strength 0.6
```

---

## 5. Evaluation & Analysis

### A. Quantitative Evaluation

The scripts in the `./metrics` directory provide comprehensive quality
assessments.

### B. Steganalysis Preparation

Use `cover_generation.py` to generate original cover images for
comparative steganalysis.

---

## Acknowledgement

Part of the code is borrowed from DiffStega:

- https://github.com/evtricks/DiffStega
