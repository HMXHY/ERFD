# ERFD: Fake News Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HMXHY/ERFD/blob/main/inference_demo.ipynb)

To ensure strict reproducibility, this repository provides a 1-click cloud execution via Google Colab and a fully containerized environment via Docker.

## 🚀 Quick Start (Google Colab)
Click the **"Open In Colab"** badge above. This will load `inference_demo.ipynb`, allowing you to sequentially execute our pipeline in your browser without any local setup. The notebook will automatically download the pre-trained checkpoints and test datasets.

## 🐳 Reproducibility via Docker
To freeze all package dependencies:
1. Build: `docker build -t erfd-fnd .`
2. Run: `docker run --gpus all -it erfd-fnd`

**Common Configurations:**
The following baseline parameters are applied across all datasets:
- `max_len`: `512`
- `batch_size`: `4`
- `hidden_dim`: `768`

**Dataset-Specific Configurations:**
The attention heads, frequency dimensions, and loss weights are dynamically adjusted based on the target dataset:

| Dataset | `freq_dim` | `attn_heads` | `loss_weight` |
| :--- | :---: | :---: | :---: |
| **PolitiFact** | 4 | 1 | 0.05 |
| **GossipCop** | 768 | 2 | 0.1 |
| **LUN** | 768 | 4 | 1.0 |
| **COVID** | 768 | 4 | 0.05 |
