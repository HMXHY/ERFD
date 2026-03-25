# ERFD: Fake News Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HMXHY/ERFD/blob/main/inference_demo.ipynb)

To ensure strict reproducibility, this repository provides a 1-click cloud execution via Google Colab and a fully containerized environment via Docker.

## 🚀 Quick Start (Google Colab - Recommended for Reviewers)
Click the **"Open In Colab"** badge above. This will load `inference_demo.ipynb`, allowing you to sequentially execute our pipeline in your browser without any local setup.

## 🐳 Reproducibility via Docker
To freeze all package dependencies:
1. Build: `docker build -t erfd-fnd .`
2. Run: `docker run --gpus all -it erfd-fnd`

## ⚙️ Configurations
- `dataset_name`: `politifact`
- `max_len`: `512`
- `batch_size`: `4`
- `epochs`: `5`
- `hidden_dim`: `768`
- `freq_dim`: `4`
- `attn_heads`: `1`
- `loss_weight`: `0.05`
