 <div align="center">

# ParaJSCC
### A Parameterized Framework for Reusable Multimodal Joint Source-Channel Coding

**ACM Multimedia 2026 (ACM MM 2026) · Oral Presentation**

[![Conference](https://img.shields.io/badge/ACM%20MM-2026-blue.svg)](https://2026.acmmm.org/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2608.15066)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)

*Official PyTorch implementation of ParaJSCC.*

</div>

---

## 📖 Overview

**ParaJSCC** is a parameterized framework for reusable multimodal joint source-channel coding (JSCC). It provides a unified parameter-based representation for image, audio, and tactile modalities, supporting flexible multimodal transmission.

## 🛠️ Requirements

**Environment:** Python 3.10+ and PyTorch with CUDA support (recommended for GPU training).

Install PyTorch with the appropriate CUDA version, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## 📂 Dataset

ParaJSCC uses the processed **LMT-108** dataset.

Specify the dataset directory using `--dataset`. The default location is:

```text
./data/LMT_108
```

Ensure that the dataset is properly prepared before training.

## 🚀 Training

ParaJSCC optimizes each selected sample independently.

The default training configuration includes:

| Configuration | Default |
|:---|:---|
| Optimization Steps | 2,000 per sample |
| Modalities | Image, Audio, and Tactile |
| Channel SNR | 10 dB |
| Optimization Strategy | Per-sample Optimization |
| GPU Acceleration | CUDA Graphs (optional) |

### Quick Start

Run the following command to start training:

```bash
python train.py \
    --dataset /path/to/LMT_108 \
    --save_dir ./runs/lmt108 \
    --code_split val \
    --max_train_samples 10 \
    --total_steps 2000 \
    --train_progressive_prob 0 \
    --cuda_graph
```

**Notes:**
- Replace `/path/to/LMT_108` with your actual dataset path.
- `--max_train_samples 10` limits training to 10 selected samples.
- `--total_steps 2000` specifies the number of optimization steps per sample.
- `--cuda_graph` enables CUDA Graph acceleration for the fixed training request.

## 📝 Citation

If you find ParaJSCC useful in your research, please consider citing our paper:

**Kemi Chen et al.**, "ParaJSCC: A Parameterized Framework for Reusable Multimodal Joint Source-Channel Coding," *Proceedings of the 34th ACM International Conference on Multimedia (ACM MM)*, 2026.

Paper: https://doi.org/10.1145/3767308.3836537

## 📬 Contact

For questions or research discussions, please refer to the project's [GitHub repository](https://github.com/ARiceChen/ParaJSCC).

