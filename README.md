# ParaJSCC: A Parameterized Framework for Reusable Multimodal Joint Source-Channel Coding (ACM MM'2026)

## Requirements

Use Python 3.10 or newer. Install PyTorch with CUDA support for GPU training, then install the
remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Set the dataset path to the processed LMT-108 directory with `--dataset`. The default path is
`./data/LMT_108`. 

## Training

Each selected sample is optimized independently. The defaults are 2,000 updates per sample, all
three modalities, and SNR 10 dB. CUDA Graphs can accelerate the fixed
training request.

```bash
python train.py --dataset /path/to/LMT_108 \
  --save_dir ./runs/lmt108 --code_split val --max_train_samples 10 \
  --total_steps 2000 --train_progressive_prob 0 --cuda_graph
```


