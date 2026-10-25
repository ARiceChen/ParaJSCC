# ParaJSCC

Multimodal joint source-channel coding for the LMT-108 surface-material dataset. Each sample is
fitted independently with its own network weights and progressive code tensors.

## Requirements

Use Python 3.10 or newer. Install PyTorch with CUDA support for GPU training, then install the
remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Set the dataset path to the processed LMT-108 directory with `--dataset`. The default path is
`./data/LMT_108`. The loader uses the dataset Training and Testing
directories and creates a seeded, material-stratified 70:30 training and validation split.

## Training

Each selected sample is optimized independently. The defaults are 2,000 updates per sample, all
three modalities, 2:1 code-width compression, and SNR 10 dB. CUDA Graphs can accelerate the fixed
training request.

```bash
python train.py --dataset /path/to/LMT_108 \
  --save_dir ./runs/lmt108 --code_split val --max_train_samples 10 \
  --total_steps 2000 --train_progressive_prob 0 --cuda_graph
```

Use `--channel_compression 1` to disable code-width compression. Use `--channel_compression 2` to
halve the number of transmitted complex symbols. Training checkpoints and parameter packages are
saved under the selected run directory.
