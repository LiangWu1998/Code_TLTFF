# TLTFF

This repository contains the code for the manuscript **Transform-Driven Low-Rank Tensor Functional Factorization for Multidimensional Data Recovery**.

## Authors

- **Liang Wu** (Southwest University)
- **Jianjun Wang** (Southwest University)
- **Deyu Meng** (Xi'an Jiaotong University)

## Submission

The manuscript has been submitted to *IEEE Transactions on Neural Networks and Learning Systems* (TNNLS).

## Color Image Completion Demo

[`demo_ColorImage_completion.py`](demo_ColorImage_completion.py) is the main script for the Boats color image completion demo. It requires a CUDA-capable GPU and CUDA-enabled PyTorch.

Set `data_path = "./data/"` in the script to load the included Boats data. Run the demo from the repository root.

```bash
python demo_ColorImage_completion.py
```

With the default training parameters, the script saves the recovered color image as `X_Out` in `results/sr5_Boats_CB1.mat`.
