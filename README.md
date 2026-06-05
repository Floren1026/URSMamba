<div align="center">
<h1>URSMamba </h1>
<h3>URSMamba: Universal remote sensing image steganography using state space model</h3>

Paper: (https://www.sciencedirect.com/science/article/abs/pii/S0893608026005939)
</div>

## Getting Started

### Installation(followed by VMamba)

**Step 1: Clone the URSMamba repository:**

To get started, first clone the URSMamba repository and navigate to the project directory:

```bash
git clone https://github.com/Floren1026/URSMamba.git
cd URSMamba
```

**Step 2: Environment Setup:**

Use the following commands to set up your environment:
BTW, we recommend using the pytorch>=2.0, cuda>=11.8. But lower version of pytorch and CUDA are also supported.

***Create and activate a new conda environment***

```bash
conda create -n ursmamba
conda activate ursmamba
```

***Install Dependencies***

```bash
cd kernels/selective_scan && pip install .
```


## Dataset
- In this work, we use the multispectral remote sensing datasets collected by SpaceNet. The detailed download operation can refer to (https://github.com/motokimura/spacenet_building_detection).
  Also, experiments were conducted on natural image datasets ImageNet, DIV2K, and COCO.
- We will provide a pre-trained model trained using ImageNet soon.

### Quick Start
**Natural images steganography**

To train models for steganography on natural datasets, use the following commands:
```bash
sh scripts/train.sh
```

If you only want to test the performance:

```bash
 sh scripts/test.sh
```
Correctly set the path and checkpoint of the pre-trained model in test.sh. The path must be consistent with the path automatically generated after training.

**Multispectral remote sensing images steganography**

To train models for steganography on natural datasets, use the following commands:
```bash
sh scripts/train_multi.sh
```

If you only want to test the performance:

```bash
 sh scripts/test_multi.sh
```

### Analysis Tools

We support analysis tools for security and robustness of steganography, including:

```bash
# recovery ability under noise attack
python robustness.py

# Analyze the changes in each channel
channel_analysis.ipynb

```


## Citation
If our work is useful for your research, please consider citing:

```
@article{yang2026ursmamba,
  title={URSMamba: Universal Remote Sensing Image Steganography Using State Space Model},
  author={Yang, Chao and Wang, Shiyuan and Huang, Ying and Guo, Mingqiang},
  journal={Neural Networks},
  pages={109132},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgment
This project is based on [VMamba](https://github.com/MzeroMiko/VMamba) and [UDH](https://github.com/ChaoningZhang/Universal-Deep-Hiding), thanks for their excellent works and all the contributors for open-sourcing.
