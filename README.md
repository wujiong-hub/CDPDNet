# CDPDNet

This repository is the official implementation for the paper:  
**[CDPDNet: Integrating Text Guidance with Hybrid Vision Encoders for Medical Image Segmentation](http://arxiv.org/abs/2505.18958)**  
*Authors: Jiong Wu, Yang Xing, Boxiao Yu, Wei Shao, and Kuang Gong

>Abstract: Most publicly available medical segmentation datasets are only partially labeled, with annotations provided for a subset of anatomical structures. When multiple datasets are combined for training, this incomplete annotation poses challenges, as it limits the model's ability to learn shared anatomical representations among datasets. Furthermore, vision-only frameworks often fail to capture complex anatomical relationships and task-specific distinctions, leading to reduced segmentation accuracy and poor generalizability to unseen datasets. In this study, we proposed a novel CLIP-DINO Prompt-Driven Segmentation Network (CDPDNet), which combined a self-supervised vision transformer with CLIP-based text embedding and introduced task-specific text prompts to tackle these challenges. Specifically, the framework was constructed upon a convolutional neural network (CNN) and incorporated DINOv2 to extract both fine-grained and global visual features, which were then fused using a multi-head cross-attention module to overcome the limited long-range modeling capability of CNNs. In addition, CLIP-derived text embeddings were projected into the visual space to help model complex relationships among organs and tumors. To further address the partial label challenge and enhance inter-task discriminative capability, a Text-based Task Prompt Generation (TTPG) module that generated task-specific prompts was designed to guide the segmentation. Extensive experiments on multiple medical imaging datasets demonstrated that CDPDNet consistently outperformed existing state-of-the-art segmentation methods.



---

## Overview

<p align="center">
  <img src="documents/fig1_wholearch.jpg" alt="Figure 1 Overview" width="1000">
  <br>
</p>

---

## Environment

```bash
# Clone the repository
git clone https://github.com/wujiong-hub/CDPDNet.git

# Create and activate a new conda environment
conda create -n cdpdnet python=3.9
conda activate cdpdnet

# Install PyTorch (please modify according to your server's CUDA version)
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

```


## Training
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1238 train.py --data_root_path DATA_DIR --dist True --uniform_sample
```
## Testing
1. Do the inference process directly adopt our trained model:
```python
cd pretrained_weights/
wget https://huggingface.co/jwu2009/CDPDNet/resolve/main/cdpdnet.pth
cd ../
CUDA_VISIBLE_DEVICES=0 python test.py --data_root_path DATA_DIR --resume pretrained_weights/cdpdnet.pth --store_result 
```

2. Do the inference process using your own trained model:
```python
CUDA_VISIBLE_DEVICES=0 python test.py --data_root_path DATA_DIR --resume CHECKPOINT_PATH --store_result 
```

## Acknowledgement

We appreciate the effort of the following repositories in providing open-source code to the community:

- [MONAI](https://monai.io/)
- [CLIP-Driven-Universal-Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model/tree/main)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

## Citation
If you find this repository useful, please consider citing this paper:
```
@misc{wu2025cdpdnetintegratingtextguidance,
      title={CDPDNet: Integrating Text Guidance with Hybrid Vision Encoders for Medical Image Segmentation}, 
      author={Jiong Wu and Yang Xing and Boxiao Yu and Wei Shao and Kuang Gong},
      year={2025},
      eprint={2505.18958},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.18958}, 
}
```





