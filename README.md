# CDPDNet

This repository is the official implementation for the paper:  
**[CDPDNet: Integrating Text Guidance with Hybrid Vision Encoders for Medical Image Segmentation](https://arxiv.org/pdf/2411.15426)**  
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
- git clone https://github.com/wujiong-hub/LDM-Morph.git
- conda create -n ldmmorph python=3.8
- conda activate ldmmorph
- pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
- pip install -r requirements.txt

## Datasets
- [CAMUS dataset, including two-chamber (CAMUS-2CH) and four-chamber (CAMUS-4CH) images](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
- [EchoNet-Dynamic dataset](https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a)
- [Automated Cardiac Diagnosis Challenge (ACDC) dataset](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)

## Datasets preprocessing
1. Download all datasets, unzip them, and put them inside the corresponding datasets folder.
2. For the CAMUS dataset, directly run the script:
   ```python
   cd datasets/scripts/
   mkdir ../CAMUS/prep/
   python preprocess_camus.py
   ```
   For the ECHO dataset, please get the github codes of echonet from the link https://github.com/echonet/dynamic, and install its environment. Then run the script:
   ```python
   cd datasets/script/
   mkdir ../ECHO/prep/
   python preprocess_echo.py
   ```
   For the ACDC dataset, directly run the script:
   ```python
   cd datasets/script/
   mkdir ../ACDC/prep/
   python preprocess_acdc.py
   ```
## Training 
The dataloader file is ldm/data/camus.py. Change the corresponding training/validation path for each dataset to start training. For more details, please refer to the [Latent Diffusion Model(LDM)](https://github.com/CompVis/latent-diffusion).
1. Train the LDM-based latent feature extraction module
   Firstly, the autoencoder should be trained by running the following script:
   ```python
   python main.py --base configs/autoencoder/autoencoder_camus_vq16.yaml --train --gpus 0,1,2,3,4,5,6,7
   ```
   Then, we train the diffusion model. Specify the file path of "ckpt_path" to the autoencoder model saved path in the first training step under the "first_stage_config". 
   ```python
   python main.py --base configs/latent-diffusion/casmus-ldm-vq16-64ch.yaml --train --gpus 0,1,2,3,4,5,6,7
   ```
3. Train the registration network
   ```python
   python train.py
   ```

## Testing
```python
   python test.py
```

## Acknowledgement

Some of the codes in this repo are borrowed from:
- [Latent Diffusion Model(LDM)](https://github.com/CompVis/latent-diffusion)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
- [Transmatch](https://github.com/tzayuan/TransMatch_TMI)

## Citation
If you find this repository useful, please consider citing this paper:
```
@article{wu2024ldm,
  title={LDM-Morph: Latent diffusion model guided deformable image registration},
  author={Wu, Jiong and Gong, Kuang},
  journal={arXiv preprint arXiv:2411.15426},
  year={2024}
}
```





