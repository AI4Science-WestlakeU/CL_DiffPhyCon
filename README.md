# CL-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems (ICLR 2025)

[Paper](https://openreview.net/forum?id=PiHGrTTnvb) | [arXiv](https://arxiv.org/pdf/2408.03124) 
<!-- | [Poster](https://github.com/AI4Science-WestlakeU/cindm/blob/main/assets/CinDM_poster.pdf)  -->
<!-- | [Tweet](https://twitter.com/tailin_wu/status/1747259448635367756)  -->

Official repo for the paper [CL-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems](https://openreview.net/pdf?id=PiHGrTTnvb).<br />
[Long Wei*](https://longweizju.github.io/), [Haodong Feng*](https://scholar.google.com/citations?user=0GOKl_gAAAAJ&hl=en), [Yuchen Yang](), [Ruiqi Feng](https://weenming.github.io/),  [Peiyan Hu](https://peiyannn.github.io/), [Xiang Zheng](), [Tao Zhang](https://zhangtao167.github.io), [Dixia Fan](https://en.westlake.edu.cn/faculty/dixia-fan.html), [Tailin Wu](https://tailin.org/)<br />
ICLR 2025. 

We propose a diffusion method with an asynchronous denoising schedule for physical systems control tasks. It achieves closed-loop control with a significant speedup of sampling efficiency. Specifically, it has the following features:

- Efficient Sampling: CL-DiffPhyCon significantly reduces the computational cost during the sampling process through an asynchronous denoising framework. Compared with existing diffusion-based control methods, CL-DiffPhyCon can generate high-quality control signals in a much shorter time.

- Closed-loop Control: CL-DiffPhyCon enables closed-loop control, adjusting strategies according to real-time environmental feedback. It outperforms open-loop diffusion-based planning methods in control effectiveness.

- Accelerated Sampling: CL-DiffPhyCon can integrate with acceleration techniques such as [DDIM](https://arxiv.org/abs/2010.02502). It further enhances control efficiency while keeping the control effect stable.

Framework of CL-DiffPhyCon:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/close_loop_diffcon/blob/main/assets/figure1.png" align="center" width="800" ></a>

This is a follow-up work of our previous DiffPhyCon (NeurIPS 2024): [Paper](https://openreview.net/pdf?id=MbZuh8L0Xg) | [Code](https://github.com/AI4Science-WestlakeU/diffphycon).

# Installation

Run the following commonds to install dependencies. In particular, when run the smoke control task, the python version must be 3.8 due to the requirement of the Phiflow software.

```code
conda env create -f environment.yml
conda activate DiffPhyCon
```

# Dataset and checkpoints
## Dataset
The checkpoints and test datasets of our CL-DiffPhyCon on both tasks (1D Burgers and 2D smoke) can be downloaded in [link](https://drive.google.com/drive/folders/1moLdtqmvmAU8FoWt6ELWOTXT0tPuY-qJ). To run the following training and inference scripts locally, replace the path names in the following scripts by your local paths.
<!-- Because the training dataset in the 2D experiment is over 100GB, it is not contained in this link. -->

# Training:
## 1D Burgers' Equation Control:

In the scripts_1d/ folder, run the following two scripts to train the synchronous and asynchronous diffusion models, respectively:
```code
bash train_syn.sh
bash train_asyn.sh
```

## 2D Smoke Control:

Similarly, in the scripts_2d/ folder, run the following two scripts:
```code
bash train_syn.sh
bash train_asyn.sh
```

# Inference:
## 1D Burgers' Equation Control:
In the scripts_1d/ folder, run the following script for closed-loop diffusion control:
```
bash inf_asyn.sh
```

## 2D Smoke Control:
### CL-DiffPhyCon
In the scripts_2d/ folder, run the following script for closed-loop diffusion control:
```
bash inf_asyn.sh
```

And then run evaluate_2d.py to evaluate the inference results
```
python evaluate_2d.py
```

## Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{
wei2025cldiffphycon,
title={{CL}-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems},
author={Long Wei and Haodong Feng and Yuchen Yang and Ruiqi Feng and Peiyan Hu and Xiang Zheng and Tao Zhang and Dixia Fan and Tailin Wu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=PiHGrTTnvb}
}
```
