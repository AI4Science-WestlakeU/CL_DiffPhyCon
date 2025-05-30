# CL-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems (ICLR 2025)

[Paper](https://openreview.net/forum?id=PiHGrTTnvb) | [arXiv](https://arxiv.org/pdf/2408.03124) 
<!-- | [Poster](https://github.com/AI4Science-WestlakeU/cindm/blob/main/assets/CinDM_poster.pdf)  -->
<!-- | [Tweet](https://twitter.com/tailin_wu/status/1747259448635367756)  -->

Official repo for the paper [CL-DiffPhyCon: Closed-loop Diffusion Control of Complex Physical Systems](https://openreview.net/pdf?id=PiHGrTTnvb).<br />
[Long Wei*](https://longweizju.github.io/), [Haodong Feng*](https://scholar.google.com/citations?user=0GOKl_gAAAAJ&hl=en), [Yuchen Yang](), [Ruiqi Feng](https://weenming.github.io/),  [Peiyan Hu](https://peiyannn.github.io/), [Xiang Zheng](), [Tao Zhang](https://zhangtao167.github.io), [Dixia Fan](https://en.westlake.edu.cn/faculty/dixia-fan.html), [Tailin Wu†](https://tailin.org/)<br />
ICLR 2025. 

We propose a diffusion method with an asynchronous denoising schedule for physical systems control tasks. It achieves closed-loop control with a significant speedup of sampling efficiency. Specifically, it has the following features:

- Efficient Sampling: CL-DiffPhyCon significantly reduces the computational cost during the sampling process through an asynchronous denoising framework. Compared with existing diffusion-based control methods, CL-DiffPhyCon can generate high-quality control signals in a much shorter time.

- Closed-loop Control: CL-DiffPhyCon enables closed-loop control, adjusting strategies according to real-time environmental feedback. It outperforms open-loop diffusion-based planning methods in control effectiveness.

- Accelerated Sampling: CL-DiffPhyCon can integrate with acceleration techniques such as [DDIM](https://arxiv.org/abs/2010.02502). It further enhances control efficiency while keeping the control effect stable.

Framework of CL-DiffPhyCon:

<a href="url"><img src="https://github.com/AI4Science-WestlakeU/close_loop_diffcon/blob/main/assets/figure1.png" align="center" width="800" ></a>

This is a follow-up work of our previous DiffPhyCon (NeurIPS 2024): [Paper](https://openreview.net/forum?id=MbZuh8L0Xg) | [Code](https://github.com/AI4Science-WestlakeU/diffphycon).

# Installation

Run the following commands to install dependencies. In particular, the Python version must be 3.8 when running the 2D smoke control task, as the Phiflow software requires.

```code
conda env create -f environment.yml
conda activate base
```

# Dataset and checkpoints
## Dataset
The training and testing datasets and checkpoints of our CL-DiffPhyCon on both tasks (1D Burgers control and 2D smoke control) can be downloaded in [link](https://drive.google.com/drive/folders/1moLdtqmvmAU8FoWt6ELWOTXT0tPuY-qJ). To run the following training and inference scripts locally, replace the path names in the following scripts with your local paths.
<!-- Because the training dataset in the 2D experiment is over 100GB, it is not contained in this link. -->

# Training:
## 1D Burgers' Equation Control:

In the scripts_1d/ folder, run the following two scripts to train the synchronous and asynchronous diffusion models, respectively:
```code
bash train_syn.sh
bash train_asyn.sh
```

## 2D Smoke Control:

In the scripts_2d/ folder, modify the configs in the file default_config.yaml and the argument "main_process_port" and "gpu_ids" according to your local GPU environments to run [accelerate](https://pypi.org/project/accelerate/) properly. Then, run the following two scripts to train the two diffusion models, respectively:
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

Then in the inference/ folder, run evaluate_2d.py to evaluate the inference results (also modify the data path variable "root" first)
```
python evaluate_2d.py
```

## Related Projects
* [DiffPhyCon](https://github.com/AI4Science-WestlakeU/diffphycon) (NeurIPS 2024): We introduce DiffPhyCon which uses diffusion generative models to jointly model control and simulation of complex physical systems as a single task. 

* [WDNO](https://github.com/AI4Science-WestlakeU/wdno) (ICLR 2025): We propose Wavelet Diffusion Neural Operator (WDNO), a novel method for generative PDE simulation and control, to address diffusion models' challenges of modeling system states with abrupt changes and generalizing to higher resolutions, via performing diffusion in the wavelet space.
  
* [CinDM](https://github.com/AI4Science-WestlakeU/cindm) (ICLR 2024 spotlight): We introduce a method that uses compositional generative models to design boundaries and initial states significantly more complex than the ones seen in training for physical simulations.

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
