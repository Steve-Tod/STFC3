# Breaking Shortcut: Exploring Fully Convolutional Cycle-Consistency for Video Correspondence Learning

[Yansong Tang](https://andytang15.github.io/) \*, [Zhenyu Jiang](http://zhenyujiang.me) \*, [Zhenda Xie](https://zdaxie.github.io/) \*, [Yue Cao](http://yue-cao.me/), [Zheng Zhang](https://stupidzz.github.io/), [Philip H. S. Torr](https://www.robots.ox.ac.uk/~phst/), [Han Hu](https://ancientmooner.github.io/) (* equal contribution)

[arxiv](https://arxiv.org/pdf/2105.05838.pdf) 

## Introduction

This is the the repository for *Breaking Shortcut: Exploring Fully Convolutional Cycle-Consistency for Video Correspondence Learning*, published in [SRVU - ICCV 2021 workshop](https://sites.google.com/view/srvu-iccv21-workshop/home).

If you find our work useful in your research, please consider citing us.

```
@article{tang2021breaking,
  title={Breaking Shortcut: Exploring Fully Convolutional Cycle-Consistency for Video Correspondence Learning},
  author={Tang, Yansong and Jiang, Zhenyu and Xie, Zhenda and Cao, Yue and Zhang, Zheng and Torr, Philip HS and Hu, Han},
  journal={arXiv preprint arXiv:2105.05838},
  year={2021}
}
```

## Installation

1. Create a conda environment with Python 3.8.

2. Install Pytorch 1.5 with CUDA 10.2.

3. Install packages list in [requirements.txt](requirements.txt).

4. Install NVIDIA Apex following the instruction [here](https://github.com/NVIDIA/apex).

## Data

We use the Kinetics400 dataset. You can find directions for downloading it [here](https://github.com/pytorch/vision/tree/master/references/video_classification).

To facilitates data preparation, we save the precomputed metadata given by `torchvision.datasets.Kinetics400`, and load it before training.

## Training and Testing

### Training

Run:

```bash
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train.py -opt $OPTION_FILE_NAME -extra amp_opt_level=O1
```

An example option file is [here](code/options/train/STFC3.yaml)

### Testing
You could download our pretrained model [here](https://drive.google.com/file/d/14QOKJZcB8GdA6W2oV0nysbBevYWebLJs/view?usp=sharing)

We follow the [CRW](https://github.com/ajabri/videowalk) to perform downstream task evaluation

An example command is:

```bash
bash davis_test_script.sh $TRAINED_MODEL_PATH reproduce 0 -1
```

## Related Repositories

1. [CRW](https://github.com/ajabri/videowalk) 
