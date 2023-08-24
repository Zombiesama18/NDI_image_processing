## NDI Image Pair Matching

### Introduction
This is a PyTorch implementation of a multi-scale losses based NDI (original paper and introduction [here](https://pubs.acs.org/doi/10.1021/acs.macromol.1c00683)) image pair matching method.

In short, the method contains the contrastive learning based pre-training step and a multi-scale loss function _OriDist Loss_ based fine-tuning step.


### Reference
* [MoCo V2](https://github.com/facebookresearch/moco)
* [Wasserstein Distance](https://github.com/VinAIResearch/DSW)
* [PyRetri](https://github.com/PyRetri/PyRetri)
