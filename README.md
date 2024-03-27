## NDI Image Pair Matching

**Published paper to be uploaded soon.**

### Introduction
This is a PyTorch implementation of a multi-scale losses based NDI image pair matching method. NanoDiffraction Imaging (NDI), a scanning transmission electron mircoscopy based nodel imaging technique, is used to help researchers find the relationships between microscopic structure and macroscopic properties, of which he original paper can be found [here](https://pubs.acs.org/doi/10.1021/acs.macromol.1c00683).

#### Why NDI Image Pair Matching?
The processing of these NDI images requires meticulous observation and selection by experts, particularly in the labor-intensive process of making NDI image pairs based on two different patterns generated in NDI process. Thus, a novel way is required to avoid consuming too much energy for researchers. 

#### NDI Images in This Project
The creation and analysis of NDI images involves a multi-step process:
1. An electron beam is irradiated onto a specimen that consists of many small crystals oriented in various directions. The geometrical relationship between the beam and each crystal results in a variety of diffraction patterns. Each of these diffraction patterns is recorded on a detector as **Observed Images**.
2. Using parameters provided from the Crystallographic Information File (CIF), **Calculated Images** are obtained from a structure database.
3. The core is in the comparison and matching of the observed and calculated diffraction patterns to evaluate the orientation of each crystal in the specimen, which is time-consuming and laborious.

![](./images/Details%20of%20dataset%20alternative.png)
This figure shows the NDI image examples in this project. For each given observed image, there exists only one corresponding calculated image. The objective of this paper is to find the corresponding calculated image to an observed image given as a query.

**For access to this dataset please contact (Upload Later).**

### Usage

*Please make sure that all packages in `requirements.txt` are installed.*

#### Pretraining
```bash
sh run_pretraining.sh
```

The pretrained model will be saved in `./checkpoints/`.


#### Fine-tuning

```bash
sh run_finetuning.sh
```

### Reference
* [MoCo V2](https://github.com/facebookresearch/moco)
* [Wasserstein Distance](https://github.com/VinAIResearch/DSW)
* [PyRetri](https://github.com/PyRetri/PyRetri)
