## NDI Image Pair Matching Deconstructed Method

### Introduction
This is a PyTorch implementation of a deconstructed NDI image pair matching method.

For more information about NDI image pair matching, please refer to `final` branch.

### Deconstruction

#### Problems
After having a closer review of `calculated images`, there exists several issues that need to be addressed.
1. Multiple observed images corresponds to the calculated images in the same pattern. In details, a match to a calculated image with a different number but the same pattern should also be correct. For example, obserevd image `21`, `25`, `26` and `39` are matched to the corresponding numbered calculated images in the same pattern, which means that matching to any of these numbers should be correct. **However, in previous setting, only matching to the calculated image with the same number is considered correct.**
2. Multiple calculated images share the same type of pattern but in different rotation. For example, calculated image `112`, `115` and `116` have the same type of pattern but different rotation angles.

#### Solution
To solve the above problems and simplify the matching task, we deconstruct the NDI image pair matching into two parts: `Pattern Prediction` and `Rotation Prediction`.
* Pattern Prediction: Manually group all calculated images by their patterns into `37` groups. 
* Rotation Prediction: Select one image from each group to rotate to horizontal as a reference image. Rotate each image in each group from `-90` to `89` degrees (due to the centrosymmetric property of the pattern) and calculated the structural similarity between the rotated image and the reference image. Label each image by the angle with the highest structural similarity. 

The grouping result for the initial 184 `Calculated Images` is shown below.
![](./images/Dataset%20Label.png)

*The whole grouping is shown in `images/groups_all.png`*

By predicting the `group` label and the rotation `angle` of each image, the matching calculated image can be generated by rotating the reference image of that `group` by the `angle`.

### Experiments
#### Implementation Details
Because the originally complex task of matching observed images to calculated images is deconstructed into the pattern prediction and the rotation prediction separately, data augmentation of the original data becomes much easier and more feasible.

In the implementation, we use the simple resampling data augmentation method to resample the original data by rotating a random angle to expand the dataset.

In this experiment, we use the expanded `Observed Images` to test the performance. We split the training and validation data into 85% and 15% respectively.

The result is surprisingly good with `100%` accuracy of Pattern Prediction and `0.075` loss of Rotation Prediction (MSE Loss), which means that the prediction on matching calculated image to a given observed image can be accurate enough for practical use.

#### Usage
*The improvements made by this branch are only for the fine-tuning part of the `final` branch, so the fine-tuning script needs the pretrained model.*

* Pretraining
  ```bash
  sh run_pretraining.sh
  ```
  The pretrained model will be saved in './checkpoints/'.

* Fine-tuning
  ```bash
  sh run_finetuning.sh
  ```

### Limitations
1. Categorization of NDI images is based on artificial visual determination, which can cause some subject error.
2. On the other hand, the categories of NDI images are only limited to those appearing in the original dataset. Therefore, the performance of the model on unknown NDI images is not guaranteed.


### Reference
* [Focal Loss](https://github.com/AdeelH/pytorch-multi-class-focal-loss/tree/master)

