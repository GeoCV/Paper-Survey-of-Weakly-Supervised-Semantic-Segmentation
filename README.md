# Paper Survey of Weakly Supervised Semantic Segmentation

## Image-level weak labels
### Generate fine-grained initial psuedo labels (CAMs)
#### [Self-supervised Scale Equivariant Network for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/1909.03714) CVPR'19
#### [Weakly-Supervised Semantic Segmentation via Sub-category Exploration](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Weakly-Supervised_Semantic_Segmentation_via_Sub-Category_Exploration_CVPR_2020_paper.pdf) CVPR'20
* Target: Optimize CAMs
  + CAMs onlu focus on the discriminative parts (incomplete response regions)
* Method: Make a more challenging task for classification model
  + Sub-category classification
    - Pseudo labels are produced by K means (unsupervised method)
    - Pseudo labels are as the GT for sub-category task (self-supervised method)
  + Random walk
### Refine initial psuedo labels
#### [Learning Pixel-level Semantic Affinity with Image-level Supervisionfor Weakly Supervised Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.pdf) CVPR'18
#### [Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Weakly_Supervised_Learning_of_Instance_Segmentation_With_Inter-Pixel_Relations_CVPR_2019_paper.pdf) CVPR'19
#### [Weakly-Supervised Semantic Segmentation via Sub-category Exploration](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Weakly-Supervised_Semantic_Segmentation_via_Sub-Category_Exploration_CVPR_2020_paper.pdf) CVPR'20
