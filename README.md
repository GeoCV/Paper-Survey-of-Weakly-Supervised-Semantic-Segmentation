# Paper Survey of Weakly Supervised Semantic Segmentation

## Image-level weak labels
### Generate fine-grained initial pseudo labels (CAMs)
* Reasons
  + CAMs only focus on the discriminative parts (incomplete response regions)
  + Bad initial labels will influence the succeeding refinement
#### [Self-supervised Scale Equivariant Network for Weakly Supervised Semantic Segmentation](https://arxiv.org/abs/1909.03714) CVPR'19
#### [Weakly-Supervised Semantic Segmentation via Sub-category Exploration](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Weakly-Supervised_Semantic_Segmentation_via_Sub-Category_Exploration_CVPR_2020_paper.pdf) CVPR'20
* Method: Make a more challenging task for classification model
  + Sub-category classification
    - Pseudo labels are produced by K means (unsupervised method)
    - Pseudo labels are as the GT for sub-category task (self-supervised method)
  + Dense CRF
  + Two-step solution
  + Random walk
### Refine initial pseudo labels
* Reason
  + CAMs only focus on the discriminative parts (incomplete response regions)
#### [Learning Pixel-level Semantic Affinity with Image-level Supervisionfor Weakly Supervised Semantic Segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ahn_Learning_Pixel-Level_Semantic_CVPR_2018_paper.pdf) CVPR'18
* Method: Using CAMs to train affinity matrix that establishes connections between each pixel in a image
  + Affinity matrix
  + Dense CRF
  + Two-step solution
  + Random Walk
#### [Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Weakly_Supervised_Learning_of_Instance_Segmentation_With_Inter-Pixel_Relations_CVPR_2019_paper.pdf) CVPR'19
* Method: Using inter-pixel relation to produce affinity matrix and CAMs refinement
  + Displacement field for CAM refinement
    - Generate a vector for each pixel that points to the centroid of the instance
    - The pixels will be classified to class of the pointed centroid belongs to
  + Class boundary map for affinity matrix
    - Use boundary map to produce affinity matrix
    - If there is a boundary between two pixels, the affinity value will be small
  + Dense CRF
  + Two-step solution
  + Random Walk
### End-to-end training
#### [Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Weakly-Supervised_Semantic_Segmentation_CVPR_2018_paper.pdf) CVPR'18
* Method: Using region growing method that expand initial CAM with training segmentation end-to-end
  + Region Growing
    - Expand the CAM labels according to neighbors
  + Dense CRF
  + One-step solution
#### [Reliability Does Matter: An End-to-End Weakly Supervised Semantic Segmentation Approach](https://arxiv.org/abs/1911.08039) AAAI'20
* Method: Generate very reliable labels (small regions) as the supervision for training segmentation end-to-end
  + Multi-scale CAMs by multi-scale images
  + Dense CRF + reliable CAM for producing very reliable labels
  + Adopt new dense energy loss for those unlabeled regions (very large regions)
  + One-step solution
