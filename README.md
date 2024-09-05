# 3D Semantic Novelty Detection

Deep learning techniques have shown significant promise in addressing challenges related to 3D point cloud processing and 3D shape understanding. However, point clouds' unstructured and unordered nature poses significant difficulties for conventional convolutional neural networks (CNNs). Recently, PointNet has provided a foundational approach for processing point clouds by independently handling each point and subsequently aggregating the information, yet it struggles with capturing local structures inherent in the data. This project focuses on 3D semantic novelty detection, specifically identifying whether a given 3D point cloud belongs to a predefined set of known semantic classes (in-distribution) or if it is an outlier (out-of-distribution). We leverage the 3DOS benchmark for the Synthetic to Real scenario, where the nominal data are synthetic point clouds, and the test samples are real-world point clouds. Our exploration includes evaluating baseline methods for novelty detection, conducting a thorough analysis of failure cases, and investigating the use of large-scale pre-trained models to enhance performance. This work contributes to the ongoing development of more robust and generalizable models for 3D point cloud analysis.

## Replicate our experiments

There are four indipendent notebooks available to run the code.

1. `pointnet++_baselines.ipynb`: it contains the code to replicate the experiments  that were proposed in the 3DOS original paper, using the PointNet++ model. 

2. `dgcnn_baselines.ipynb`: it contains the code to replicate the experiments  that were proposed in the 3DOS original paper, using the DGCNN model.

4. `pointnet++_failure_case.ipynb`: it contains the code conduct a failure case analysis, using the PointNet++ model.

4. `openshape.ipynb`: it contains the code to evaluate the OpenShape model (PointBERT-ViTg14) in the Synthetic-to-Real setting.


