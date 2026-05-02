# Pneumonia-Prediction-Model-using-PneumoniaMNIST-database

This repository contains a Google Colab Notebook implementing a convolutional neural network (CNN) for pneumonia classification using PneumoniaMNIST dataset.

The primary objective of this project was to develop a foundational understanding of **convolutional neural networks for image analysis (computer vision)**. Through this work, I implemented key CNN components, including convolutional layers (Conv2d) for feature extraction and pooling layers (MaxPool2d) for spatial feature downsampling, enabling the model to capture relevant patterns while reducing dimensionality.

### <u>How to access the code</u>

The code is available in the Jupyter Notebook (.ipynb) file within this repo. You can either download the file or open it directly to view the implementation.

### <u>Model Architecture</u>

MultiLayerModel(

(network): Sequential(

(0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(1): ReLU()

(2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(4): ReLU()

(5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(6): Flatten(start_dim=1, end_dim=-1)

(7): Linear(in_features=1568, out_features=64, bias=True)

(8): ReLU()

(9): Linear(in_features=64, out_features=1, bias=True)

)
)

- **Convolutional layers (Conv2d)** are used to extract spatial features from input chest X-ray images. The number of output channels increases from 16 to 32, allowing the network to learn progressively more complex patterns. Padding of 1x1 is applied to preserve spatial dimensions and prevent loss of edge information when using 3x3 kernels on 28x28 images.
- **Max pooling (MaxPool2d)** is applied after each convolutional block to reduce spatial dimensions, improving computational efficiency while retaining most meaningful features.
- **Flattening** converts the final feature maps (eg. 3D tensors of shape [32,7,7], given 28x28 inputs and two pooling operations) into a 1D tensor for input into linear layers.
- **Loss function:** BCEWithLogitsLoss is used, which combines sigmoid activation with binary cross-entropy into a single numerically stable operation. This is why the **final layer** outputs a raw logit rather than a probability.

### <u>Model Improvement Notes</u>

Through hyperparameter tuning, the model achieved a test accuracy of ~86%.

| **Epochs** | **Learning rate** | **Batch size** | **Accuracy (val)** | **Accuracy (test)** |
| --- | --- | --- | --- | --- |
| 100 | 0.001 | 64 | 96.76% | 85.90% |
| 100 | 0.01 | 64 | 96.18% | 83.97% |
| 100 | 0.001 | 32 | 96.95% | 84.94% |

Although a test accuracy of roughly 86% is a solid result for an initial CNN model, accuracy alone is insufficient in a clinical context. False negatives — failing to detect pneumonia — carries greater risk than a false positive, as missed diagnoses can delay critical treatment. Future work should focus on additional metrics such as recall, precision, and AUC-ROC, and explore techniques like data augmentation to improve model generalization.
