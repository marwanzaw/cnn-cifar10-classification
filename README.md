# CIFAR-10 Image Classification with CNN

This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset.

## Dataset
- CIFAR-10 consists of 60,000 32x32 color images in 10 different classes (airplane, cat, dog, etc.)

## Architecture
- 3 Convolutional layers with ReLU
- MaxPooling after each conv layer
- Flatten → Dense → Dropout → Output layer (Softmax)

## How to Run
```bash
pip install tensorflow matplotlib
python cifar10_cnn.py
```

## Output
- Model saved to `cnn_cifar10_model.h5`
- Accuracy and loss plots saved in the `results/` directory

## Example Plots
Accuracy and loss curves are automatically saved after training.