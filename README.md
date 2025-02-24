# Digit Recognition Using Deep Learning

This project focuses on recognizing handwritten digits using advanced deep learning techniques, leveraging a Convolutional Neural Network (CNN) trained on the MNIST dataset. The system demonstrates high accuracy in classifying digits, showcasing the power of neural networks for image recognition tasks.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Project Overview
This project involves the development and evaluation of a digit recognition system. A custom CNN was implemented and trained on the MNIST dataset using optimizers such as Adam and RMSprop. Regularization techniques, including dropout, were applied to improve the model’s generalization ability. Additionally, pre-trained models like LeNet-5, ResNet-like, and VGG16 were evaluated to enhance performance, with VGG16 achieving the highest accuracy.

## Dataset
The MNIST dataset was used for training and testing the model. It contains:
- **Training Set**: 60,000 examples (split into 50,000 for training and 10,000 for validation).
- **Testing Set**: 10,000 examples.

Each image is 28x28 pixels and represented by grayscale values ranging from 0 to 255, where 0 is black, 255 is white, and intermediate values represent shades of gray.

## Model Architecture
### Custom CNN
- A custom 8-layer CNN was designed for digit recognition.
- Regularization techniques (L1, L2, and dropout) were used to enhance generalization.
- Models were trained using Adam and RMSprop optimizers, achieving high accuracy.

### Pre-Trained Models
1. **LeNet-5**: A lightweight CNN model, efficient for digit recognition tasks.
2. **ResNet-like**: A scaled-down ResNet with residual connections to address vanishing gradient issues.
3. **VGG16**: A deeper architecture with 16 layers, offering higher accuracy at the cost of computational complexity.

## Installation and Usage

### Prerequisites
- Python 3.x
- Required libraries: TensorFlow, Keras, NumPy, Matplotlib (install via `requirements.txt`).

### Steps to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Naidu-2002/Digit-Recognition.git
   cd Digit-Recognition

2. **Install Dependencies**:  
   Install all the required Python libraries using `pip`:  

   ```bash
   pip install -r requirements.txt

## Results

The digit recognition system achieved excellent performance through both custom CNN and pre-trained models.

### Custom CNN
- **Adam Optimizer**: Achieved 97% test accuracy with minimal loss.
- **RMSprop Optimizer**: Slightly higher accuracy of 97.9% but with higher validation loss.

### Pre-Trained Models
The performance of pre-trained models was evaluated using Adam and RMSprop optimizers:

| Model       | Optimizer | Test Accuracy | Test Loss |
|-------------|-----------|---------------|-----------|
| LeNet-5     | Adam      | 97.9%         | 0.07      |
| ResNet-like | RMSprop   | 99.48%        | 0.02      |
| VGG16       | Adam      | 99.5%         | 0.022     |

### Regularization Methods (Custom CNN)
The impact of regularization techniques was analyzed:

| Method     | Test Accuracy | Test Loss |
|------------|---------------|-----------|
| L1         | 96.9%         | 0.75      |
| L2         | 95.0%         | 0.32      |
| Dropout    | 97.54%        | 0.14      |

### Visualizations
- **Accuracy and Loss Curves**: Accuracy and loss trends during training for both Adam and RMSprop optimizers are stored in the `results/` directory.
- **Predicted Outputs**: Example digit classifications are saved as images in the `results/` directory.

These results demonstrate the effectiveness of deep learning techniques for handwritten digit recognition, with VGG16 achieving the highest accuracy among the evaluated models.

## Future Work

The digit recognition system has shown promising results, but there are several areas for future improvements:
- **Model Enhancement**: Experiment with advanced architectures like transformers or attention-based models to improve accuracy.
- **Real-Time Deployment**: Develop a web or mobile application for real-time digit recognition.
- **Extended Dataset**: Expand the dataset to include more complex and diverse digit samples (e.g., EMNIST or custom datasets).
- **Explainability**: Integrate model interpretability techniques like Grad-CAM to visualize the features influencing predictions.
- **Hyperparameter Optimization**: Use automated tools like Grid Search or Bayesian Optimization to find the best hyperparameters.
- **Cross-Platform Support**: Develop a lightweight model for deployment on edge devices like mobile phones or embedded systems.

## Contributors

- **Naidu-2002** - Project Lead and Developer  
