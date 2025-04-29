# MNIST-Digit-Classification

## Project Title
MNIST Digit Classification and Neural Network Analysis

## Overview
This report contains a comprehensive study of Support Vector Machines (SVMs), Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs), and activation function evaluation on different datasets, mainly focusing on the MNIST digit classification problem.

---

## Contents

### 1. MNIST Digit Classification using Support Vector Machines (SVM)
- **Support Vectors**: Critical points closest to decision boundary.
- **Objective**: Minimize (1/2)||w||² under classification constraints.
- **Techniques Applied**:
  - Linear SVM (without and with scaling)
  - Non-linear SVM (RBF Kernel)
- **Results**:
  - Linear SVM (Standard Scaler): ~92.63% test accuracy
  - Non-Linear SVM (RBF Kernel): ~97.89% test accuracy
- **Insights**: Non-linear SVM with RBF kernel outperforms linear SVM, scaling improves linear SVM performance.

---

### 2. ANN and CNN for MNIST Classification

#### Part A: ANN (Artificial Neural Network)
- **Architecture**:
  - Input: 784 nodes
  - 1st Hidden Layer: 128 neurons (ReLU, Dropout 0.2)
  - 2nd Hidden Layer: 64 neurons (ReLU, Dropout 0.2)
  - Output: 10 neurons (Softmax)
- **Training Details**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
- **Results**:
  - Single Layer ANN: 97.82% test accuracy
  - Two Hidden Layer ANN: 98.07% test accuracy
    ![image](https://github.com/user-attachments/assets/0b2320da-bd56-4669-862a-8602ea4e92cf)
    ![image](https://github.com/user-attachments/assets/bf8c1f69-1aea-4195-9d3a-635acf7f50ea)



#### Part B: CNN (Convolutional Neural Network)
- **Architecture**:
  - 3 Convolutional layers (filters: 32 → 64 → 64, ReLU activation)
  - 2 Max-Pooling layers
  - Flatten → Dense(64) → Dense(10, Softmax)
- **Results**:
  - Train accuracy: 98.62%
  - Test accuracy: 98.07%
![image](https://github.com/user-attachments/assets/817bf5d6-8eda-449e-91b3-7d3520a3260e)

---

### 3. Activation Function Evaluation
- **Dataset**: 5 binary features, binary output (15 samples)
- **Neural Network**:
  - Hidden Layer: 5 neurons
  - Output Layer: 1 neuron
  - Optimizer: Adam
  - Loss: Binary Crossentropy
- **Activation Functions Tested**:
  - Hidden layer: Sigmoid, Tanh, ReLU
  - Output layer: Sigmoid, Tanh, ReLU
- **Best Combination**:
  - Hidden: ReLU
  - Output: Sigmoid
  - Accuracy: 1.00
- **Visualization**: Accuracy heatmap generated.
  ![image](https://github.com/user-attachments/assets/c9d06f11-d2b0-49dd-affb-4a503ce0d68d)


---

## Key Conclusions
- **SVM**: RBF kernel yields highest classification accuracy for MNIST.
- **ANN/CNN**: CNN significantly boosts performance due to feature extraction layers.
- **Activation Functions**: ReLU hidden layers combined with Sigmoid output layers perform best for binary classification.
---

## Notes
- Scaling input features improves SVM performance.
- Deeper networks with Dropout layers help mitigate overfitting in ANN.
- CNN architectures are more effective than simple feedforward ANN for image datasets.
- Activation function choice impacts final model performance significantly.

