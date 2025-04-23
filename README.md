#  PetVision: Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN)-based machine learning project that classifies images as either a cat or a dog. The model is trained on the popular Kaggle Cats and Dogs dataset and achieves an accuracy of **0.8**.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

---

## Project Overview

PetVision leverages a deep learning CNN model to automatically classify input images as either a cat or a dog. The project demonstrates the effectiveness of CNNs for image classification tasks and provides a practical example using a well-known public dataset.

---

## Features

- Classifies images as either **cat** or **dog**
- Built using **Keras** and **TensorFlow**
- Trained on the **Kaggle Cats and Dogs** dataset
- Achieves ~80% accuracy
- Simple prediction API for new images

---

## Tech Stack

- **Python**
- **Conda** (for environment management)
- **NumPy**
- **Pandas**
- **Keras**
- **TensorFlow**

---

## Dataset

- **Source:** [Kaggle Dogs and Cats Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- **Structure:**
```
data/
├── train_set/
│ ├── cat/
│ └── dog/
├── test_set/
│ ├── cat/
│ └── dog/
```
- Contains thousands of labeled images of cats and dogs for robust training and evaluation.

---

## Model Architecture

The CNN model consists of:

- Multiple **convolutional layers** with ReLU activation
- **Max pooling** layers for downsampling
- **Flatten** layer to convert feature maps to a vector
- **Dense (fully connected)** layers
- **Output layer** with sigmoid activation for binary classification
---
## Results

- **Test Accuracy:** ~0.8 (80%)
- The model generalizes well to unseen cat and dog images.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
