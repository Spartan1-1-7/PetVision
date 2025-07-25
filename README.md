#  🐾 PetVision: Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN)-based machine learning project that classifies images as either a cat or a dog. The model is trained on the popular Kaggle Cats and Dogs dataset and achieves an accuracy of **0.8**. Now featuring a **responsive web interface** built with Streamlit!

## 🚀 Live Demo

**Try it now:** [https://petvision.streamlit.app/](https://petvision.streamlit.app/)

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Web Interface](#web-interface)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Contributing](#contributing)

---

## Project Overview

PetVision leverages a deep learning CNN model to automatically classify input images as either a cat or a dog. The project demonstrates the effectiveness of CNNs for image classification tasks and provides a practical example using a well-known public dataset. 

**Now features a beautiful, responsive web interface** that allows users to easily upload images and get instant predictions through an intuitive Streamlit application.

---

## Features

- 🔍 Classifies images as either **cat** or **dog**
- 🧠 Built using **Keras** and **TensorFlow**
- 📊 Trained on the **Kaggle Cats and Dogs** dataset
- 🎯 Achieves ~80% accuracy
- 🌐 **Responsive web interface** with Streamlit
- 📱 **Mobile-friendly design** that works on all devices
- 🎨 **Modern UI** with professional styling
- ⚡ **Instant predictions** with real-time image processing
- 📁 **Drag-and-drop file upload** for easy image submission

---

## 🎨 Web Interface

The PetVision web application features:

- **Responsive Design:** Works seamlessly on desktop, tablet, and mobile devices
- **User-Friendly Interface:** Clean, intuitive design with easy navigation
- **Real-Time Processing:** Upload an image and get instant AI predictions
- **Professional Styling:** Modern CSS with responsive layouts and visual feedback
- **Error Handling:** Robust file validation and user guidance

### Interface Features:
- Welcome message with project description
- Drag-and-drop file uploader supporting JPG, JPEG, and PNG formats
- Responsive image display that adapts to screen size
- Styled prediction results with success messaging
- Mobile-optimized layout for on-the-go usage

---

## Tech Stack

### Core ML Stack:
- **Python**
- **TensorFlow/Keras** (Deep Learning Framework)
- **NumPy** (Numerical Computing)

### Web Interface:
- **Streamlit** (Web Framework)
- **CSS3** (Responsive Styling)
- **HTML5** (Structure)

### Development & Deployment:
- **Conda** (Environment Management)
- **Streamlit Cloud** (Hosting Platform)
- **Git** (Version Control)

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
- **Model Performance:** The CNN generalizes well to unseen cat and dog images
- **Web Interface:** Successfully deployed on Streamlit Cloud with responsive design
- **User Experience:** Mobile-friendly interface with intuitive image upload and prediction display

---

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.7+
- Git

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Spartan1-1-7/PetVision.git
   cd PetVision
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

### Using the Web Interface

1. **Visit the live demo:** [https://petvision.streamlit.app/](https://petvision.streamlit.app/)
2. **Upload an image:** Click "Browse files" or drag and drop a JPG/PNG image
3. **Process:** Click the "🚀 Process Files" button
4. **View results:** See your uploaded image and the AI prediction result

### Supported Image Formats
- JPG/JPEG
- PNG
- Maximum file size: 200MB per file

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Areas
- Model improvements and accuracy optimization
- UI/UX enhancements
- Additional image format support
- Performance optimizations
- Mobile app development

---