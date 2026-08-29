# 🐱🐶🧑 Image Classification with PyTorch

A modular **image classification project built with PyTorch** that classifies images into three categories:

* 🐱 **Cat**
* 🐶 **Dog**
* 🧑 **Person**

The project is structured using separate components for **dataset handling, data loading, model architecture, training, evaluation, and inference**, making it easy to understand, maintain, and extend.

---

## 📌 Project Overview

This project uses a custom **Convolutional Neural Network (CNN)** to classify images into one of three classes.

The complete pipeline is:

```text
Images
   │
   ▼
Dataset
   │
   ▼
DataLoader
   │
   ▼
CNN Model
   │
   ├── Training ──► Trained Model (.pth)
   │
   └── Testing ──► Accuracy
                    

New Image
   │
   ▼
Preprocessing
   │
   ▼
Trained CNN
   │
   ▼
Predicted Class
(Cat / Dog / Person)
```

---

## ✨ Features

* Custom PyTorch `Dataset`
* PyTorch `DataLoader`
* Custom CNN architecture
* Image resizing and normalization
* Batch training
* Cross-entropy loss
* Adam optimizer
* GPU/CPU support
* Class-name mapping saved using Pickle

---

## 📁 Project Structure

```text
ImageClassification/
│
├── data/
│   └── Classification_dataset_v3/
│       └── images/
│           ├── train/
│           │   ├── cat/
│           │   ├── dog/
│           │   └── person/
│           │
│           └── test/
│               ├── cat/
│               ├── dog/
│               └── person/
│
├── src/
│   ├── __init__.py
│   ├── datasets.py
│   ├── dataloader.py
│   ├── model.py
│   ├── trainer.py
│   └── predicter.py
│
├── train.py
├── predict.py
│
├── output_results/
│   ├── cnn.pth
│   └── class_name_mapping.pkl
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

The model is a custom CNN consisting of four convolutional blocks.

Each block contains:

```text
Conv2D
   ↓
Batch Normalization
   ↓
ReLU
   ↓
Max Pooling
```

The convolutional layers use the following channel sizes:

```text
3 → 32 → 64 → 128 → 256
```

After the convolutional layers, the feature maps are flattened and passed through fully connected layers:

```text
Flatten
   ↓
Linear → 512
   ↓
ReLU
   ↓
Linear → 128
   ↓
ReLU
   ↓
Linear → 3
```

The final layer produces **3 logits**, one for each class:

```text
0 → Cat
1 → Dog
2 → Person
```

The actual mapping is automatically created from the dataset folders and saved as:

```text
class_name_mapping.pkl
```

---


## 🚂 Training

Run:

```bash
python train.py
```

The training script:

1. Creates the training dataset.
2. Creates the test dataset.
3. Creates DataLoaders.
4. Initializes the CNN.
5. Trains the model.
6. Saves the trained model.
7. Evaluates the model on the test dataset.

The training configuration currently uses:

```text
Batch size: 32
Learning rate: 0.001
Epochs: 40
Optimizer: Adam
Loss: CrossEntropyLoss
```

---

## 📊 Dataset

The dataset used in this project is available on Google Drive:

**[Download the Dataset](https://drive.google.com/file/d/1G9H2W0R6JLYYBXyHNM_kAcBufUJr4Qsy/view?usp=drive_link)**

After downloading the dataset, extract it and place the `Classification_dataset_v3` folder inside the project's `data` directory.

The final structure should look like:

```text
ImageClassification/
│
├── data/
│   └── Classification_dataset_v3/
│       └── images/
│           ├── train/
│           │   ├── cat/
│           │   ├── dog/
│           │   └── person/
│           │
│           └── test/
│               ├── cat/
│               ├── dog/
│               └── person/
│
├── src/
├── train.py
├── predict.py
└── requirements.txt
```

> **Important:** The dataset must be placed inside the `data/` folder before running the training script.

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Nesma-Osama/Image-classification-pytorch.git
cd Image-classification-pytorch
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from:

**[Google Drive Dataset](https://drive.google.com/file/d/1G9H2W0R6JLYYBXyHNM_kAcBufUJr4Qsy/view?usp=drive_link)**

Extract the downloaded folder and place:

```text
Classification_dataset_v3/
```

inside:

```text
data/
```

So you should have:

```text
data/Classification_dataset_v3/images/train/
data/Classification_dataset_v3/images/test/
```

### 5. Train the model

Run:

```bash
python train.py
```

After training, the trained model and class mapping will be saved in:

```text
output_results/
├── cnn.pth
└── class_name_mapping.pkl
```

### 6. Run inference

After training, you can use the trained model to classify a new image:

```bash
python predict.py
```

The predictor loads the saved model and class mapping and returns one of:

```text
Cat
Dog
Person
```



---

## 🛠️ Technologies Used

| Technology  | Purpose                      |
| ----------- | ---------------------------- |
| Python      | Programming language         |
| PyTorch     | Deep learning framework      |
| Torchvision | Image transformations        |
| Pillow      | Image loading and processing |
| NumPy       | Tensor/image manipulation    |
| Matplotlib  | Image visualization          |
| Pickle      | Saving class mappings        |

---

## 🎯 Project Goals

This project demonstrates how to build a complete image-classification pipeline with PyTorch, including:

* Custom dataset implementation
* DataLoader creation
* CNN architecture design
* Image preprocessing
* Model training
* Model evaluation
* Model serialization
* Class mapping management
* Single-image inference

The modular architecture also makes it easier to replace the CNN, modify preprocessing, change the dataset, or integrate the trained model into another application.

---

