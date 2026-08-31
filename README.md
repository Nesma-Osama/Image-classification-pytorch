# 🐱🐶🧑 Image Classification with PyTorch

This is a simple image classification project using **PyTorch**. The model classifies images into three classes:

* 🐱 Cat
* 🐶 Dog
* 🧑 Person

I implemented two models in this project:

* A custom CNN built using PyTorch
* VGG16 using transfer learning

The project is divided into different files for the dataset, dataloader, models, training, and prediction.

---

## 📌 Project Overview

The main workflow is:

```text
Images
   ↓
Dataset
   ↓
DataLoader
   ↓
CNN / VGG16
   ↓
Training
   ↓
Saved Model
   ↓
Prediction
```

For VGG16, I used the pretrained model and froze the feature extraction layers, then replaced the last layer to work with the three classes in my dataset.

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
│   ├── cnn_model.py
│   ├── vgg_model.py
│   ├── trainer.py
│   └── predicter.py
│
├── train.py
├── predict.py
├── output_results/
├── requirements.txt
└── README.md
```

---

## 🧠 CNN Model

The first model is a custom CNN with four convolutional blocks.

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

The number of channels increases as follows:

```text
3 → 32 → 64 → 128 → 256
```

After the convolutional layers, the output is flattened and passed through fully connected layers:

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

The final layer has 3 outputs for:

```text
0 → Cat
1 → Dog
2 → Person
```

The class mapping is created from the dataset folders and saved using Pickle.

---

## 🧠 VGG16 Transfer Learning

I also added **VGG16** using transfer learning.

Instead of training VGG16 from the beginning, I used the pretrained weights from ImageNet.

The feature extraction layers are frozen:

```python
for param in model.features.parameters():
    param.requires_grad = False
```

Then I replaced the last layer of the classifier because the original VGG16 model is made for 1000 ImageNet classes, while this project only has 3 classes.

```python
in_features = model.classifier[-1].in_features

model.classifier[-1] = nn.Linear(
    in_features,
    3
)
```

So the basic idea is:

```text
Image
  ↓
VGG16 Features
  ↓
Frozen pretrained layers
  ↓
Classifier
  ↓
3 classes
```

---

## 🔄 Image Preprocessing

The images are resized to:

```text
224 × 224
```

For training, I used random horizontal flipping as a simple data augmentation technique.

The images are also normalized using the ImageNet mean and standard deviation:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Training:

```text
Resize
  ↓
Random Horizontal Flip
  ↓
ToTensor
  ↓
Normalize
```

Testing:

```text
Resize
  ↓
ToTensor
  ↓
Normalize
```

---

## 🚂 Training

To start training:

```bash
python train.py
```

The script asks which model to use:

```text
Enter model name (vgg/cnn):
```

Enter:

```text
cnn
```

to train the custom CNN, or:

```text
vgg
```

to train the VGG16 model.

The training uses:

```text
Batch size: 32
Optimizer: Adam
Loss: CrossEntropyLoss
```

The trained models are saved as `.pth` files.

For example:

```text
output_results/
├── cnn.pth
├── vgg.pth
└── class_name_mapping.pkl
```

---

## 📊 Dataset

The dataset contains three classes:

```text
cat
dog
person
```

The dataset is available here:

**[Download the Dataset](https://drive.google.com/file/d/1G9H2W0R6JLYYBXyHNM_kAcBufUJr4Qsy/view?usp=drive_link)**

After downloading it, place the `Classification_dataset_v3` folder inside the `data` folder.

The structure should be:

```text
data/
└── Classification_dataset_v3/
    └── images/
        ├── train/
        │   ├── cat/
        │   ├── dog/
        │   └── person/
        │
        └── test/
            ├── cat/
            ├── dog/
            └── person/
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Nesma-Osama/Image-classification-pytorch.git

cd Image-classification-pytorch
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the dataset and put it inside:

```text
data/Classification_dataset_v3/
```

### 5. Train

```bash
python train.py
```

Choose either:

```text
cnn
```

or:

```text
vgg
```

### 6. Predict an image

After training:

```bash
python predict.py
```
Choose either:

```text
cnn
```

or:

```text
vgg
```

The model will predict one of:

```text
Cat
Dog
Person
```

---

## 🛠️ Technologies

* Python
* PyTorch
* Torchvision
* Pillow
* NumPy
* Matplotlib
* Pickle

---
