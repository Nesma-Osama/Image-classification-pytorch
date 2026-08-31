import os
from src.datasets import ImageClassificationDataset
from src.dataloader import ImageClassificationLoader
from src.cnn_model import CNNModel
from src.vgg_model import VGG_Model
from src.trainer import ClassificationTrainer
from torchvision import transforms

model_name=input("Enter model name (vgg/cnn) ")
print(model_name)
os.makedirs("output_results",exist_ok=True)
images_transforms_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ]
)
images_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ]
)
training_dataset = ImageClassificationDataset(
    "data/Classification_dataset_v3/images/train",
    output_folder_path="output_results",
    transforms=images_transforms_train,
)
training_loader = ImageClassificationLoader(
    training_dataset, batch_size=32, shuffle=True
).load()
testing_dataset = ImageClassificationDataset(
    "data/Classification_dataset_v3/images/test", transforms=images_transforms
)

testing_loader = ImageClassificationLoader(
    testing_dataset, batch_size=32, shuffle=False
).load()

if model_name=="cnn":
    print("Start Training")
    model = CNNModel(224, 3)
    trainer = ClassificationTrainer(model, epochs=15)
    trainer.fit(training_loader, output_folder_path="output_results")
    trainer.testing(testing_loader)
if model_name=="vgg":
    model = VGG_Model(3)
    trainer = ClassificationTrainer(model, epochs=5)
    trainer.fit(training_loader, output_folder_path="output_results",file_name="vgg.pth")
    trainer.testing(testing_loader)
