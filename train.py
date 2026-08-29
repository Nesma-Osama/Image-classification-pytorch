import os
from src.datasets import ImageClassificationDataset
from src.dataloader import ImageClassificationLoader
from src.model import CNNModel
from src.trainer import ClassificationTrainer
from torchvision import transforms


os.makedirs("output_results",exist_ok=True)
images_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
training_dataset = ImageClassificationDataset(
    "data/Classification_dataset_v3/images/train",
    output_folder_path="output_results",
    transforms=images_transforms,
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


model = CNNModel(224, 3)

trainer = ClassificationTrainer(model, epochs=40)
trainer.fit(training_loader, output_folder_path="output_results")
trainer.testing(testing_loader)
