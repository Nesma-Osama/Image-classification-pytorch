import os
from src.predicter import Predicter
from src.model import CNNModel
from torchvision import transforms

images_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
model_path=os.path.join("output_results","cnn.pth")
model = CNNModel(224, 3)
class_mapping_path = os.path.join("output_results", "class_name_mapping.pkl")
predictor=Predicter(model=model,transforms=images_transforms,model_path=model_path,class_mapping_path=class_mapping_path)
print(predictor.predict("data/Classification_dataset_v3/images/test/person/person_2011.jpg"))