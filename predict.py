import os
from src.predicter import Predicter
from src.cnn_model import CNNModel
from torchvision import transforms
from src.vgg_model import VGG_Model

model_name=input("Enter model name (vgg/cnn) ")
print(model_name)
images_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ]
)
class_mapping_path = os.path.join("output_results", "class_name_mapping.pkl")
if model_name=="cnn":
    model_path=os.path.join("output_results","cnn.pth")
    model = CNNModel(224, 3)
    predictor=Predicter(model=model,transforms=images_transforms,model_path=model_path,class_mapping_path=class_mapping_path)
    print(predictor.predict("data/Classification_dataset_v3/images/test/person/person_2011.jpg"))
if model_name=="vgg":
    model_path=os.path.join("output_results","vgg.pth")
    model = VGG_Model(3)
    predictor=Predicter(model=model,transforms=images_transforms,model_path=model_path,class_mapping_path=class_mapping_path)
    print(predictor.predict("data/Classification_dataset_v3/images/test/person/person_2011.jpg"))