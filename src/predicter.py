import torch
from PIL import Image
import pickle

class Predicter:
    def __init__(self, model, transforms, model_path,class_mapping_path):
        self.transforms = transforms
        self.model = model.to(self.device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        with open(class_mapping_path,"rb")as f:
            self.class_mapping=pickle.load(f)
    def predict(self, image_path):
        self.model.eval()
        with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            image = self.transforms(image).unsqueeze(0).to(self.device)
            output = self.model(image)
            predict = torch.argmax(output, dim=1)
            return self.class_mapping[predict.item()]
