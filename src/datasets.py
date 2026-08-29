import os
from torch.utils.data import Dataset
from PIL import Image
import pickle


class ImageClassificationDataset(Dataset):
    def __init__(self, folder_path, output_folder_path=None, transforms=None):
        self.images = []
        self.labels = []
        self.transforms = transforms
        self.class_name_mapping = {}
        for label, class_name in enumerate(os.listdir(folder_path)):
            self.class_name_mapping[label] = class_name
            class_dir_path = os.path.join(folder_path, class_name)
            for image_name in os.listdir(class_dir_path):
                self.labels.append(label)
                image_path = os.path.join(class_dir_path, image_name)
                self.images.append(image_path)
        if output_folder_path:
            path = os.path.join(output_folder_path, "class_name_mapping.pkl")
            with open(path,"wb")as f:
                pickle.dump(self.class_name_mapping, f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, label
