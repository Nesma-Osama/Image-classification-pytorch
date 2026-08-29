import torch
import torch.optim as optim
import torch.nn as nn
import os

class ClassificationTrainer:
    def __init__(self, model, lr=0.001, epochs=5):
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def fit(self, loader, output_folder_path):
        for e in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(images)
                loss = self.loss(predictions, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss
            print(f"Epoch {e+1}/{self.epochs} loss {running_loss/len(loader)} ")
        path = os.path.join(output_folder_path, "cnn.pth")

        torch.save(self.model.state_dict(), path)

    def testing(self, loader):
        correct = 0.0
        total = 0.0
        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += images.size(0)
        print(f" Testing Result {correct/total}")
