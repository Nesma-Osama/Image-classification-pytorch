from torch.utils.data import DataLoader


class ImageClassificationLoader:
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load(self):
        return DataLoader(
            dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
