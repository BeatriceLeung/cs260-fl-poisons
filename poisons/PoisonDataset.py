from torchvision.datasets import MNIST # https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html#MNIST
from pandas import read_csv
from PIL import Image
import os
import torch

class PoisonDataset(MNIST):
    def __init__(self, label_file, image_file, transform=None):
        # super(PoisonDataSet, self).__init__()
        self.root_dir = image_file
        self.annot = read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, index: int):
        img_name = os.path.join(self.root_dir,
                                self.annot.iloc[index, 0]) # csv: image_name,label\n
        print("Opening image: ", img_name)
        image = Image.open(img_name)
        image = image.convert('L')

        if self.transform:
            image = self.transform(image)

        # print(image.shape)

        return image, torch.tensor(self.annot.iloc[index, 1])

#https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
