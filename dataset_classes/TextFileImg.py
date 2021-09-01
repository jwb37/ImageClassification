from PIL import Image
import os.path as path
from torch.utils.data import Dataset
from collections import namedtuple
import torchvision.transforms as transforms

from Params import Params


DataPoint = namedtuple('DataPoint', ('filename', 'category'))


class TextFileImg_dataset(Dataset):
    def __init__(self, phase):
        textfile_path = path.join(Params.TextFileDir, f"{phase}.txt")

        with open( textfile_path, 'r' ) as in_file:
            self.all_data = [
                DataPoint(
                    filename = line.split()[0],
                    category = int(line.split()[1])
                )
                for line in in_file
            ]

        self.transform = transforms.Compose( [
            transforms.Resize(Params.CropSize),
            transforms.RandomCrop( (Params.CropSize, Params.CropSize) ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ] )


    def __len__(self):
        return len(self.all_data)


    def __getitem__(self, idx):
        img_filename = path.join(Params.ImageDir, self.all_data[idx].filename )
        image = Image.open( img_filename ).convert('RGB')
        image = self.transform(image)
        return { 'image': image, 'category': self.all_data[idx].category }
