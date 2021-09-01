from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from collections import namedtuple
import torchvision.transforms as transforms

from Params import Params


DataPoint = namedtuple('DataPoint', ('filename', 'category'))
ImageSuffixes = set([
    '.jpg', '.png', '.bmp', '.tiff'
])


class SubdirImg_dataset(Dataset):
    def __init__(self, phase):
        self.all_data = []

        split_paths = {
            'train': Params.TrainingSet,
            'test': Params.TestSet,
            'valid': Params.ValidationSet
        }
        base_path = Path( split_paths[phase] )

        subdirs = [entry for entry in base_path.iterdir() if entry.is_dir()]
        subdirs.sort(key=lambda e: e.name)

        for category_idx, subdir in enumerate(subdirs):
            new_data = [
                DataPoint(
                    filename = str(filepath),
                    category = category_idx
                )
                for filepath in subdir.iterdir()
                if filepath.suffix in ImageSuffixes
            ]
            self.all_data += new_data

        self.transform = transforms.Compose( [
            transforms.Resize(Params.CropSize),
            transforms.RandomCrop( (Params.CropSize, Params.CropSize) ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ] )

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        fname = self.all_data[idx].filename
        image = Image.open( fname ).convert('RGB')
        image = self.transform(image)
        return { 'image': image, 'category': self.all_data[idx].category }
