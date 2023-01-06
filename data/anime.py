import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import utils.image


class AnimeDataset(Dataset):
    def __init__(
        self,
        path,
    ):
        self.image_path = path
        self.tf_reference = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        self.tf_condition = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.filenames = []
        for filename in os.listdir(self.image_path):
            if utils.image.is_image(filename) and os.access(os.path.join(self.image_path, filename), os.R_OK):
                self.filenames.append(filename)
        # get the number of channles via a flag image
        self.num_channel = 3
        if len(self.filenames) > 0:
            flag_image = Image.open(os.path.join(self.image_path, self.filenames[0]))
            self.num_channel = min(self.num_channel, len(flag_image.split()))

    def __getitem__(self, index):
        ret = {}
        filename = self.filenames[index]
        img = Image.open(os.path.join(self.image_path, filename))
        if self.num_channel == 3:
            img = img.convert('RGB')
            ret['image'] = self.tf_reference(img)
        else:
            ret['image'] = self.tf_condition(img)
        ret['name'] = filename
        return ret

    def __len__(self):
        return len(self.filenames)