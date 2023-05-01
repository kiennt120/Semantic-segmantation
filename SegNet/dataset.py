import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BioImageDataset(Dataset):
    def __init__(self, image_path, test=False, transform=None):
        img_list = os.listdir(image_path)
        self.test_mode = test
        self.imgs_path = []
        for i in img_list:
            if i[:5] == 'image':
                self.imgs_path.append([os.path.join(image_path, i), os.path.join(image_path, 'mask' + i[5:])])
        self.transform = transform

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_path = self.imgs_path[index][0]
        mask_path = self.imgs_path[index][1]
        img = Image.open(image_path)
        mask_img = Image.open(mask_path)
        if self.transform:
            img = self.transform(img)
        img = transforms.ToTensor()(img)
        mask_img = transforms.ToTensor()(mask_img)
        sample = {
            'image': img,
            'label': mask_img,
        }
        return sample
