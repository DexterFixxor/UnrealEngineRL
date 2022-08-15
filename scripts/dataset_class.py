import os
import PIL.Image as Image
import PIL.ImageShow
from matplotlib import pyplot as plt

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn.functional as F

from definitions import ROOT_DIR

class CustomDataset(Dataset):
    """
    CustomDataset class that loads list of all PNG and JPG images in given path (only the root folder).
    Images are transformed to given image_size, which must be of size 64 * n where n is power of 2, so
    supported image sizes are 64, 128, 256, 512...
    """
    def __init__(self, root, image_size : int, padding = None):
        super(CustomDataset).__init__()
        self.root = root
        self.img_list = [img for img in os.listdir(root) if img.endswith("png") or img.endswith("jpg")]
        self.image_size = image_size
        assert self.image_size % 64 == 0

        if type(padding) == int:
            self.padding = [padding, ]
        else:
            self.padding = padding  # [left/right, top/bottom] for len of 2

        transforms = []
        if padding is not None:
            transforms.append(T.Pad(padding=self.padding))

        transforms.extend([
            T.Resize(size=self.image_size),
            T.RandomCrop(size = self.image_size//2),
            T.ToTensor()
        ])
        self.transform = T.Compose( transforms )

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.root, self.img_list[item]))
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list)



if __name__ == "__main__":
    ds = CustomDataset(
        root= ROOT_DIR + "/dataset",
        image_size=512,
        padding= [0 ,abs(512-480)]
    )

    for img in zip(ds, range(10)):
       pass

    print(len(ds))

