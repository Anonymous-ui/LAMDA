from torchvision.datasets import VisionDataset
from tqdm import tqdm
import PIL.Image as Image
import numpy as np
from torchvision import transforms
import os

class ImageNet(VisionDataset):
    def __init__(self, root, train=True, transform=None):
        super(ImageNet, self).__init__(root, transform=transform)
        self.root = root
        self.train = train

        self.data = []
        self.targets = []

        if self.train:
            self.root_folder_path = os.path.join(self.root, 'train')
        else:
            self.root_folder_path = os.path.join(self.root, 'val')
        label_names = os.listdir(self.root_folder_path)
        self.n_cls = len(label_names)
        self.img_size = 224
        print(f'''\nLoading ImageNetSubset--{'train' if self.train else 'test'} dataset...''')

        with tqdm(total=len(label_names)) as _tqdm:
            for label_int in range(len(label_names)):
                label_path = os.path.join(self.root_folder_path, label_names[label_int])
                num_skipped_images = 0
                for file_name in os.listdir(label_path):
                    full_file_path = os.path.join(label_path, file_name)
                    img = Image.open(full_file_path)
                    img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
                    img = np.asarray(img)
                    if len(img.flatten()) == self.img_size * self.img_size:
                        num_skipped_images += 1
                        continue
                    img = img.reshape((1, self.img_size, self.img_size, 3))
                    self.data.append(img)
                    self.targets.append(label_int)
                _tqdm.set_postfix(num_skipped_imgs=f'{num_skipped_images}')
                _tqdm.update(1)
        self.data = np.vstack(self.data)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)