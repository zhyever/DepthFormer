from torch.utils.data import Dataset
import os

from torchvision.transforms.transforms import ToTensor
import skimage
import torchvision.transforms as T
import random

class WXYDataset(Dataset):
    """WXY dataset."""

    def __init__(self, data_path, mode='train', transform=None):
        self.transform = transform
        self.data_path = data_path
        print("Dataset path: {}".format(self.data_path))
        self.load_data()
        self.totensor = T.ToTensor()
        self.mode = mode

    def load_data(self):
        # walk over all the data folder
        # 1. data very large: over your momery. -> Cannot read all of them directly.
        # 2. data very small: read all. xxx
        # path mark
        
        self.img_path = []
        self.img_label = []

        random_sample_idx = random.random() # random -> idxes
        
        if self.mode=='train':
            for file in os.listdir(self.data_path)[0:10000]:
                # save image path
                self.img_path.append(os.path.join(self.data_path, file))

                # save image label
                class_name = file.split(".")[0]
                if class_name == 'cat':
                    label = 0
                elif class_name == 'dog':
                    label = 1
                else:
                    raise NotImplementedError
                self.img_label.append(label)

        elif self.mode=='val':
            for file in os.listdir(self.data_path)[10001:10001+2000]:
                # save image path
                self.img_path.append(os.path.join(self.data_path, file))

                # save image label
                class_name = file.split(".")[0]
                if class_name == 'cat':
                    label = 0
                elif class_name == 'dog':
                    label = 1
                else:
                    raise NotImplementedError
                self.img_label.append(label)
    
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = skimage.io.imread(img_path)
        img = self.totensor(img)
        label = self.img_label[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class Normalize():
    def __call__(self, img):
        img = 2 * (img / 255) - 1

if __name__ == '__main__':
    data_path = '/home/lizhenyu1/catdog'

    # strong aug for training 
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        Normalize(),
        T.RandomVerticalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5)
        ])

    # normal aug for val and test
    transform_easy = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        Normalize(),
        ])

    my_dataset_train = WXYDataset(data_path, transform) ### sampler val
    my_dataset_test = WXYDataset(data_path, transform_easy)

