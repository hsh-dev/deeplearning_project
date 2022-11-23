import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data


def imshow(img):    # function to show an image
    img = img.cpu().detach()
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=( 4, 8 ))
    plt.imshow(np.transpose(npimg, ( 1, 2, 0 )))
    plt.axis("off")
    plt.show()


def imshow_double(
    img1, img2, file_name="result", show=True, save=False
):    # function to show an image
    img1 = img1.cpu().detach()
    img1 = img1 / 2 + 0.5
    npimg = img1.numpy()
    plt.figure(figsize=( 8, 4 ))

    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(npimg, ( 1, 2, 0 )))

    img2 = img2.cpu().detach()
    img2 = img2 / 2 + 0.5
    npimg = img2.numpy()
    plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(npimg, ( 1, 2, 0 )))

    plt.axis("off")

    if show:
        plt.show()

    if save:
        save_file_name = file_name + str(".png")
        plt.savefig(save_file_name, dpi=300)


class CustomDataSet(Dataset):

    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)

        return tensor_image


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)