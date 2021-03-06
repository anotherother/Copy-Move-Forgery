import torch
from torch.utils.data import Dataset
import os
import argparse
import cv2
import numpy as np

class CoMoFodDataloader(Dataset):
    def __init__(self, datasetPath, imgSize):
        self.datasetPath = datasetPath
        self.images = list(filter(lambda x: x.split('_')[1] in ['F', 'O'], self.__get_all_files()))
        self.imgSize = imgSize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        # Load Image
        img = cv2.imread(os.path.join(self.datasetPath, image_name))
        # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.cv2.COLOR_RGBA2RGB)#[:,:,:3]

        mask = self.__get_image_mask(image_name)
        # resize to new shape
        img = self.__pad_resize(img, size=self.imgSize)
        mask = self.__pad_resize(mask,size=self.imgSize)[np.newaxis, :,:]

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()

    def __get_image_mask(self, img_name):

        """
        Masks description:
            "F" - forged image,
            "B" - binary mask (black and white mask),
            "M" - mask (colored mask),
            "O" - original image.

        :return:
        """
        if img_name.split('_')[1] == 'O':
            return np.zeros((self.imgSize, self.imgSize, 1), np.uint8)
        elif img_name.split('_')[1] == 'F':
            fullname =os.path.join(self.datasetPath, img_name.split('_')[0] + '_B.png')
            mask = cv2.imread(fullname, cv2.IMREAD_GRAYSCALE)
            return mask[:,:, np.newaxis]

    def __get_all_files(self):
        imageFiles = os.listdir(self.datasetPath)
        return list(filter(lambda x: x.split(".")[-1] in ['png', 'jpg'] and x.split('_')[1] in ['F', 'O'], imageFiles))

    def __pad_resize(self, img, size=416, color=(127.5, 127.5, 127.5)):
        shape = img.shape[:2]
        ratio = float(size) / max(shape)  # ratio  = old / new
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
        dw = (size - new_shape[0]) / 2  # width padding
        dh = (size - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DataLoader')

    parser.add_argument("--dataset_path", default='/media/jacob/DATA_DRIVE1/DATA/IMG_FORGERY',
                            help='Path with data')
    parser.add_argument("--image_size", default=256,
                            help='size of input image')
    args = parser.parse_args()

    dataloader = CoMoFodDataloader(datasetPath=args.dataset_path, imgSize=args.image_size)
    for _ in range(100):
        img, mask = next(iter(dataloader))
