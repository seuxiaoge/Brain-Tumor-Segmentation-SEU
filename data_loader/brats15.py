import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np

import math
import SimpleITK as sitk


class Brats15DataLoader(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir  #
        self.img_lists = []
        subjects = os.listdir(self.data_dir)
        for sub in subjects:
            self.img_lists.append(os.path.join(self.data_dir, sub))

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, item):
        subject = self.img_lists[item]
        files = os.listdir(subject)  # 5 dierctory

        for im in files:
            if 'T1.' in im:
                img_name = os.path.join(subject, im)  # absolute directory
                img_name = img_name + '/' + im + '.mha'
                img = self.load_mha_as_array(img_name)
                continue

            if 'OT.' in im:
                label_name = os.path.join(subject, im)  # absolute directory
                label_name = label_name + '/' + im + '.mha'
                label = self.load_mha_as_array(label_name)

        return img, label

    def load_mha_as_array(self, img_name):
        img = sitk.ReadImage(img_name)
        nda = sitk.GetArrayFromImage(img)

        return nda


if __name__ =="__main__":
    data_dir = '../data/train/'

    brats15 = Brats15DataLoader(data_dir=data_dir)
    img, label = brats15[0]
    print 'image size ......'
    print img.shape
    print 'image max value ......'
    print np.max(img), np.min(img)

    print 'label size ......'
    print label.shape
    print 'label max value ......'
    print np.max(label), np.min(label)

