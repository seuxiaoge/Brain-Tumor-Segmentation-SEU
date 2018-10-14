import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import random

import SimpleITK as sitk


class Brats15DataLoader(Dataset):
    def __init__(self, data_dir, direction = 0, volume_size=16, num_class=4):
        self.data_dir = data_dir  #
        self.num_class = num_class
        self.img_lists = []

        self.volume_size = volume_size


        subjects = os.listdir(self.data_dir)
        for sub in subjects:
            if sub == '.DS_Store':
                continue
            self.img_lists.append(os.path.join(self.data_dir, sub))

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, item):
        subject = self.img_lists[item]
        files = os.listdir(subject)  # 5 dierctory

        for im in files:
            path = os.path.join(subject, im)  # absolute directory

            if 'Flair.' in im:
                img_name = path + '/' + im + '.mha'
                img = self.load_mha_as_array(img_name)

                continue

            if 'OT.' in im:
                label_name = path + '/' + im + '.mha'
                label = self.load_mha_as_array(label_name)
                labels = self.get_whole_tumor_labels(label)

        img, labels = self.get_volumes(img, labels)

        img = img[np.newaxis, :, :, :]
        labels = labels[np.newaxis, :, :, :]

        img = torch.from_numpy(img)
        labels = torch.from_numpy(labels)

        return img.float(), labels.float()

    def get_volumes(self, img, label):
        """
        get volume randomly
        :param img:
        :param label:
        :return:
        """

        start = np.random.randint(0, img.shape[0] - self.volume_size + 1)
        img = img[start: start +self.volume_size, :, :]
        label = label[start: start +self.volume_size, :, :]
        return img, label

    @staticmethod
    def get_whole_tumor_labels(label):
        """
        whole tumor in patient data is label 1 + 2 + 3 + 4
        :param label:  numpy array      size : 155 * 240 * 240  value 0-4
        :return:
        label 1 * 155 * 240 * 240
        """
        label = (label > 0) + 0  # label 1,2,3,4
        return label

    @staticmethod
    def load_mha_as_array(img_name):
        img = sitk.ReadImage(img_name)
        nda = sitk.GetArrayFromImage(img)
        return nda


if __name__ =="__main__":
    data_dir = '../data/train/'

    brats15 = Brats15DataLoader(data_dir=data_dir)
    img, label = brats15[0]
    print ('image size ......')
    print (img.shape)

    print ('label size ......')
    print (label.shape)







