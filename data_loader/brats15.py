import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np

import math
import SimpleITK as sitk


class Brats15DataLoader(Dataset):
    def __init__(self, data_dir, num_class=4, train=True):
        self.data_dir = data_dir  #
        self.num_class = num_class
        self.img_lists = []
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
                img = img[np.newaxis, :, :, :]
                continue

            if 'OT.' in im:
                label_name = path + '/' + im + '.mha'
                label = self.load_mha_as_array(label_name)
                labels = self.get_labels(label)

        img = torch.from_numpy(img)
        label = torch.from_numpy(labels)

        return img.float(), label.float()

    def get_labels(self, label):
        """
        generate one hot label
        :param label: 155 * 240 * 240
        :return:  4 * 155 * 240 * 240
        """

        labels = []
        for i in range(1, self.num_class+1):
            tmp = (label == i) + 0
            labels.append(tmp)

        return np.asarray(labels)

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
    print img.shape

    print ('label size ......')
    print label.shape

    a = np.asarray(label == 1)





