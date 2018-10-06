from model.unet3d import UNet3D

from data_loader.brats15 import Brats15DataLoader

import configparser


data = Brats15DataLoader(data_dir='data/train/')


net = UNet3D(in_ch=1, out_ch=4)

X, Label = data[0]
print X.shape

Y = net(X)
print Y.shape