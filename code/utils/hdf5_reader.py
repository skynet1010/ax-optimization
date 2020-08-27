import torch
import torchvision
import skimage.transform as transform
import numpy as np
import h5py
import os
import cv2

class Hdf5Dataset(torch.utils.data.Dataset):
    def __init__(self, in_file_name, training_mode, ds_kind,data_composition, model_key, transform=None):
        super(Hdf5Dataset, self).__init__()
        self.file = h5py.File(in_file_name, "r")
        self.root_ds_dir = "{}/".format(ds_kind)
        self.dir_dict = {"data":"fus_data","labels":"labels"}
        self.n_images, self.nx, self.ny, self.nz = self.file[self.root_ds_dir+self.dir_dict["data"]].shape
        self.transform = transform
        self.model_key = model_key

    def __getitem__(self, index):
        data={}
        data["imagery"] = self.file[self.root_ds_dir+self.dir_dict["data"]][index]
        data["labels"] = self.file[self.root_ds_dir+self.dir_dict["labels"]][index]
        return data

    def __len__(self):
        return self.n_images

