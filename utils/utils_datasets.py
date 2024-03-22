import os
from torch.utils.data import Dataset
from skimage import metrics
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from utils import *
from torchvision import transforms


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        if args.task == 'SR':
            self.dataset_dir = args.path_for_train + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.scale_factor) + 'x/'
        elif args.task == 'RE':
            self.dataset_dir = args.path_for_train + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                               str(args.angRes_out) + 'x' + str(args.angRes_out) + '/'
            pass

        if args.data_name == 'ALL':
            self.data_list = os.listdir(self.dataset_dir)
        else:
            self.data_list = [args.data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)


    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y')) # Lr_SAI_y
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y')) # Hr_SAI_y
            Lr_SAI_y, Hr_SAI_y, Lr_y, Hr_y = augmentation(Lr_SAI_y, Hr_SAI_y)
            Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
            Lr_y = ToTensor()(Lr_y.copy())
            Hr_y = ToTensor()(Hr_y.copy())
            # ==== crop to 5x5 =========

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out

        return Lr_SAI_y, Hr_SAI_y, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args, half_res=False):
    # get testdataloader of every test dataset
    data_list = None
    if half_res is False:
        if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
            if args.task == 'SR':
                dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                            str(args.scale_factor) + 'x/'
                data_list = os.listdir(dataset_dir)
            elif args.task == 'RE':
                dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                            str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
                data_list = os.listdir(dataset_dir)
        else:
            data_list = [args.data_name]
    else:
        if args.data_name in ['ALL', 'RE_Lytro', 'RE_HCI']:
            if args.task == 'SR':
                dataset_dir = args.path_for_test + 'SR_' + '5x5' + '_' + \
                            str(args.scale_factor) + 'x/'
                data_list = os.listdir(dataset_dir)
            elif args.task == 'RE':
                dataset_dir = args.path_for_test + 'RE_' + '5x5' + '_' + \
                            str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name
                data_list = os.listdir(dataset_dir)
        else:
            data_list = [args.data_name]

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=data_list.index(data_name), half_res=half_res)
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name = 'ALL', Lr_Info=None, half_res=False):
        super(TestSetDataLoader, self).__init__()
        if half_res is True:
            self.angRes_in = 5
            self.angRes_out = 5
        else:
            self.angRes_in = args.angRes_in
            self.angRes_out = args.angRes_out
        if half_res is False:
            if args.task == 'SR':
                self.dataset_dir = args.path_for_test + 'SR_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                                str(args.scale_factor) + 'x/'
                self.data_list = [data_name]
            elif args.task == 'RE':
                self.dataset_dir = args.path_for_test + 'RE_' + str(args.angRes_in) + 'x' + str(args.angRes_in) + '_' + \
                                str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
                self.data_list = [data_name]
        else:
            if args.task == 'SR':
                self.dataset_dir = args.path_for_test + 'SR_' + '5x5' + '_' + \
                                str(args.scale_factor) + 'x/'
                self.data_list = [data_name]
            elif args.task == 'RE':
                self.dataset_dir = args.path_for_test + 'RE_' + '5x5' + '_' + \
                                str(args.angRes_out) + 'x' + str(args.angRes_out) + '/' + args.data_name + '/'
                self.data_list = [data_name]

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('Lr_SAI_y'))
            Hr_SAI_y = np.array(hf.get('Hr_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('Sr_SAI_cbcr'), dtype='single')
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        Lr_angRes_in = self.angRes_in
        Lr_angRes_out = self.angRes_out
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    if random.random() < 0.5:
        k = random.randint(1, 3)
        data = np.rot90(data, k)
        label = np.rot90(label, k)

    patch = 32
    # ix = random.randint(0, 4)
    # iy = random.randint(0, 4)
    ix = 2
    iy = 2

    patch_data = data[ix*patch:ix*patch+160, iy*patch:iy*patch+160]
    patch_label = label[4*ix*patch:4*ix*patch+640, 4*iy*patch:4*iy*patch+640]

    # if random.random() < 0.5:
    #     num_list = range(8)
    #     row_list = sorted(random.sample(num_list, 5))
    #     row_count = 0
    #     for i in row_list:
    #         col_list = sorted(random.sample(num_list, 5))
    #         col_count = 0
    #         for j in col_list:
    #             patch_data[row_count*patch:(row_count+1)*patch, col_count*patch:(col_count+1)*patch] = data[i*patch:(i+1)*patch, j*patch:(j+1)*patch]
    #             patch_label[4*row_count*patch:4*(row_count+1)*patch, 4*col_count*patch:4*(col_count+1)*patch] = label[4*i*patch:4*(i+1)*patch, 4*j*patch:4*(j+1)*patch]
    #             col_count += 1
    #         row_count += 1

    return patch_data, patch_label, data, label

