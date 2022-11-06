# -*- coding:utf-8 -*-

import random, math
import glob
import numpy as np
import os
import SimpleITK as sitk
import torch
import cv2
import torch.nn.functional as F

from torch.utils.data import Dataset

from PIL import Image

from torchvision import transforms


def ACDC2017_Split(datapath,ratio):
    num_patients = 100
    num_per_class = 20
    train_num_per_class = 15
    np.random.seed(1)
    random_permu_lists = [np.random.permutation(np.arange(i*20+1,(i+1)*20+1)) for i in range(5)]

    # vali list
    Vali_list = []
    for i in range(5):
        Vali_list = Vali_list + random_permu_lists[i][-5:].tolist()

    sup_num_per_class = round(train_num_per_class*ratio)
    Sup_list = []
    Unsup_list = []
    for i in range(5):
        Sup_list = Sup_list + random_permu_lists[i][:sup_num_per_class].tolist()
        Unsup_list = Unsup_list + random_permu_lists[i][sup_num_per_class:-5].tolist()

    Vali_Files = []
    for id in Vali_list:
        Vali_Files = Vali_Files + glob.glob(datapath+'/*%03d'%(id)+'/*_frame??.nii*')

    Sup_Files = []
    for id in Sup_list:
        Sup_Files = Sup_Files + glob.glob(datapath + '/*%03d' % (id) + '/*_frame??.nii*')

    Unsup_Files = []
    for id in Unsup_list:
        Unsup_Files = Unsup_Files + glob.glob(datapath + '/*%03d' % (id) + '/*_frame??.nii*')

    return Sup_Files, Unsup_Files, Vali_Files

def obtain_filenames_from_id_list(datapath,id_list):
    Files = []
    for id in id_list:
        Files = Files + glob.glob(datapath + '/*%03d' % (id) + '/*_frame??.nii*')

    return Files

def FiveFold_ACDC2017_Split(datapath,ratio):
    num_patients = 100
    num_per_class = 20
    train_num_per_class = 16
    sup_num_per_class = round(train_num_per_class * ratio)
    np.random.seed(1)
    id_numpys = np.random.permutation(np.arange(1,101).reshape(5,20).transpose())
    Five_Labled_Train_Files, Five_Unlabeled_Train_Files, Five_Vali_Files = [],[],[]

    for cv in range(5):
        vali_id_list_cv = id_numpys[4*cv:4*(cv+1),:].reshape(-1).tolist()
        train_id_array_cv = np.random.permutation(np.delete(id_numpys,np.arange(4*cv,4*(cv+1)).tolist(),axis=0))
        labeled_train_id_list_cv = train_id_array_cv[0:sup_num_per_class,:].reshape(-1).tolist()
        unlabeled_train_id_list_cv = np.delete(train_id_array_cv,np.arange(0,sup_num_per_class).tolist(),axis=0).reshape(-1).tolist()

        vali_filenames_list = obtain_filenames_from_id_list(datapath,vali_id_list_cv)
        labeled_train_filenames_list = obtain_filenames_from_id_list(datapath,labeled_train_id_list_cv)
        unlabeled_train_filenames_list = obtain_filenames_from_id_list(datapath,unlabeled_train_id_list_cv)

        Five_Labled_Train_Files.append(labeled_train_filenames_list)
        Five_Unlabeled_Train_Files.append(unlabeled_train_filenames_list)
        Five_Vali_Files.append(vali_filenames_list)

    return Five_Labled_Train_Files, Five_Unlabeled_Train_Files, Five_Vali_Files





class ACDC2017_RandomCase(Dataset):
    def __init__(self,datapath,ratio,mode,resize,rotate,flip,crop,crop_size):
        Sup_ImgFiles, Unsup_ImgFiles, Vali_ImgFiles =  ACDC2017_Split(datapath,ratio)

        if mode == 'train_l':
            self.ImgFiles = Sup_ImgFiles
        elif mode == 'train_ul':
            self.ImgFiles = Unsup_ImgFiles
        else:
            self.ImgFiles = Vali_ImgFiles

        self.resize = resize
        self.rotate = rotate
        self.flip = flip
        self.crop = crop
        self.crop_size = crop_size

    def __len__(self):

        return len(self.ImgFiles)

    def __getitem__(self, item):
        itkimg = sitk.ReadImage(self.ImgFiles[item])
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,
        npimg = np.uint8((npimg-npimg.min())/(npimg.max()-npimg.min())*255)

        itklab = sitk.ReadImage(self.ImgFiles[item].replace('.nii', '_gt.nii'))
        nplab = sitk.GetArrayFromImage(itklab)

        # to PIL Image
        Img_PIL = None

        if self.resize:
            pass

        return None


class ACDC2017_RandomSlice(Dataset):
    def __init__(self, ImgFiles, rotate, flip, crop, crop_size):

        self.ImgFiles = ImgFiles

        self.ori_size = 128
        imgs = np.zeros((1, 128, 128))
        labs = np.zeros((1, 128, 128))

        for imgfile in self.ImgFiles:
            itkimg = sitk.ReadImage(imgfile)
            npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,
            npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
            npimg = (npimg-npimg.mean())/npimg.std()
            npimg = np.pad(npimg,((0,0),(50,50),(50,50)),'minimum')

            itklab = sitk.ReadImage(imgfile.replace('.nii', '_gt.nii'))
            nplab = sitk.GetArrayFromImage(itklab)
            nplab = np.pad(nplab, ((0, 0), (50, 50), (50, 50)), 'minimum')

            index = np.where(nplab != 0)

            npimg = npimg[:,
                   (np.min(index[1]) + np.max(index[1])) // 2 - 64:(np.min(index[1]) + np.max(index[1])) // 2 + 64,
                   (np.min(index[2]) + np.max(index[2])) // 2 - 64:(np.min(index[2]) + np.max(index[2])) // 2 + 64]
            nplab = nplab[:, (np.min(index[1]) + np.max(index[1])) // 2 - 64:(np.min(index[1]) + np.max(index[1])) // 2 + 64,
                  (np.min(index[2]) + np.max(index[2])) // 2 - 64:(np.min(index[2]) + np.max(index[2])) // 2 + 64]

            imgs = np.concatenate((imgs, npimg), axis=0)
            labs = np.concatenate((labs, nplab), axis=0)

        self.imgs = imgs[1:, :, :]
        self.labs = labs[1:, :, :]
        self.imgs = self.imgs.astype(np.float32)
        self.labs = self.labs.astype(np.uint8)

        self.rotate = rotate
        self.flip = flip
        self.crop = crop
        self.crop_size = crop_size


    def __len__(self):

        return self.imgs.shape[0]

    def __getitem__(self, item):
        img = self.imgs[item]
        lab = self.labs[item]


        if self.flip:
            if random.random() > 0.5:
                img = np.fliplr(img).copy()
                lab = np.fliplr(lab).copy()

            if random.random() > 0.5:
                img = np.flipud(img)
                lab = np.flipud(lab)

        if self.rotate:
            angle = random.randint(-10, 10)
            center = (self.ori_size / 2, self.ori_size/ 2)

            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            img = cv2.warpAffine(img, rot_matrix, (self.ori_size, self.ori_size), flags=cv2.INTER_CUBIC)  # , borderMode=cv2.BORDER_REFLECT)

            lab = cv2.warpAffine(lab, rot_matrix, (self.ori_size, self.ori_size),
                                   flags=cv2.INTER_NEAREST)  # ,  borderMode=cv2.BORDER_REFLECT)


        if self.crop:
            img = img[self.ori_size // 2-self.crop_size//2:self.ori_size // 2+self.crop_size//2,self.ori_size // 2-self.crop_size//2:self.ori_size // 2+self.crop_size//2]
            lab = lab[self.ori_size // 2 - self.crop_size // 2:self.ori_size // 2 + self.crop_size// 2,
                  self.ori_size // 2 - self.crop_size // 2:self.ori_size // 2 + self.crop_size //2]

        #img_s2 = cv2.resize(img, (self.crop_size//2, self.crop_size//2), interpolation=cv2.INTER_CUBIC)
        lab_s2 = cv2.resize(lab, (self.crop_size // 2, self.crop_size // 2), interpolation=cv2.INTER_NEAREST)

        #img_s4 = cv2.resize(img, (self.crop_size // 4, self.crop_size // 4), interpolation=cv2.INTER_CUBIC)
        lab_s4 = cv2.resize(lab, (self.crop_size // 4, self.crop_size // 4), interpolation=cv2.INTER_NEAREST)

        pi = torch.FloatTensor([np.sum((lab==0)*1.0),np.sum((lab==1)*1.0),np.sum((lab==2)*1.0),np.sum((lab==3)*1.0)])/(self.crop_size*self.crop_size)

        return torch.from_numpy(img).unsqueeze(0).type(dtype=torch.FloatTensor),torch.from_numpy(lab).type(dtype=torch.LongTensor),torch.from_numpy(lab_s2).type(dtype=torch.LongTensor),torch.from_numpy(lab_s4).type(dtype=torch.LongTensor),pi






