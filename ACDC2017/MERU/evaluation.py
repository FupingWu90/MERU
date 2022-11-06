# -*- coding:utf-8 -*-

import torch
from torch import nn
import os
import math
import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob

import torchvision
import torchvision.transforms as transforms
import time
import scipy.misc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random








def dice_compute(pred, groundtruth):           #batchsize*channel*W*W
    # for j in range(pred.shape[0]):
    #     for i in range(pred.shape[1]):
    #         if np.sum(pred[j,i,:,:])==0 and np.sum(groundtruth[j,i,:,:])==0:
    #             pred[j, i, :, :]=pred[j, i, :, :]+1
    #             groundtruth[j, i, :, :]=groundtruth[j,i,:,:]+1
    #
    # dice = 2*np.sum(pred*groundtruth,axis=(2,3),dtype=np.float16)/(np.sum(pred,axis=(2,3),dtype=np.float16)+np.sum(groundtruth,axis=(2,3),dtype=np.float16))
    dice=[]
    for i in range(4):
        dice_i = 2*(np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)+0.0001)
        dice=dice+[dice_i]


    return np.array(dice,dtype=np.float32)




def IOU_compute(pred, groundtruth):
    iou=[]
    for i in range(4):
        iou_i = (np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)/(np.sum(pred==i,dtype=np.float32)+np.sum(groundtruth==i,dtype=np.float32)-np.sum((pred==i)*(groundtruth==i),dtype=np.float32)+0.0001)
        iou=iou+[iou_i]


    return np.array(iou,dtype=np.float32)


def Hausdorff_compute(pred,groundtruth,num_class,spacing):
    pred = np.squeeze(pred)
    groundtruth = np.squeeze(groundtruth)

    ITKPred = sitk.GetImageFromArray(pred, isVector=False)
    ITKPred.SetSpacing(spacing)
    ITKTrue = sitk.GetImageFromArray(groundtruth, isVector=False)
    ITKTrue.SetSpacing(spacing)

    overlap_results = np.zeros((1,num_class, 5))
    surface_distance_results = np.zeros((1,num_class, 5))

    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for i in range(num_class):
        pred_i = (pred==i).astype(np.float32)
        if np.sum(pred_i)==0:
            overlap_results[0,i,:]=0
            surface_distance_results[0,i,:]=0
        else:
            # Overlap measures
            overlap_measures_filter.Execute(ITKTrue==i, ITKPred==i)
            overlap_results[0,i, 0] = overlap_measures_filter.GetJaccardCoefficient()
            overlap_results[0,i, 1] = overlap_measures_filter.GetDiceCoefficient()
            overlap_results[0,i, 2] = overlap_measures_filter.GetVolumeSimilarity()
            overlap_results[0,i, 3] = overlap_measures_filter.GetFalseNegativeError()
            overlap_results[0,i, 4] = overlap_measures_filter.GetFalsePositiveError()
            # Hausdorff distance
            hausdorff_distance_filter.Execute(ITKTrue==i, ITKPred==i)

            surface_distance_results[0,i, 0] = hausdorff_distance_filter.GetHausdorffDistance()
            # Symmetric surface distance measures

            reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKTrue == i, squaredDistance=False, useImageSpacing=True))
            reference_surface = sitk.LabelContour(ITKTrue == i)
            statistics_image_filter = sitk.StatisticsImageFilter()
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(reference_surface)
            num_reference_surface_pixels = int(statistics_image_filter.GetSum())

            segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(ITKPred==i, squaredDistance=False, useImageSpacing=True))
            segmented_surface = sitk.LabelContour(ITKPred==i)
            # Get the number of pixels in the reference surface by counting all pixels that are 1.
            statistics_image_filter.Execute(segmented_surface)
            num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

            # Multiply the binary surface segmentations with the distance maps. The resulting distance
            # maps contain non-zero values only on the surface (they can also contain zero on the surface)
            seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
            ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

            # Get all non-zero distances and then add zero distances if required.
            seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
            seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
            seg2ref_distances = seg2ref_distances + \
                                list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
            ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
            ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
            ref2seg_distances = ref2seg_distances + \
                                list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

            all_surface_distances = seg2ref_distances + ref2seg_distances

            # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
            # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
            # segmentations, though in our case it is. More on this below.
            surface_distance_results[0,i, 1] = np.mean(all_surface_distances)
            surface_distance_results[0,i, 2] = np.median(all_surface_distances)
            surface_distance_results[0,i, 3] = np.std(all_surface_distances)
            surface_distance_results[0,i, 4] = np.max(all_surface_distances)


    return overlap_results,surface_distance_results


def multi_dice_iou_compute(pred,label):
    truemax, truearg = torch.max(pred, 1, keepdim=False)
    truearg = truearg.detach().cpu().numpy()
    # nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, \
    #                    truearg == 4, truearg == 5, truearg == 6, truearg == 7), 1)
    nplabs = np.stack((truearg == 0, truearg == 1, truearg == 2, truearg == 3, truearg == 4, truearg == 5), 1)
    # truelabel = (truearg == 0) * 550 + (truearg == 1) * 420 + (truearg == 2) * 600 + (truearg == 3) * 500 + \
    #             (truearg == 4) * 250 + (truearg == 5) * 850 + (truearg == 6) * 820 + (truearg == 7) * 0

    dice = dice_compute(nplabs, label.cpu().numpy())
    Iou = IOU_compute(nplabs, label.cpu().numpy())

    return dice,Iou


class BalancedBCELoss(nn.Module):
    def __init__(self,target):
        super(BalancedBCELoss,self).__init__()
        self.eps=1e-6
        weight = torch.tensor([torch.reciprocal(torch.sum(target==0).float()+self.eps),torch.reciprocal(torch.sum(target==1).float()+self.eps),torch.reciprocal(torch.sum(target==2).float()+self.eps),torch.reciprocal(torch.sum(target==3).float()+self.eps)])
        self.criterion = nn.CrossEntropyLoss(weight)

    def forward(self, output,target):
        loss = self.criterion(output,target)

        return loss






def ACDC2017_Evaluation(testfile_list, model, epoch, save_DIR):
    num_class = 4

    total_overlap = np.zeros((1, num_class, 5))
    total_surface_distance = np.zeros((1, num_class, 5))
    dice_dict = {}
    assd_dict = {}

    model.eval()
    for imgfile in testfile_list:
        itkimg = sitk.ReadImage(imgfile)
        npimg = sitk.GetArrayFromImage(itkimg)  # Z,Y,X,
        npimg = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        npimg = (npimg - npimg.mean()) / npimg.std()
        npimg_pad = np.pad(npimg, ((0, 0), (50, 50), (50, 50)), 'minimum')

        itklab = sitk.ReadImage(imgfile.replace('.nii', '_gt.nii'))
        nplab = sitk.GetArrayFromImage(itklab)
        nplab_pad = np.pad(nplab, ((0, 0), (50, 50), (50, 50)), 'minimum')

        index = np.where(nplab_pad != 0)

        npimg_crop = npimg_pad[:,
                (np.min(index[1]) + np.max(index[1])) // 2 - 64:(np.min(index[1]) + np.max(index[1])) // 2 + 64,
                (np.min(index[2]) + np.max(index[2])) // 2 - 64:(np.min(index[2]) + np.max(index[2])) // 2 + 64]

        npimg_crop = npimg_crop.astype(np.float32)

        data = torch.from_numpy(np.expand_dims(npimg_crop, axis=1)).type(dtype=torch.FloatTensor).cuda()

        truearg_crop = np.zeros_like(npimg_crop,dtype=np.uint8)
        truearg_pad = np.zeros_like(npimg_pad,dtype=np.uint8)

        for slice in range(data.size(0)):
            _,_,_,output,_,_= model(data[slice:slice + 1, :, :, :])

            truemax, truearg0 = torch.max(output, 1, keepdim=False)
            truearg_crop[slice:slice + 1, :, :] = truearg0.detach().cpu().numpy()

        truearg_pad[:,
                (np.min(index[1]) + np.max(index[1])) // 2 - 64:(np.min(index[1]) + np.max(index[1])) // 2 + 64,
                (np.min(index[2]) + np.max(index[2])) // 2 - 64:(np.min(index[2]) + np.max(index[2])) // 2 + 64]=truearg_crop

        truearg = truearg_pad[:,50:-50,50:-50]

        overlap_result, surface_distance_result = Hausdorff_compute(truearg, nplab,num_class, itkimg.GetSpacing())

        total_overlap = np.concatenate((total_overlap, overlap_result), axis=0)
        total_surface_distance = np.concatenate((total_surface_distance, surface_distance_result), axis=0)

        dice_dict[imgfile.replace('.nii', '_gt.nii').split('/')[-1]] = overlap_result[0,:,1]
        assd_dict[imgfile.replace('.nii', '_gt.nii').split('/')[-1]] = surface_distance_result[0, :, 1]


    mean_overlap = np.mean(total_overlap[1:], axis=0)
    std_overlap = np.std(total_overlap[1:], axis=0)

    mean_surface_distance = np.mean(total_surface_distance[1:], axis=0)
    std_surface_distance = np.std(total_surface_distance[1:], axis=0)

    meanDice = np.mean(total_overlap[1:,:,1], axis=0)
    meanAssd = np.mean(total_surface_distance[1:,:,1], axis=0)


    with open("%s/evaluation_index.txt" % (save_DIR), "a") as f:
        f.writelines(["\n\nepoch:", str(epoch)," ", "mean dice:", str(meanDice.tolist())," ", "mean assd:",str(meanAssd.tolist()),
                      "\n\n", "jaccard, dice, volume_similarity, false_negative, false_positive:", "\n",
                      "mean:", str(mean_overlap.tolist()), "\n", "std:", "", str(std_overlap.tolist()), \
                      "", "\n\n",
                      "hausdorff_distance, mean_surface_distance, median_surface_distance, std_surface_distance, max_surface_distance:",
                      "\n", "mean:", str(mean_surface_distance.tolist()), "\n", "std:",
                      str(std_surface_distance.tolist())])
    return dice_dict, assd_dict, total_overlap[1:], total_surface_distance[1:]