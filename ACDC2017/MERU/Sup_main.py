# -*- coding:utf-8 -*-


import argparse
import os
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from Sup_network import *
from Sup_Trainer import *
from losses import *
from dataset import *





def training_validation(configs):

    # init models
    #Seg_Net = ResSeg(configs.resnet1_type,configs.attention1_type,configs.Resencoder_out_channel,configs.num_class,configs.upscale,configs.pretrained,configs.c_scale)

    Sup_ImgFiles, Unsup_ImgFiles, Vali_ImgFiles = FiveFold_ACDC2017_Split(configs.datapath, configs.ratio)
    # loss param
    if configs.sup_loss_mode == 'BCE':
        sup_loss_sep = ACDC2017_SepNet_BCE_l
        sup_loss_fus = ACDC2017_SepNet_CE_l
    elif configs.sup_loss_mode == 'Sigmiod':
        sup_loss_sep = ACDC2017_SepNet_Sigmiod_Loss_l
        sup_loss_fus = nn.CrossEntropyLoss().cuda()#ACDC2017_SepNet_NLLoss_l #nn.CrossEntropyLoss().cuda()

    if configs.piloss_mode == 'mse':
        sup_pi_loss = softmax_mse_loss
        unsup_pi_loss = softmax_cons_mse_loss

    elif configs.piloss_mode == 'kl':
        sup_pi_loss = softmax_kl_loss
        unsup_pi_loss = softmax_js_loss




    if configs.unsup_negloss_mode == 'Sigmoid_negloss':
        unsup_negloss = ACDC2017_PULoss_Mean_SepNet_negloss_ul

    elif configs.unsup_negloss_mode == 'Sigmoid_posloss':
        unsup_negloss = ACDC2017_PULoss_Mean_SepNet_posloss_ul

    elif configs.unsup_negloss_mode == 'Sigmoid_negloss_posloss':
        unsup_negloss = ACDC2017_PULoss_Mean_SepNet_negloss_posloss_ul

    test_dice_dict = {}
    test_assd_dict = {}
    total_assd_np = np.zeros((1, 4))
    total_dice_np = np.zeros((1, 4))

    cur_path = os.path.abspath(os.curdir)
    SAVE_DIR_Fix = cur_path+'/'+'thresh{}-negw{}/'.format(configs.unsup_negloss_thresh,configs.neg_unsup_negloss_w)
    if not os.path.exists(SAVE_DIR_Fix):
        # os.mkdir(SAVE_DIR)
        os.makedirs(SAVE_DIR_Fix)

    for cv in range(5):
        Seg_Net = UNet2d()
        #priorPred_Net = PriorPredNet()

        # prepare dataset loader

        supervised_dataset = ACDC2017_RandomSlice(Sup_ImgFiles[cv], configs.rotate, configs.flip, configs.crop,
                                                  configs.crop_size)
        supervised_loader = DataLoader(supervised_dataset, batch_size=configs.BatchSize_l, shuffle=True, num_workers=20,
                                       pin_memory=True,drop_last=True)

        unsupervised_dataset = ACDC2017_RandomSlice(Unsup_ImgFiles[cv], configs.rotate, configs.flip, configs.crop,
                                                    configs.crop_size)
        unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=configs.BatchSize_ul, shuffle=True,
                                         num_workers=20, pin_memory=True,drop_last=True)

        iter_per_epoch = len(
            unsupervised_loader)  # ratio<0.5, using unsupervised_loader, ratio>0.5, using supervised_loader

        configs.epochs = configs.Iter_Num // iter_per_epoch


        # save dir
        SAVE_DIR = cur_path+'/'+'thresh{}-negw{}/'.format(configs.unsup_negloss_thresh,configs.neg_unsup_negloss_w)+'cv{}'.format(cv)
        if not os.path.exists(SAVE_DIR):
            #os.mkdir(SAVE_DIR)
            os.makedirs(SAVE_DIR)

        # training
        trainer = Trainer(Seg_Net,configs,supervised_loader,unsupervised_loader,sup_loss_sep,sup_loss_fus,unsup_negloss,sup_pi_loss,unsup_pi_loss,SAVE_DIR,Vali_ImgFiles[cv],iter_per_epoch)
        dice_dict,assd_dict,dice_ny,assd_ny=trainer.train()

        test_dice_dict.update(dice_dict)
        test_assd_dict.update(assd_dict)
        total_dice_np = np.concatenate((total_dice_np, dice_ny[:,:,1]), axis=0)
        total_assd_np = np.concatenate((total_assd_np, assd_ny[:,:,1]), axis=0)

        del Seg_Net,supervised_dataset,supervised_loader,unsupervised_dataset,unsupervised_loader

    np.save(SAVE_DIR_Fix + '/test_dice_dict.npy', np.array(test_dice_dict))
    np.save(SAVE_DIR_Fix + '/test_assd_dict.npy', np.array(test_assd_dict))

    mean_avghausdorff = np.mean(total_assd_np[1:], axis=0)
    std_avghausdorff = np.std(total_assd_np[1:], axis=0)

    mean_itkdice = np.mean(total_dice_np[1:], axis=0)
    std_itkdice = np.std(total_dice_np[1:], axis=0)

    with open("%s/fivefold_testout_index.txt" % SAVE_DIR_Fix, "a") as f:
        f.writelines(["meanavghausdorff:", "", str(mean_avghausdorff.tolist()), "stdavghausdorff:", "",
                      str(std_avghausdorff.tolist()), \
                      "", "meanitkdice:", "", str(mean_itkdice.tolist()), "stditkdice:", "",
                      str(std_itkdice.tolist())])


def main(configs):

    # unsup_negloss_modes = ['Sigmoid_posloss',] # 'prob','sigmoid','probsigmoid','logprob','Sigmoid_posloss','Sigmoid_negloss_posloss'
    # unsup_negloss_ws = [1e0,]
    # neg_unsup_negloss_ws = [-1e1,-1e0]
    #
    #
    #
    # cls_ids = [0,]
    # unsup_negloss_threshs = [0.05,0.1,-0.05,-0.1]



    # for unsup_negloss_mode in unsup_negloss_modes:
    #     for cls_id in cls_ids:
    #         for neg_unsup_negloss_w in neg_unsup_negloss_ws:
    #             for unsup_negloss_w in unsup_negloss_ws:
    #                 for unsup_negloss_thresh in unsup_negloss_threshs:
    #
    #                     configs.unsup_negloss_w = unsup_negloss_w
    #                     configs.neg_unsup_negloss_w = neg_unsup_negloss_w
    #                     configs.unsup_negloss_thresh = unsup_negloss_thresh
    #
    #
    #                     configs.unsup_negloss_mode = unsup_negloss_mode
    #                     configs.cls_id = cls_id

    training_validation(configs)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True

    ## set parameters
    parser = argparse.ArgumentParser()

    # network param
    parser.add_argument('--resnet1_type', type=str, default='deepbase_resnet50_dilated8')
    parser.add_argument('--input_channel', type=int, default=1, help='channel number of imput image data')
    parser.add_argument('--num_class', type=int, default=4, help='number of classes')
    parser.add_argument('--upscale', type=int, default=8, help='times of decoder upscaling the features to the same size of the input image')
    parser.add_argument('--attention1_type', type=str, default='no_attention')  #'no_attention', 'channel_attention','spatial_attention','self_attention'
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--Resencoder_out_channel', type=int, default=2048, help='number of Resencoder_out_channel')
    parser.add_argument('--c_scale', type=int, default=1, help='downscale of Resencoder_out_channel')

    # train param
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--Iter_Num', type=int, default=12000)
    parser.add_argument('--sup_loss_mode', type=str, default='Sigmiod')  # 'prob','probsigmoid', 'CE', 'abCE'
    parser.add_argument('--ignore_index', type=int, default=-100)
    parser.add_argument('--sup_loss_w', type=float, default=1.0)
    parser.add_argument('--BatchSize_l', type=int, default=20)
    parser.add_argument('--BatchSize_ul', type=int, default=20)
    parser.add_argument('--unsup_negloss_mode', type=str, default='Sigmoid_posloss')   # pu loss mode: 'prob', 'probsigmoid','sigmoid','logprob','prob_weighted','prob_mean','prob_mean_igback','prob_mean_balanced'
    parser.add_argument('--unsup_negloss_w', type=float, default=1.0)
    parser.add_argument('--unsup_negloss_thresh', type=float, default=0.0)
    parser.add_argument('--neg_unsup_negloss_w', type=float, default= -1.0)
    parser.add_argument('--unsup_pu_mode', type=str, default='nnpu') # 'pu' or 'nnpu'
    parser.add_argument('--rampup_starts', type=int, default=25)
    parser.add_argument('--rampup_ends', type=int, default=50)
    parser.add_argument('--cls_id', type=int, default=0)
    #parser.add_argument('--pi_mode', type=str, default='soft')
    parser.add_argument('--sup_pi_w', type=int, default=1e1)
    parser.add_argument('--unsup_pi_consis_w', type=int, default=1e0)
    parser.add_argument('--piloss_mode', type=str, default='kl')


    # dataset param
    parser.add_argument('--datapath', type=str, default='/home/wfp/2020-Semi-PU/Datasets/ACDC2017/training')
    parser.add_argument('--ratio', type=float, default=0.0625)  # labeled data / training data
    parser.add_argument('--rotate', type=bool, default=True)
    parser.add_argument('--flip', type=bool, default=True)
    parser.add_argument('--crop', type=bool, default=True)
    parser.add_argument('--crop_size', type=int, default=96)




    CONFIGs = parser.parse_args()

    main(CONFIGs)