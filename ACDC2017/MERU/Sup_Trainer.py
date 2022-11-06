# -*- coding:utf-8 -*-


import torch
from torch.utils import tensorboard
import tensorboardX as tb
import logging, os
from itertools import cycle
from  torch.optim.lr_scheduler import _LRScheduler
from losses import *
from evaluation import *
import torch.nn.functional as F

class Poly(_LRScheduler):

    def __init__(self, optimizer, num_epochs, iters_per_epoch, warmup_epochs=0, last_epoch=-1):

        self.iters_per_epoch = iters_per_epoch

        self.cur_iter = 0

        self.N = num_epochs * iters_per_epoch

        self.warmup_iters = warmup_epochs * iters_per_epoch

        super(Poly, self).__init__(optimizer, last_epoch)



    def get_lr(self):

        T = self.last_epoch * self.iters_per_epoch + self.cur_iter

        factor =  pow((1 - 1.0 * T / self.N), 0.9)

        if self.warmup_iters > 0 and T < self.warmup_iters:

            factor = 1.0 * T / self.warmup_iters



        self.cur_iter %= self.iters_per_epoch

        self.cur_iter += 1

        assert factor >= 0, 'error in lr_scheduler'

        return [base_lr * factor for base_lr in self.base_lrs]

class Trainer(object):
    def __init__(self,model,config,supervised_loader,unsupervised_loader,sup_loss_sep,sup_loss_fus,unsup_ul_negloss,sup_pi_loss,unsup_pi_loss,save_dir,vali_files,iter_per_epoch):
        self.model = model.cuda()
        #self.prior_net = priorPred_Net.cuda()
        self.optimizer = torch.optim.Adam([{'params':model.parameters()}],lr = config.lr,  weight_decay = 1e-4)
        self.lr_scheduler = Poly(optimizer=self.optimizer, num_epochs=config.epochs, iters_per_epoch=len(supervised_loader))
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.sup_loss_sep = sup_loss_sep
        self.sup_loss_fus = sup_loss_fus
        self.unsup_ul_negloss = unsup_ul_negloss
        self.sup_pi_loss = sup_pi_loss
        self.unsup_pi_loss = unsup_pi_loss
        #self.unsup_l_negloss = unsup_l_negloss
        self.sup_type = config.sup_loss_mode
        self.ignore_index = config.ignore_index
        self.sup_loss_w = config.sup_loss_w
        self.unsup_negloss_w = consistency_weight(final_w=config.unsup_negloss_w, iters_per_epoch=iter_per_epoch,
                                               rampup_starts=config.rampup_starts, rampup_ends=config.rampup_ends)
        self.unsup_pi_consis_w = consistency_weight(final_w=config.unsup_pi_consis_w, iters_per_epoch=iter_per_epoch,
                                               rampup_starts=0, rampup_ends=config.rampup_ends)
        self.neg_unsup_negloss_w = config.neg_unsup_negloss_w
        self.unsup_negloss_thresh = config.unsup_negloss_thresh
        self.iter_per_epoch = iter_per_epoch

        self.pu_mode = config.unsup_pu_mode

        self.epochs = config.epochs
        self.save_dir = save_dir
        self.logger = self._create_logger(save_dir)
        self.writer = tensorboard.SummaryWriter(os.path.join(save_dir,'tensorboard_writer'))
        self.wrt_step = 0

        self.vali_files = vali_files

        self.logger.info(f'\niters_per_epoch: {iter_per_epoch}, epochs:{self.epochs}')

        self.epochs_per_vali = config.epochs//5

        self.pi = [0.0,0.0,0.0,0.0]

        self.CELoss = nn.CrossEntropyLoss().cuda()

        self.MSELoss = nn.MSELoss().cuda()
        self.cls_id = config.cls_id

        self.sup_pi_w = config.sup_pi_w

        self.pi_regression_error = []
        self.pi_segaverage_error = []
        self.start_epoch = config.rampup_starts



    def train(self):

        for epoch in range(self.epochs):
            self.model.train()
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))

            for batch_idx in range(self.iter_per_epoch):

                (input_l, target_l, target_l_s2, target_l_s4,pi_l),(input_ul,_,_,_,pi_ul)= next(dataloader)
                input_l, target_l, target_l_s2, target_l_s4,pi_l, input_ul, pi_ul = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True), target_l_s2.cuda(non_blocking=True), target_l_s4.cuda(non_blocking=True),pi_l.cuda(non_blocking=True), input_ul.cuda(non_blocking=True),pi_ul.cuda(non_blocking=True)

                self.optimizer.zero_grad()

                # pi
                # self.pi[0] = self.pi[0] + torch.sum((target_l==0)*1).item()
                # self.pi[1] = self.pi[1] + torch.sum((target_l == 1) * 1).item()
                # self.pi[2] = self.pi[2] + torch.sum((target_l == 2) * 1).item()
                # self.pi[3] = self.pi[3] + torch.sum((target_l == 3) * 1).item()

                # supervised
                output_s4_l_main, output_s2_l_main, output_l_main, output_fusion_l_main, pi_all_main, out5_main= self.model(input_l)
                #pi_single_main = self.prior_net(out5_main)

                l_prior_pred = self._predict_prior_hard(output_fusion_l_main)


                loss_supseg_main = self.sup_loss_sep(output_s4_l_main, target_l_s4)+ self.sup_loss_sep(output_s2_l_main, target_l_s2)+ self.sup_loss_sep(output_l_main, target_l)+ self.sup_loss_fus(output_fusion_l_main, target_l)
                # loss_suppi_mian = self.sup_pi_loss(pi_all_main,pi_l,use_softmax=False)#+self.sup_pi_loss(pi_single_main,pi_l,use_softmax=True)
                loss_sup_main = loss_supseg_main* self.sup_loss_w  #+loss_suppi_mian*self.sup_pi_w


                # unsupervised
                output_s4_ul_main, output_s2_ul_main, output_ul_main, output_fusion_ul_main,pi_ul_main,out5_ul_main = self.model(input_ul)
                #pi_ul_single_main = self.prior_net(out5_ul_main)

                ul_prior_pred = self._predict_prior_hard(output_fusion_ul_main)

                pi_ul_main_ = ul_prior_pred.clone().detach()


                loss_pu_list_main_s4= self.unsup_ul_negloss(output_s4_ul_main,output_s4_l_main,target_l_s4,pi_ul_main_,self.cls_id)
                loss_pu_list_main_s2= self.unsup_ul_negloss(output_s2_ul_main,output_s2_l_main,target_l_s2,pi_ul_main_,self.cls_id)
                loss_pu_list_main = self.unsup_ul_negloss(output_ul_main,output_l_main,target_l,pi_ul_main_,self.cls_id)

                #loss_suppi_ul_mian = self.unsup_pi_loss(pi_ul_single_main,pi_ul_main)
                #weight_ul_pi = self.unsup_pi_consis_w(epoch=epoch, curr_iter=batch_idx)

                l_pi_error1 = -torch.mean(pi_l[:, self.cls_id:self.cls_id + 1] - l_prior_pred[:, self.cls_id:self.cls_id + 1])
                ul_pi_error1 = -torch.mean(
                    pi_ul[:, self.cls_id:self.cls_id + 1] - ul_prior_pred[:, self.cls_id:self.cls_id + 1])

                self.pi_regression_error.append(ul_pi_error1.item())
                self.pi_segaverage_error.append(l_pi_error1.item())


                weight_pu = self.unsup_negloss_w(epoch=epoch, curr_iter=batch_idx)
                loss_pu = loss_pu_list_main_s4 + loss_pu_list_main_s2 + loss_pu_list_main

                if self.pu_mode=='pu':
                    loss_main = loss_sup_main

                else:

                    if loss_pu < self.unsup_negloss_thresh:
                        loss_main = self.neg_unsup_negloss_w * loss_pu  # -1

                    else:
                        loss_main = loss_sup_main + weight_pu * loss_pu

                # total loss
                loss_main.backward()

                self.optimizer.step()

                self.lr_scheduler.step(epoch=epoch - 1)

                # log
                if batch_idx% 10 == 0 :
                    self.logger.info(f'\nEpoch: {epoch} cur_iter: {batch_idx}, '
                                     f'\nmain net--- sup seg loss: {loss_sup_main},ul pi error:{ul_pi_error1.item(),l_pi_error1.item()}, unsup pu loss: {loss_pu}, total loss: {loss_main}')
                self.wrt_step = epoch*self.iter_per_epoch+ batch_idx
                self.writer.add_scalars(f'train/loss',{'sup loss':loss_sup_main ,'pu loss': loss_pu, 'total loss':loss_main},self.wrt_step)

                del input_l, target_l, input_ul

                del loss_main, loss_sup_main, loss_pu_list_main,loss_pu_list_main_s2,loss_pu_list_main_s4, output_s4_ul_main, output_s2_ul_main, output_ul_main, output_fusion_ul_main,output_s4_l_main, output_s2_l_main, output_l_main, output_fusion_l_main



            # evaluate
            #if epoch % self.epochs_per_vali==self.epochs_per_vali-1:
                #_, _,_,_ = ACDC2017_Evaluation(self.vali_files,self.model,epoch,self.save_dir)


                # save best model


        # test model , save results
        dice_dict, assd_dict, overlap_ndarray, surface_ndarray = ACDC2017_Evaluation(self.vali_files, self.model, self.epochs, self.save_dir)
        np.save(self.save_dir + '/test_dice_dict.npy', np.array(dice_dict))
        np.save(self.save_dir + '/test_assd_dict.npy', np.array(assd_dict))
        np.save(self.save_dir + '/test_overlap_ndarray.npy', overlap_ndarray)
        np.save(self.save_dir + '/test_surface_ndarray.npy', surface_ndarray)

        np.save(os.path.join(self.save_dir, 'pi_regression_error.npy'), np.array(self.pi_regression_error))
        np.save(os.path.join(self.save_dir, 'pi_segaverage_error.npy'), np.array(self.pi_segaverage_error))


        # save model
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_param.pkl'))
        #torch.save(self.prior_net.state_dict(), os.path.join(self.save_dir, 'priornet_param.pkl'))

        return dice_dict,assd_dict,overlap_ndarray,surface_ndarray



    def _create_logger(self,save_dir):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logfile = os.path.join(save_dir,'training.log')

        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)

        logger.addHandler(fh)

        return logger


    def _predict_prior_hard(self,prob_map):
        _, truearg0 = torch.max(prob_map, 1, keepdim=False)

        n,w,h = truearg0.size()

        pixel_count = torch.ones(n,4).cuda()

        for i in range(4):
            pixel_count[:,i] = torch.sum(truearg0==i,dim=(1,2))

        prior = pixel_count/(w*h)


        return prior













