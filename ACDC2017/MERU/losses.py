import numpy as np

import torch

import torch.nn.functional as F

import torch.nn as nn

import rampers


class consistency_weight(object):
    """

    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']

    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=7, ramp_type='sigmoid_rampup'):
        self.final_w = final_w

        self.iters_per_epoch = iters_per_epoch

        self.rampup_starts = rampup_starts * iters_per_epoch

        self.rampup_ends = rampup_ends * iters_per_epoch

        self.rampup_length = (self.rampup_ends - self.rampup_starts)

        self.rampup_func = getattr(rampers, ramp_type)

        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter

        if cur_total_iter < self.rampup_starts:
            return 0

        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)

        return self.final_w * self.current_rampup


def CE_loss(input_logits, target_targets, ignore_index, temperature=1):
    return F.cross_entropy(input_logits / temperature, target_targets, ignore_index=ignore_index)


class abCE_loss(nn.Module):
    """

    Annealed-Bootstrapped cross-entropy loss

    """

    def __init__(self, iters_per_epoch, epochs, num_classes, weight=None,

                 reduction='mean', thresh=0.7, min_kept=1, ramp_type='log_rampup'):

        super(abCE_loss, self).__init__()

        self.weight = torch.FloatTensor(weight) if weight is not None else weight

        self.reduction = reduction

        self.thresh = thresh

        self.min_kept = min_kept

        self.ramp_type = ramp_type

        if ramp_type is not None:
            self.rampup_func = getattr(rampers, ramp_type)

            self.iters_per_epoch = iters_per_epoch

            self.num_classes = num_classes

            self.start = 1 / num_classes

            self.end = 0.9

            self.total_num_iters = (epochs - (0.6 * epochs)) * iters_per_epoch

    def threshold(self, curr_iter, epoch):

        cur_total_iter = self.iters_per_epoch * epoch + curr_iter

        current_rampup = self.rampup_func(cur_total_iter, self.total_num_iters)

        return current_rampup * (self.end - self.start) + self.start

    def forward(self, predict, target, ignore_index, curr_iter, epoch):

        batch_kept = self.min_kept * target.size(0)

        prob_out = F.softmax(predict, dim=1)

        tmp_target = target.clone()

        tmp_target[tmp_target == ignore_index] = 0

        prob = prob_out.gather(1, tmp_target.unsqueeze(1))

        mask = target.contiguous().view(-1, ) != ignore_index

        sort_prob, sort_indices = prob.contiguous().view(-1, )[mask].contiguous().sort()

        if self.ramp_type is not None:

            thresh = self.threshold(curr_iter=curr_iter, epoch=epoch)

        else:

            thresh = self.thresh

        min_threshold = sort_prob[min(batch_kept, sort_prob.numel() - 1)] if sort_prob.numel() > 0 else 0.0

        threshold = max(min_threshold, thresh)

        loss_matrix = F.cross_entropy(predict, target,

                                      weight=self.weight.to(predict.device) if self.weight is not None else None,

                                      ignore_index=ignore_index, reduction='none')

        loss_matirx = loss_matrix.contiguous().view(-1, )

        sort_loss_matirx = loss_matirx[mask][sort_indices]

        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]

        if self.reduction == 'sum' or select_loss_matrix.numel() == 0:

            return select_loss_matrix.sum()

        elif self.reduction == 'mean':

            return select_loss_matrix.mean()

        else:

            raise NotImplementedError('Reduction Error!')


def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False

    assert inputs.size() == targets.size()



    if use_softmax:
        inputs = F.softmax(inputs, dim=1)

    if conf_mask:

        loss_mat = F.mse_loss(inputs, targets, reduction='none')

        mask = (targets.max(1)[0] > threshold)

        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]

        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)

        return loss_mat.mean()

    else:

        return F.mse_loss(inputs, targets, reduction='mean')

def softmax_cons_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False

    assert inputs.size() == targets.size()


    inputs = F.softmax(inputs, dim=1)

    if conf_mask:

        loss_mat = F.mse_loss(inputs, targets, reduction='none')

        mask = (targets.max(1)[0] > threshold)

        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]

        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)

        return loss_mat.mean()

    else:

        return torch.mean((inputs-targets)**2)


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False

    assert inputs.size() == targets.size()
    epsilon = 1e-5

    if use_softmax:
        input_log_softmax = F.log_softmax(inputs, dim=1)

    else:
        input_log_softmax = torch.log(inputs + epsilon)

    if conf_mask:

        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')

        mask = (targets.max(1)[0] > threshold)

        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]

        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)

        return loss_mat.sum() / mask.shape.numel()

    else:

        return F.kl_div(input_log_softmax, targets, reduction='mean')


def softmax_js_loss(inputs, targets, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False

    assert inputs.size() == targets.size()

    if use_softmax:
        targets = F.softmax(targets, dim=1)

    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5

    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')

    kl2 = F.kl_div(torch.log(targets + epsilon), M, reduction='mean')

    return (kl1 + kl2) * 0.5


def pair_wise_loss(unsup_outputs, size_average=True, nbr_of_pairs=8):
    """

    Pair-wise loss in the sup. mat.

    """

    if isinstance(unsup_outputs, list):
        unsup_outputs = torch.stack(unsup_outputs)

    # Only for a subset of the aux outputs to reduce computation and memory

    unsup_outputs = unsup_outputs[torch.randperm(unsup_outputs.size(0))]

    unsup_outputs = unsup_outputs[:nbr_of_pairs]

    temp = torch.zeros_like(unsup_outputs)  # For grad purposes

    for i, u in enumerate(unsup_outputs):
        temp[i] = F.softmax(u, dim=1)

    mean_prediction = temp.mean(0).unsqueeze(0)  # Mean over the auxiliary outputs

    pw_loss = ((temp - mean_prediction) ** 2).mean(0)  # Variance

    pw_loss = pw_loss.sum(1)  # Sum over classes

    if size_average:
        return pw_loss.mean()

    return pw_loss.sum()

# class Neg_ul_Loss(nn.Module):
#     def __init__(self,target):
#         super(Neg_ul_Loss,self).__init__()
#         self.C = 4
#         self.criterion = nn.NLLLoss()
#         self.u_target = []
#         self.soft = nn.Softmax2d()
#         for i in range(self.C):
#             u_target_i  = i* torch.ones_like(target)
#             self.u_target.append(u_target_i)
#
#     def forward(self, output):
#         output = -self.soft(output)
#
#         u_loss_n0 = self.criterion(output,self.u_target[0])
#         u_loss_n1 = self.criterion(output, self.u_target[1])
#         u_loss_n2 = self.criterion(output, self.u_target[2])
#         u_loss_n3 = self.criterion(output, self.u_target[3])
#
#
#         loss = u_loss_n0+ u_loss_n1+ u_loss_n2+ u_loss_n3
#
#         return loss
# class NegSegLoss(nn.Module):
#     def __init__(self,):
#         super(MinSegLoss,self).__init__()
#         self.criterion = nn.NLLLoss()
#         self.soft = nn.Softmax2d()
#
#     def forward(self, output,lab):
#         output = -self.soft(output)
#
#         loss = self.criterion(output,lab)
#
#         return loss

def ACDC2017_Neg_ul_ProbLoss(output):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss(ignore_index=0,reduction = 'sum').cuda()
    num_cls = 4
    u_target = []
    for i in range(num_cls):
        u_target_i = i * torch.ones(output.size(0),output.size(2),output.size(3),dtype=torch.long)
        u_target.append(u_target_i.cuda())

    # compute loss
    output = -S(output)
    #neg_loss_0 = criterion(output,u_target[0])
    neg_loss_1 = criterion(output, u_target[1])
    neg_loss_2 = criterion(output, u_target[2])
    neg_loss_3 = criterion(output, u_target[3])

    neg_loss =  neg_loss_1 + neg_loss_2 + neg_loss_3

    return neg_loss


def Neg_l_ProbLoss(output,target):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss(ignore_index=0,reduction = 'sum').cuda()

    # compute loss
    output = -S(output)
    neg_loss = criterion(output,target)

    return neg_loss


def ACDC2017_Neg_ul_ProbLoss_Background(output):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss(reduction = 'sum').cuda()
    num_cls = 4
    u_target = []
    for i in range(num_cls):
        u_target_i = i * torch.ones(output.size(0),output.size(2),output.size(3),dtype=torch.long)
        u_target.append(u_target_i.cuda())

    # compute loss
    output = -S(output)
    neg_loss_0 = criterion(output,u_target[0])


    return neg_loss_0


def Neg_l_ProbLoss_Background(output,target):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss(ignore_index=1,reduction = 'sum').cuda()
    target = 1-(target==0)*1

    # compute loss
    output = -S(output)
    neg_loss = criterion(output,target)

    return neg_loss

def ACDC2017_Neg_ul_ProbLoss_Seperate(ul_output, l_output, l_target,pi):
    S = nn.Softmax2d().cuda()
    ul_output = S(ul_output)
    l_output = S(l_output)

    loss_list = []
    neg_index = []
    weight_list = []

    # pu loss for class 0
    neg_loss_ul_0 = torch.sum(ul_output[:,0,:,:])/ul_output.size(0)

    l_target_0 = (l_target==0)*1
    weight_0 = l_output.size(0)/float(torch.sum(1-l_target_0)+1)
    neg_loss_l_0 = torch.sum(l_output[:,0,:,:]*l_target_0)/l_output.size(0)

    pu_loss_0 = neg_loss_ul_0 - neg_loss_l_0

    loss_list.append(pu_loss_0)
    neg_index.append((pu_loss_0<0)*1)
    weight_list.append(weight_0)

    # pu loss for class 1
    neg_loss_ul_1 = torch.sum(ul_output[:, 1, :, :]) / ul_output.size(0)

    l_target_1 = (l_target == 1) * 1
    weight_1 = l_output.size(0) / float(torch.sum(1 - l_target_1)+1)
    neg_loss_l_1 = torch.sum(l_output[:, 1, :, :] * l_target_1) / l_output.size(0)

    pu_loss_1 = neg_loss_ul_1 - neg_loss_l_1

    loss_list.append(pu_loss_1)
    neg_index.append((pu_loss_1 < 0) * 1)
    weight_list.append(weight_1)

    # pu loss for class 2
    neg_loss_ul_2 = torch.sum(ul_output[:, 2, :, :]) / ul_output.size(0)

    l_target_2 = (l_target == 2) * 1
    weight_2 = l_output.size(0) / float(torch.sum(1 - l_target_2)+1)
    neg_loss_l_2 = torch.sum(l_output[:, 2, :, :] * l_target_2) / l_output.size(0)

    pu_loss_2 = neg_loss_ul_2 - neg_loss_l_2

    loss_list.append(pu_loss_2)
    neg_index.append((pu_loss_2 < 0) * 1)
    weight_list.append(weight_2)

    # pu loss for class 3
    neg_loss_ul_3 = torch.sum(ul_output[:, 3, :, :]) / ul_output.size(0)

    l_target_3 = (l_target == 3) * 1
    weight_3 = l_output.size(0) / float(torch.sum(1 - l_target_3)+1)
    neg_loss_l_3 = torch.sum(l_output[:, 3, :, :] * l_target_3) / l_output.size(0)

    pu_loss_3 = neg_loss_ul_3 - neg_loss_l_3

    loss_list.append(pu_loss_3)
    neg_index.append((pu_loss_3 < 0) * 1)
    weight_list.append(weight_3)

    return loss_list, neg_index, weight_list


def ACDC2017_Neg_ul_ProbLoss_Seperate_Mean(ul_output, l_output, l_target, pi):
    S = nn.Softmax2d().cuda()
    ul_output = S(ul_output)
    l_output = S(l_output)

    loss_list = []
    neg_index = []
    weight_list = []
    pi_list = [pi[0]/sum(pi),pi[1]/sum(pi),pi[2]/sum(pi),pi[3]/sum(pi)]

    #print(pi_list)

    # pu loss for class 0
    neg_loss_ul_0 = torch.mean(ul_output[:,0,:,:])

    l_target_0 = (l_target==0)*1
    neg_loss_l_0 = torch.sum(l_output[:,0,:,:]*l_target_0)/torch.sum(l_target_0)

    pu_loss_0 = neg_loss_ul_0 - neg_loss_l_0*pi_list[0]

    loss_list.append(pu_loss_0)
    neg_index.append((pu_loss_0<0)*1)
    weight_list.append(1.0)

    # pu loss for class 1
    neg_loss_ul_1 = torch.mean(ul_output[:, 1, :, :])

    l_target_1 = (l_target == 1) * 1
    neg_loss_l_1 = torch.sum(l_output[:, 1, :, :] * l_target_1)/torch.sum(l_target_1)

    pu_loss_1 = neg_loss_ul_1 - neg_loss_l_1*pi_list[1]

    loss_list.append(pu_loss_1)
    neg_index.append((pu_loss_1 < 0) * 1)
    weight_list.append(1.0)

    # pu loss for class 2
    neg_loss_ul_2 = torch.mean(ul_output[:, 2, :, :])

    l_target_2 = (l_target == 2) * 1
    neg_loss_l_2 = torch.sum(l_output[:, 2, :, :] * l_target_2)/torch.sum(l_target_2)

    pu_loss_2 = neg_loss_ul_2 - neg_loss_l_2*pi_list[2]

    loss_list.append(pu_loss_2)
    neg_index.append((pu_loss_2 < 0) * 1)
    weight_list.append(1.0)

    # pu loss for class 3
    neg_loss_ul_3 = torch.mean(ul_output[:, 3, :, :])

    l_target_3 = (l_target == 3) * 1
    neg_loss_l_3 = torch.sum(l_output[:, 3, :, :] * l_target_3)/torch.sum(l_target_3)

    pu_loss_3 = neg_loss_ul_3 - neg_loss_l_3*pi_list[3]

    loss_list.append(pu_loss_3)
    neg_index.append((pu_loss_3 < 0) * 1)
    weight_list.append(1.0)

    return loss_list, neg_index, weight_list

def ACDC2017_Neg_ul_ProbLoss_Seperate_Mean_Balanced(ul_output, l_output, l_target, pi):
    S = nn.Softmax2d().cuda()
    ul_output = S(ul_output)
    l_output = S(l_output)

    loss_list = []
    neg_index = []
    weight_list = []
    pi_list = [pi[0]/sum(pi),pi[1]/sum(pi),pi[2]/sum(pi),pi[3]/sum(pi)]

    #print(pi_list)

    # pu loss for class 0
    neg_loss_ul_0 = torch.mean(ul_output[:,0,:,:])

    l_target_0 = (l_target==0)*1
    neg_loss_l_0 = torch.sum(l_output[:,0,:,:]*l_target_0)/torch.sum(l_target_0)

    pu_loss_0 = neg_loss_ul_0 - neg_loss_l_0*pi_list[0]

    loss_list.append(pu_loss_0)
    neg_index.append((pu_loss_0<0)*1)
    weight_list.append(1.0/(1-pi_list[0]))

    # pu loss for class 1
    neg_loss_ul_1 = torch.mean(ul_output[:, 1, :, :])

    l_target_1 = (l_target == 1) * 1
    neg_loss_l_1 = torch.sum(l_output[:, 1, :, :] * l_target_1)/torch.sum(l_target_1)

    pu_loss_1 = neg_loss_ul_1 - neg_loss_l_1*pi_list[1]

    loss_list.append(pu_loss_1)
    neg_index.append((pu_loss_1 < 0) * 1)
    weight_list.append(1.0/(1-pi_list[1]))

    # pu loss for class 2
    neg_loss_ul_2 = torch.mean(ul_output[:, 2, :, :])

    l_target_2 = (l_target == 2) * 1
    neg_loss_l_2 = torch.sum(l_output[:, 2, :, :] * l_target_2)/torch.sum(l_target_2)

    pu_loss_2 = neg_loss_ul_2 - neg_loss_l_2*pi_list[2]

    loss_list.append(pu_loss_2)
    neg_index.append((pu_loss_2 < 0) * 1)
    weight_list.append(1.0/(1-pi_list[2]))

    # pu loss for class 3
    neg_loss_ul_3 = torch.mean(ul_output[:, 3, :, :])

    l_target_3 = (l_target == 3) * 1
    neg_loss_l_3 = torch.sum(l_output[:, 3, :, :] * l_target_3)/torch.sum(l_target_3)

    pu_loss_3 = neg_loss_ul_3 - neg_loss_l_3*pi_list[3]

    loss_list.append(pu_loss_3)
    neg_index.append((pu_loss_3 < 0) * 1)
    weight_list.append(1.0/(1-pi_list[3]))

    return loss_list, neg_index, weight_list

def Pos_Balanced_ProbLoss(output,target):
    S = nn.Softmax2d().cuda()
    output = S(output)

    target_0 = (target == 0) * 1
    loss_0 = torch.sum(output[:, 0, :, :] * target_0) / torch.sum(target_0)

    target_1 = (target == 1) * 1
    loss_1 = torch.sum(output[:, 1, :, :] * target_1) / torch.sum(target_1)

    target_2 = (target == 2) * 1
    loss_2 = torch.sum(output[:, 2, :, :] * target_2) / torch.sum(target_2)

    target_3 = (target == 3) * 1
    loss_3 = torch.sum(output[:, 3, :, :] * target_3) / torch.sum(target_3)


    return -1.0*(loss_0+ loss_1+ loss_2+ loss_3)

def ACDC2017_Neg_ul_SigmoidProbLoss_Seperate_Mean_Balanced(ul_output, l_output, l_target, pi,unsup_negloss_thresh):
    S = nn.Softmax2d().cuda()
    ul_output = S(ul_output)
    l_output = S(l_output)

    loss_list = []
    neg_index = []
    weight_list = []
    pi_list = [pi[0]/sum(pi),pi[1]/sum(pi),pi[2]/sum(pi),pi[3]/sum(pi)]

    #print(pi_list)

    # pu loss for class 0
    neg_loss_ul_0 = torch.mean(F.sigmoid(2*ul_output[:,0,:,:]-1))

    l_target_0 = (l_target==0)*1
    neg_loss_l_0 = torch.sum(F.sigmoid(2*l_output[:,0,:,:]-1)*l_target_0)/torch.sum(l_target_0)

    pu_loss_0 = neg_loss_ul_0 - neg_loss_l_0*pi_list[0]

    loss_list.append(pu_loss_0)
    neg_index.append((1.0/(1-pi_list[0])*pu_loss_0<unsup_negloss_thresh)*1)
    weight_list.append(1.0/(1-pi_list[0]))

    # pu loss for class 1
    neg_loss_ul_1 = torch.mean(F.sigmoid(2*ul_output[:, 1, :, :]-1))

    l_target_1 = (l_target == 1) * 1
    neg_loss_l_1 = torch.sum(F.sigmoid(2*l_output[:, 1, :, :]-1) * l_target_1)/torch.sum(l_target_1)

    pu_loss_1 = neg_loss_ul_1 - neg_loss_l_1*pi_list[1]

    loss_list.append(pu_loss_1)
    neg_index.append((1.0/(1-pi_list[1])*pu_loss_1 < unsup_negloss_thresh) * 1)
    weight_list.append(1.0/(1-pi_list[1]))

    # pu loss for class 2
    neg_loss_ul_2 = torch.mean(F.sigmoid(2*ul_output[:, 2, :, :]-1))

    l_target_2 = (l_target == 2) * 1
    neg_loss_l_2 = torch.sum(F.sigmoid(2*l_output[:, 2, :, :]-1) * l_target_2)/torch.sum(l_target_2)

    pu_loss_2 = neg_loss_ul_2 - neg_loss_l_2*pi_list[2]

    loss_list.append(pu_loss_2)
    neg_index.append((1.0/(1-pi_list[2])*pu_loss_2 < unsup_negloss_thresh) * 1)
    weight_list.append(1.0/(1-pi_list[2]))

    # pu loss for class 3
    neg_loss_ul_3 = torch.mean(F.sigmoid(2*ul_output[:, 3, :, :]-1))

    l_target_3 = (l_target == 3) * 1
    neg_loss_l_3 = torch.sum(F.sigmoid(2*l_output[:, 3, :, :]-1) * l_target_3)/torch.sum(l_target_3)

    pu_loss_3 = neg_loss_ul_3 - neg_loss_l_3*pi_list[3]

    loss_list.append(pu_loss_3)
    neg_index.append((1.0/(1-pi_list[3])*pu_loss_3 < unsup_negloss_thresh) * 1)
    weight_list.append(1.0/(1-pi_list[3]))

    return loss_list, neg_index, weight_list

def Pos_Balanced_SigmoidProbLoss(output,target):
    S = nn.Softmax2d().cuda()
    output = S(output)

    target_0 = (target == 0) * 1
    loss_0 = torch.sum(F.sigmoid(1-2*output[:, 0, :, :]) * target_0) / torch.sum(target_0)

    target_1 = (target == 1) * 1
    loss_1 = torch.sum(F.sigmoid(1-2*output[:, 1, :, :] )* target_1) / torch.sum(target_1)

    target_2 = (target == 2) * 1
    loss_2 = torch.sum(F.sigmoid(1-2*output[:, 2, :, :]) * target_2) / torch.sum(target_2)

    target_3 = (target == 3) * 1
    loss_3 = torch.sum(F.sigmoid(1-2*output[:, 3, :, :]) * target_3) / torch.sum(target_3)


    return loss_0+ loss_1+ loss_2+ loss_3



def ACDC2017_Neg_ul_ProbLoss_Seperate_Mean_IgnoreBackgorund(ul_output, l_output, l_target, pi):
    S = nn.Softmax2d().cuda()
    ul_output = S(ul_output)
    l_output = S(l_output)

    loss_list = []
    neg_index = []
    weight_list = []
    pi_list = [pi[0]/sum(pi),pi[1]/sum(pi),pi[2]/sum(pi),pi[3]/sum(pi)]

    # pu loss for class 0
    neg_loss_ul_0 = torch.mean(ul_output[:,0,:,:])

    l_target_0 = (l_target==0)*1
    neg_loss_l_0 = torch.sum(l_output[:,0,:,:]*l_target_0)/torch.sum(l_target_0)

    pu_loss_0 = neg_loss_ul_0 - neg_loss_l_0*pi_list[0]

    loss_list.append(pu_loss_0)
    neg_index.append((pu_loss_0<0)*1)
    weight_list.append(1.0)

    # pu loss for class 1
    neg_loss_ul_1 = torch.mean(ul_output[:, 1, :, :])

    l_target_1 = (l_target == 1) * 1
    neg_loss_l_1 = torch.sum(l_output[:, 1, :, :] * l_target_1)/torch.sum(l_target_1)

    neg_loss_l_1_0 = torch.sum(l_output[:, 1, :, :] * l_target_0)/torch.sum(l_target_0)

    pu_loss_1 = neg_loss_ul_1 - neg_loss_l_1*pi_list[1] - neg_loss_l_1_0*pi_list[0]

    loss_list.append(pu_loss_1)
    neg_index.append((pu_loss_1 < 0) * 1)
    weight_list.append(1.0)

    # pu loss for class 2
    neg_loss_ul_2 = torch.mean(ul_output[:, 2, :, :])

    l_target_2 = (l_target == 2) * 1
    neg_loss_l_2 = torch.sum(l_output[:, 2, :, :] * l_target_2)/torch.sum(l_target_2)

    neg_loss_l_2_0 = torch.sum(l_output[:, 2, :, :] * l_target_0)/torch.sum(l_target_0)

    pu_loss_2 = neg_loss_ul_2 - neg_loss_l_2*pi_list[2]- neg_loss_l_2_0*pi_list[0]

    loss_list.append(pu_loss_2)
    neg_index.append((pu_loss_2 < 0) * 1)
    weight_list.append(1.0)

    # pu loss for class 3
    neg_loss_ul_3 = torch.mean(ul_output[:, 3, :, :])

    l_target_3 = (l_target == 3) * 1
    neg_loss_l_3 = torch.sum(l_output[:, 3, :, :] * l_target_3)/torch.sum(l_target_3)

    neg_loss_l_3_0 = torch.sum(l_output[:, 3, :, :] * l_target_0)/torch.sum(l_target_0)

    pu_loss_3 = neg_loss_ul_3 - neg_loss_l_3*pi_list[3]- neg_loss_l_3_0*pi_list[0]

    loss_list.append(pu_loss_3)
    neg_index.append((pu_loss_3 < 0) * 1)
    weight_list.append(1.0)

    return loss_list, neg_index, weight_list


def ACDC2017_Var_PU_Seperate_Mean(ul_output, l_output, l_target, pi):
    S = nn.Softmax2d().cuda()
    ul_output = S(ul_output)
    l_output = S(l_output)

    loss_list = []
    neg_index = []
    weight_list = []
    pi_list = [pi[0]/sum(pi),pi[1]/sum(pi),pi[2]/sum(pi),pi[3]/sum(pi)]

    print(pi_list)

    # pu loss for class 0
    neg_loss_ul_0 = torch.mean(ul_output[:,0,:,:])

    l_target_0 = (l_target==0)*1
    neg_loss_l_0 = torch.mean(l_output[:,0,:,:]*l_target_0)

    pu_loss_0 = neg_loss_ul_0 - neg_loss_l_0*pi_list[0]

    loss_list.append(pu_loss_0)
    neg_index.append((pu_loss_0<0)*1)
    weight_list.append(1.0)

    # pu loss for class 1
    neg_loss_ul_1 = torch.mean(ul_output[:, 1, :, :])

    l_target_1 = (l_target == 1) * 1
    neg_loss_l_1 = torch.mean(l_output[:, 1, :, :] * l_target_1)

    pu_loss_1 = neg_loss_ul_1 - neg_loss_l_1*pi_list[1]

    loss_list.append(pu_loss_1)
    neg_index.append((pu_loss_1 < 0) * 1)
    weight_list.append(1.0)

    # pu loss for class 2
    neg_loss_ul_2 = torch.mean(ul_output[:, 2, :, :])

    l_target_2 = (l_target == 2) * 1
    neg_loss_l_2 = torch.mean(l_output[:, 2, :, :] * l_target_2)

    pu_loss_2 = neg_loss_ul_2 - neg_loss_l_2*pi_list[2]

    loss_list.append(pu_loss_2)
    neg_index.append((pu_loss_2 < 0) * 1)
    weight_list.append(1.0)

    # pu loss for class 3
    neg_loss_ul_3 = torch.mean(ul_output[:, 3, :, :])

    l_target_3 = (l_target == 3) * 1
    neg_loss_l_3 = torch.mean(l_output[:, 3, :, :] * l_target_3)

    pu_loss_3 = neg_loss_ul_3 - neg_loss_l_3*pi_list[3]

    loss_list.append(pu_loss_3)
    neg_index.append((pu_loss_3 < 0) * 1)
    weight_list.append(1.0)

    return loss_list, neg_index, weight_list

def ACDC2017_Neg_ul_LogProbLoss(output):
    epsilon = 1e-6
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()
    num_cls = 4
    u_target = []
    for i in range(num_cls):
        u_target_i = i * torch.ones(output.size(0),output.size(2),output.size(3),dtype=torch.long)
        u_target.append(u_target_i.cuda())

    # compute loss
    output = -S(output)
    neg_loss_0 = torch.log(criterion(output,u_target[0])+epsilon)
    neg_loss_1 = torch.log(criterion(output, u_target[1])+epsilon)
    neg_loss_2 = torch.log(criterion(output, u_target[2])+epsilon)
    neg_loss_3 = torch.log(criterion(output, u_target[3])+epsilon)

    neg_loss = neg_loss_0 + neg_loss_1 + neg_loss_2 + neg_loss_3

    return neg_loss


def Neg_l_LogProbLoss(output,target):
    epsilon = 1e-6
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()

    # compute loss
    output = -S(output)
    neg_loss = torch.log(criterion(output,target)+epsilon)

    return neg_loss


def ACDC2017_Neg_ul_SigmoidLoss(output):
    #S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()
    num_cls = 4
    u_target = []
    for i in range(num_cls):
        u_target_i = i * torch.ones(output.size(0), output.size(2), output.size(3),dtype=torch.long)
        u_target.append(u_target_i.cuda())

    # compute loss
    #output = S(output)
    output = torch.sigmoid(output)
    neg_loss_0 = -criterion(output, u_target[0])
    neg_loss_1 = -criterion(output, u_target[1])
    neg_loss_2 = -criterion(output, u_target[2])
    neg_loss_3 = -criterion(output, u_target[3])

    neg_loss = neg_loss_0 + neg_loss_1 + neg_loss_2 + neg_loss_3

    return neg_loss

def Neg_l_SigmoidLoss(output, target):
    #S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()

    # compute loss
    #output = S(output)
    output = torch.sigmoid(output)
    neg_loss = -criterion(output, target)

    return neg_loss

def ACDC2017_Neg_ul_ProbSigmoidLoss(output):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()
    num_cls = 4
    u_target = []
    for i in range(num_cls):
        u_target_i = i * torch.ones(output.size(0), output.size(2), output.size(3),dtype=torch.long)
        u_target.append(u_target_i.cuda())

    # compute loss
    output = S(output)
    output = torch.sigmoid(output)
    neg_loss_0 = -criterion(output, u_target[0])
    neg_loss_1 = -criterion(output, u_target[1])
    neg_loss_2 = -criterion(output, u_target[2])
    neg_loss_3 = -criterion(output, u_target[3])

    neg_loss = neg_loss_0 + neg_loss_1 + neg_loss_2 + neg_loss_3

    return neg_loss


def Neg_l_ProbSigmoidLoss(output, target):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()

    # compute loss
    output = S(output)
    output = torch.sigmoid(output)
    neg_loss = -criterion(output, target)

    return neg_loss


def Pos_l_ProbLoss(output, target):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss(reduction = 'sum').cuda()

    # compute loss
    output = S(output)
    pos_loss = criterion(output, target)

    return pos_loss

def Pos_l_SigmoidbLoss(output, target):
    #S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()

    # compute loss
    #output = S(output)
    output = torch.sigmoid(-output)
    pos_loss = -criterion(output, target)

    return pos_loss

def Pos_l_ProSigmoidbLoss(output, target):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()

    # compute loss
    output = S(output)
    output = torch.sigmoid(-output)
    pos_loss = -criterion(output, target)

    return pos_loss

def loss_ce(u_p):
    S = nn.Softmax(dim=1).cuda()

    LS = nn.LogSoftmax(dim=1).cuda()

    loss = S(u_p) * LS(u_p)

    loss = -torch.mean(loss)

    return loss


def ACDC2017_SepNet_BCE_l(pred, target):
    criterion = nn.BCELoss().cuda()
    sigmiod = nn.Sigmoid().cuda()

    pred = sigmiod(pred)

    Loss_0 = criterion(pred[:,0,:,:],(target==0)*1.0)
    Loss_1 = criterion(pred[:, 1, :, :], (target == 1) * 1.0)
    Loss_2 = criterion(pred[:, 2, :, :], (target == 2) * 1.0)
    Loss_3 = criterion(pred[:, 3, :, :], (target == 3) * 1.0)

    return Loss_0+ Loss_1+ Loss_2+ Loss_3

def ACDC2017_SepNet_CE_l(pred, target):
    criterion = nn.CrossEntropyLoss().cuda()

    return criterion(pred, target)

def ACDC2017_SepNet_Sigmiod_Loss_l(pred, target):
    sigmiod = nn.Sigmoid().cuda()

    # 0
    index_0 = 1.0-(target == 0) * 2.0
    loss_0 = torch.mean(sigmiod(pred[:, 0, :, :]* index_0))

    # 1
    index_1 = 1.0 - (target == 1) * 2.0
    loss_1 = torch.mean(sigmiod(pred[:, 1, :, :] * index_1))

    # 2
    index_2 = 1.0 - (target == 2) * 2.0
    loss_2 = torch.mean(sigmiod(pred[:, 2, :, :] * index_2))

    # 3
    index_3 = 1.0 - (target == 3) * 2.0
    loss_3 = torch.mean(sigmiod(pred[:, 3, :, :] * index_3))

    return loss_0 + loss_1 + loss_2 + loss_3

def ACDC2017_SepNet_NLLoss_l(pred, target):
    S = nn.Softmax2d().cuda()
    criterion = nn.NLLLoss().cuda()

    pred = S(pred)

    return criterion(pred, target)


def ACDC2017_PULoss_Mean_SepNet_negloss_ul(output_ul,output_l,target,pi_ul,cls_id):

    sigmiod = nn.Sigmoid().cuda()

    pi = torch.mean(pi_ul, dim=0)
    pi = torch.clamp(pi,1e-5,1.0-1e-5)

    # neg loss
    output_neg = sigmiod(output_ul)
    output_l_neg = sigmiod(output_l)

    #output_neg = torch.cat((output_ul_neg,output_l_neg),dim=0)

    normed_loss = 0


    if cls_id == 0:
        # 0
        index_0 = (target == 0) * 1.0
        index_0_n = 1 - index_0

        Norm_pos_0 = 1.0 - pi[0].item()
        Norm_neg_0 = pi[0].item()

        neg_ul_0 = torch.mean(output_neg[:, 0, :, :])
        neg_l_0 = torch.sum(output_l_neg[:, 0, :, :] * index_0) / torch.sum(index_0)
        neg_0 = neg_ul_0 - neg_l_0 * Norm_neg_0

        normed_loss = neg_0 * (1.0 / Norm_pos_0)

    if cls_id == 1:
        # 1
        index_1 = (target == 1) * 1.0
        index_1_n = 1 - index_1

        Norm_pos_1 = 1.0 - pi[1].item()
        Norm_neg_1 = pi[1].item()

        if Norm_pos_1 == 0:
            Norm_pos_1 = 1e-6

        if Norm_neg_1 == 0:
            Norm_neg_1 = 1e-6

        neg_ul_1 = torch.mean(output_neg[:, 1, :, :])
        neg_l_1 = torch.sum(output_l_neg[:, 1, :, :] * index_1) / torch.sum(index_1)
        neg_1 = neg_ul_1 - neg_l_1 * Norm_neg_1

        normed_loss = neg_1*(1.0/Norm_pos_1)

    if cls_id == 2:
        # 2
        index_2 = (target == 2) * 1.0
        index_2_n = 1 - index_2

        Norm_pos_2 = 1.0 - pi[2].item()
        Norm_neg_2 = pi[2].item()

        neg_ul_2 = torch.mean(output_neg[:, 2, :, :])
        neg_l_2 = torch.sum(output_l_neg[:, 2, :, :] * index_2) / torch.sum(index_2)
        neg_2 = neg_ul_2 - neg_l_2 * Norm_neg_2
        normed_loss = neg_2*(1.0/Norm_pos_2)

    if cls_id == 3:
        # 3
        index_3 = (target == 3) * 1.0
        index_3_n = 1 - index_3

        Norm_pos_3 = 1.0 - pi[3].item()
        Norm_neg_3 = pi[3].item()


        neg_ul_3 = torch.mean(output_neg[:, 3, :, :])
        neg_l_3 = torch.sum(output_l_neg[:, 3, :, :] * index_3) / torch.sum(index_3)
        neg_3 = neg_ul_3 - neg_l_3 * Norm_neg_3
        normed_loss = neg_3*(1.0/Norm_pos_3)

    return normed_loss


def ACDC2017_PULoss_Mean_SepNet_posloss_ul(output_ul,output_l,target,pi_ul,cls_id):

    sigmiod = nn.Sigmoid().cuda()

    pi = torch.mean(pi_ul, dim=0)
    pi = torch.clamp(pi, 1e-8, 1.0 - 1e-8)



    # pos loss
    output_pos = sigmiod((-1.0) * output_ul)
    output_l_pos = sigmiod((-1.0) * output_l)

    #output_pos = torch.cat((output_ul_pos, output_l_pos), dim=0)


    normed_loss = 0


    if cls_id == 0:
        # 0
        index_0 = (target == 0) * 1.0
        index_0_n = 1 - index_0

        Norm_pos_0 = 1.0 - pi[0].item()
        Norm_neg_0 =pi[0].item()

        pos_ul_0 = torch.mean(output_pos[:, 0, :, :])
        pos_l_0 = torch.sum(output_l_pos[:, 0, :, :] * index_0_n) / (torch.sum(index_0_n)+1e-8)
        pos_0 = pos_ul_0 - pos_l_0 * Norm_pos_0

        normed_loss = pos_0*(1.0/Norm_neg_0)

    if cls_id == 1:
        # 1
        index_1 = (target == 1) * 1.0
        index_1_n = 1 - index_1

        Norm_pos_1 = 1.0 - pi[1].item()
        Norm_neg_1 = pi[1].item()

        if Norm_pos_1 == 0:
            Norm_pos_1 = 1e-6

        if Norm_neg_1 == 0:
            Norm_neg_1 = 1e-6

        pos_ul_1 = torch.mean(output_pos[:, 1, :, :])
        pos_l_1 = torch.sum(output_l_pos[:, 1, :, :] * index_1_n) / torch.sum(index_1_n)
        pos_1 = pos_ul_1 - pos_l_1 * Norm_pos_1

        normed_loss = pos_1*(1.0/Norm_neg_1)

    if cls_id == 2:
        # 2
        index_2 = (target == 2) * 1.0
        index_2_n = 1 - index_2

        Norm_pos_2 = 1.0 - pi[2].item()
        Norm_neg_2 = pi[2].item()

        pos_ul_2 = torch.mean(output_pos[:, 2, :, :])
        pos_l_2 = torch.sum(output_l_pos[:, 2, :, :] * index_2_n) / torch.sum(index_2_n)
        pos_2 = pos_ul_2 - pos_l_2 * Norm_pos_2

        normed_loss = pos_2 * (1.0 / Norm_neg_2)

    if cls_id == 3:
        # 3
        index_3 = (target == 3) * 1.0
        index_3_n = 1 - index_3

        Norm_pos_3 = 1.0 - pi[3].item()
        Norm_neg_3 = pi[3].item()

        pos_ul_3 = torch.mean(output_pos[:, 3, :, :])
        pos_l_3 = torch.sum(output_l_pos[:, 3, :, :] * index_3_n) / torch.sum(index_3_n)
        pos_3 = pos_ul_3 - pos_l_3 * Norm_pos_3

        normed_loss = pos_3*(1.0/Norm_neg_3)

    return normed_loss

def ACDC2017_PULoss_Mean_SepNet_negloss_posloss_ul(output_ul,output_l,target,pi_ul,cls_id):

    sigmiod = nn.Sigmoid().cuda()

    pi = torch.mean(pi_ul,dim=0)
    pi = torch.clamp(pi, 1e-5, 1.0 - 1e-5)

    # neg loss
    output_neg = sigmiod(output_ul)
    output_l_neg = sigmiod(output_l)

    #output_neg = torch.cat((output_ul_neg,output_l_neg),dim=0)



    # pos loss
    output_pos = sigmiod((-1.0) * output_ul)
    output_l_pos = sigmiod((-1.0) * output_l)

    #output_pos = torch.cat((output_ul_pos, output_l_pos), dim=0)

    normed_loss = 0

    if cls_id == 0:
        # 0
        index_0 = (target == 0) * 1.0
        index_0_n = 1- index_0

        Norm_pos_0 = 1.0-pi[0].item()
        Norm_neg_0 = pi[0].item()


        pos_ul_0 = torch.mean(output_pos[:,0,:,:])
        pos_l_0 = torch.sum(output_l_pos[:,0,:,:]* index_0_n)/torch.sum(index_0_n)
        pos_0 = pos_ul_0 - pos_l_0*Norm_pos_0

        neg_ul_0 = torch.mean(output_neg[:,0,:,:])
        neg_l_0 = torch.sum(output_l_neg[:,0,:,:]* index_0)/torch.sum(index_0)
        neg_0 = neg_ul_0 - neg_l_0*Norm_neg_0

        normed_loss = pos_0*(1.0/Norm_neg_0) + neg_0*(1.0/Norm_pos_0)

    if cls_id == 1:
        # 1
        index_1 = (target == 1) * 1.0
        index_1_n = 1 - index_1

        Norm_pos_1 = 1.0 - pi[1].item()
        Norm_neg_1 = pi[1].item()
        if Norm_pos_1 ==0:
            Norm_pos_1 = 1e-6

        if Norm_neg_1 == 0:
            Norm_neg_1 = 1e-6

        pos_ul_1 = torch.mean(output_pos[:, 1, :, :])
        pos_l_1 = torch.sum(output_l_pos[:, 1, :, :] * index_1_n) / torch.sum(index_1_n)
        pos_1 = pos_ul_1 - pos_l_1*Norm_pos_1

        neg_ul_1 = torch.mean(output_neg[:, 1, :, :])
        neg_l_1 = torch.sum(output_l_neg[:, 1, :, :] * index_1) / torch.sum(index_1)
        neg_1 = neg_ul_1 - neg_l_1*Norm_neg_1

        normed_loss = pos_1*(1.0/Norm_neg_1)  + neg_1*(1.0/Norm_pos_1)

    if cls_id == 2:

        # 2
        index_2 = (target == 2) * 1.0
        index_2_n = 1 - index_2

        Norm_pos_2 = 1.0 - pi[2].item()
        Norm_neg_2 = pi[2].item()

        pos_ul_2 = torch.mean(output_pos[:, 2, :, :])
        pos_l_2 = torch.sum(output_l_pos[:, 2, :, :] * index_2_n) / torch.sum(index_2_n)
        pos_2 = pos_ul_2 - pos_l_2*Norm_pos_2

        neg_ul_2= torch.mean(output_neg[:, 2, :, :])
        neg_l_2 = torch.sum(output_l_neg[:, 2, :, :] * index_2) / torch.sum(index_2)
        neg_2 = neg_ul_2 - neg_l_2*Norm_neg_2

        normed_loss = pos_2*(1.0/Norm_neg_2)+ neg_2*(1.0/Norm_pos_2)

    if cls_id == 3:
        # 3
        index_3 = (target == 3) * 1.0
        index_3_n = 1 - index_3

        Norm_pos_3 = 1.0 - pi[3].item()
        Norm_neg_3 = pi[3].item()

        pos_ul_3 = torch.mean(output_pos[:, 3, :, :])
        pos_l_3 = torch.sum(output_l_pos[:, 3, :, :] * index_3_n) / torch.sum(index_3_n)
        pos_3 = pos_ul_3 - pos_l_3*Norm_pos_3

        neg_ul_3 = torch.mean(output_neg[:, 3, :, :])
        neg_l_3 = torch.sum(output_l_neg[:, 3, :, :] * index_3) / torch.sum(index_3)
        neg_3 = neg_ul_3 - neg_l_3*Norm_neg_3

        normed_loss = pos_3*(1.0/Norm_neg_3) + neg_3*(1.0/Norm_pos_3)

    return normed_loss


