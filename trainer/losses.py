# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import numpy as np
import torch
from utils.utils import _tranpose_and_gather_feature,_gather_feature
import math
import torch.nn.functional as F


def loss_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def loss_topN(scores, N=40):
    batch, cat, height, width = scores.size()

    # each class, top N in h*w    [b, c, N]
    topk_scores, topk_index = torch.topk(scores.view(batch, cat, -1), N)

    topk_index = topk_index % (height * width)
    topk_ys = (topk_index / width).int().float()
    topk_xs = (topk_index % width).int().float()

    # cross class, top N    [b, N]
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), N)

    topk_classes = (topk_ind / N).int()
    topk_index = _gather_feature(topk_index.view(batch, -1, 1), topk_ind).view(batch, N)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, N)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, N)

    return topk_score, topk_index, topk_classes, topk_ys, topk_xs



def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(torch.nn.Module):
    '''torch.nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)



class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
        self.gamma = 1.5
        self.beta = 1.0
        self.alpha = 0.5
        self.Beta = 1./9
        self.b =  np.e ** (self.gamma / self.alpha) - 1


    def forward(self, output, mask, index, target, index_all=None):
        pred = _tranpose_and_gather_feature(output, index, index_all=index_all)
        # pred --> b, N, 2*K
        # mask --> b, N ---> b, N, 2*K
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # L1 loss
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class TubeROIConsistencyLoss(torch.nn.Module):
    """Motion-aware consistency over the same tube object across K frames.

    Every frame uses its own ground-truth center and size, so the sampled ROI
    follows real object translation and scale changes.  The normalized ROI
    descriptor is matched both to the key frame and to adjacent frames.
    """

    def __init__(self, roi_size=3, beta=0.25):
        super(TubeROIConsistencyLoss, self).__init__()
        self.roi_size = roi_size
        self.beta = beta

    def _sample_tube_rois(self, sequence, centers, sizes):
        batch, steps, channels, height, width = sequence.shape
        num_objects = centers.size(1)
        relative = torch.linspace(
            -0.5,
            0.5,
            steps=self.roi_size,
            device=sequence.device,
            dtype=sequence.dtype,
        )
        grid_y, grid_x = torch.meshgrid(relative, relative)
        grid_x = grid_x.view(1, 1, self.roi_size, self.roi_size)
        grid_y = grid_y.view(1, 1, self.roi_size, self.roi_size)

        descriptors = []
        for frame_index in range(steps):
            frame_centers = centers[:, :, frame_index]
            frame_sizes = sizes[:, :, frame_index].clamp(min=1.0)
            sample_x = (
                frame_centers[:, :, 0, None, None]
                + frame_sizes[:, :, 0, None, None] * grid_x)
            sample_y = (
                frame_centers[:, :, 1, None, None]
                + frame_sizes[:, :, 1, None, None] * grid_y)
            sample_x = sample_x.clamp(min=0.0, max=float(width - 1))
            sample_y = sample_y.clamp(min=0.0, max=float(height - 1))

            if width > 1:
                sample_x = sample_x * (2.0 / float(width - 1)) - 1.0
            else:
                sample_x = torch.zeros_like(sample_x)
            if height > 1:
                sample_y = sample_y * (2.0 / float(height - 1)) - 1.0
            else:
                sample_y = torch.zeros_like(sample_y)

            grid = torch.stack([sample_x, sample_y], dim=-1)
            grid = grid.reshape(
                batch,
                num_objects * self.roi_size,
                self.roi_size,
                2,
            )
            sampled = F.grid_sample(
                sequence[:, frame_index],
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            )
            sampled = sampled.reshape(
                batch,
                channels,
                num_objects,
                self.roi_size,
                self.roi_size,
            )
            descriptor = sampled.mean(dim=(-1, -2)).permute(0, 2, 1)
            descriptors.append(descriptor)

        descriptors = torch.stack(descriptors, dim=2)
        return F.normalize(descriptors, p=2, dim=-1, eps=1e-6)

    def forward(self, sequence, center_kpoints, target_wh, mask):
        if sequence is None:
            raise ValueError('DPDF consistency requires refined sequence features.')
        if sequence.dim() != 5:
            raise ValueError(
                'Expected DPDF sequence [B,K,C,H,W], got {}.'.format(
                    tuple(sequence.shape)))

        batch, steps = sequence.shape[:2]
        if center_kpoints.size(-1) != steps * 2:
            raise ValueError(
                'centerKpoints has {} values, but DPDF sequence has {} frames.'
                .format(center_kpoints.size(-1), steps))
        if target_wh.size(-1) != steps * 2:
            raise ValueError(
                'wh has {} values, but DPDF sequence has {} frames.'.format(
                    target_wh.size(-1), steps))

        num_objects = center_kpoints.size(1)
        centers = center_kpoints.reshape(batch, num_objects, steps, 2)
        sizes = target_wh.reshape(batch, num_objects, steps, 2)
        centers = centers.to(dtype=sequence.dtype)
        sizes = sizes.to(dtype=sequence.dtype)
        descriptors = self._sample_tube_rois(sequence, centers, sizes)

        valid_objects = mask.gt(0)
        if not bool(valid_objects.any()):
            zero = sequence.sum() * 0.0
            return zero, zero, zero

        key_index = steps // 2
        key_descriptor = descriptors[:, :, key_index].detach()
        key_cosine = (
            descriptors * key_descriptor.unsqueeze(2)).sum(dim=-1)
        key_cosine = key_cosine.clamp(min=-1.0, max=1.0)
        non_key = torch.ones(
            steps, dtype=torch.bool, device=sequence.device)
        non_key[key_index] = False
        key_mask = valid_objects.unsqueeze(-1) & non_key.view(1, 1, steps)
        key_loss = ((1.0 - key_cosine) * key_mask.float()).sum()
        key_loss = key_loss / (key_mask.float().sum() + 1e-4)

        if steps > 1:
            adjacent_forward = (
                descriptors[:, :, :-1]
                * descriptors[:, :, 1:].detach()).sum(dim=-1)
            adjacent_backward = (
                descriptors[:, :, 1:]
                * descriptors[:, :, :-1].detach()).sum(dim=-1)
            adjacent_cosine = 0.5 * (
                adjacent_forward + adjacent_backward)
            adjacent_cosine = adjacent_cosine.clamp(min=-1.0, max=1.0)
            adjacent_mask = valid_objects.unsqueeze(-1).expand(
                batch, num_objects, steps - 1)
            adjacent_loss = (
                (1.0 - adjacent_cosine) * adjacent_mask.float()).sum()
            adjacent_loss = adjacent_loss / (
                adjacent_mask.float().sum() + 1e-4)
        else:
            adjacent_loss = sequence.sum() * 0.0

        consistency_loss = key_loss + self.beta * adjacent_loss
        return consistency_loss, key_loss, adjacent_loss

class STAloss(torch.nn.Module):
    def __init__(self,opt):
        super(STAloss, self).__init__()
        self.opt = opt

    def forward(self, centerKpoints,target_wh,output_hm,output_wh, output_STA_offset,mask,index, index_all=None):
        '''
        Args:
            centerKpoints: GT box centers of K frames
            target_wh: GT box width and height of K frames
            output_hm: predicted center heatmap
            output_wh: predicted width and height of K frames
            output_STA_offset: width and height offsets of STAloss
            mask: mask indicate how many objects in this tube
            index: index is key frame's boox center position
            index_all: index_all are all frame's bbox center position
        Returns:
        '''

        #pred_wh:[B N 2*K] mask:[B N]->[B N 2*K]
        pred_wh = _tranpose_and_gather_feature(output_wh, index, index_all=index_all)
        mask = mask.unsqueeze(2).expand_as(pred_wh).float()

        #heat:[B 1 W H ] offset:[B N 2*K]  N:object num
        #key frame center xs:[B N]  ys:[B N]
        heat = loss_nms(output_hm)# B 1 72 72
        scores, index_pred, classes, ys, xs = loss_topN(heat, N= 1)
        offset = _tranpose_and_gather_feature(output_STA_offset, index_pred)#B N(1) 2*K

        #extract the objects to be trained
        # [B N 2*K]
        pred = pred_wh #* mask
        target = target_wh * mask
        centerKpoints = centerKpoints*mask
        # offset = offset * mask_expand

        # [B N]
        # xs = xs * mask
        # ys = ys * mask

        #only one box in IOD-Video dataset
        pred = pred[:,0,:].unsqueeze(1)#B 1 2*K
        offset = offset[:,0,:].unsqueeze(1)#B 1 2*K
        target = target[:,0,:].unsqueeze(1)#B 1 2*K
        centerKpoints = centerKpoints[:,0,:].unsqueeze(1)
        xs = xs[:,0].unsqueeze(1)
        ys = ys[:,0].unsqueeze(1)

        # [B N 2*K] -> [B*N, K, 2]
        B, N, K2  =pred.size()
        K = K2//2
        pred = pred.view(B, N, K, 2).contiguous()
        pred = pred.view(B * N, K, 2)
        target = target.view(B, N, K, 2).contiguous()
        target = target.view(B * N, K, 2)
        centerKpoints = centerKpoints.view(B, N, K, 2).contiguous()
        centerKpoints = centerKpoints.view(B * N, K, 2)
        offset = offset.view(B, N, K, 2).contiguous()
        offset = offset.view(B * N, K, 2)

        # [B N ] -> [B*N, K]
        xs = xs.view(B*N,1).repeat(1,K)
        ys = ys.view(B*N,1).repeat(1,K)

        #predicted boxes for K frames
        pred_x1y1x2y2 = torch.zeros(B*N,K,4).cuda()#B*N K 4
        target_x1y1x2y2 = torch.zeros(B*N,K,4).cuda()#B*N K 4

        pred_x1y1x2y2[:, :, 0] = xs + offset[:, :, 0] * self.opt.offset_w_ratio - pred[:, :, 0] * 0.5
        pred_x1y1x2y2[:, :, 1] = ys + offset[:, :, 1] * self.opt.offset_h_ratio - pred[:, :, 1] * 0.5
        pred_x1y1x2y2[:, :, 2] = xs + offset[:, :, 0] * self.opt.offset_w_ratio + pred[:, :, 0] * 0.5
        pred_x1y1x2y2[:, :, 3] = ys + offset[:, :, 1] * self.opt.offset_h_ratio + pred[:, :, 1] * 0.5

        target_x1y1x2y2[:, :, 0] = centerKpoints[:, :, 0] - target[:, :, 0] * 0.5
        target_x1y1x2y2[:, :, 1] = centerKpoints[:, :, 1] - target[:, :, 1] * 0.5
        target_x1y1x2y2[:, :, 2] = centerKpoints[:, :, 0] + target[:, :, 0] * 0.5
        target_x1y1x2y2[:, :, 3] = centerKpoints[:, :, 1] + target[:, :, 1] * 0.5
        # print("target_x1y1x2y2 v5",target_x1y1x2y2.sum().detach().cpu().numpy())
        # print("pred_x1y1x2y2 v5",pred_x1y1x2y2.sum().detach().cpu().numpy())

        loss_3d_sin,loss_3d_cos = self.STAloss_core(pred_x1y1x2y2, target_x1y1x2y2, eps=1e-7)
        return loss_3d_sin,loss_3d_cos


    def STAloss_core(self, pred, target, eps=1e-7):
        """`
        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, K, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, K, 4).
            B: BATCHSIZE
        Return:
            Tensor: Loss tensor.
        """

        # pred: B*N K 4   target: B*N K 4
        # Px1 Py1...: B*N K
        Px1, Py1 = pred[:, :, 0], pred[:, :, 1]
        Px2, Py2 = pred[:, :, 2], pred[:, :, 3]
        Gx1, Gy1 = target[:, :, 0], target[:, :, 1]
        Gx2, Gy2 = target[:, :, 2], target[:, :, 3]

        #boxes center
        Px, Py = (Px1 + Px2)/2, (Py1 + Py2)/2#B*N K
        Gx, Gy = (Gx1 + Gx2)/2, (Gy1 + Gy2)/2#B*N K

        # vector
        Px_nex, Gx_nex =Px[:,1:], Gx[:,1:]#B*N K-1
        Px_pre, Gx_pre =Px[:,:-1], Gx[:,:-1]
        Py_nex, Gy_nex =Py[:,1:], Gy[:,1:]
        Py_pre, Gy_pre =Py[:,:-1], Gy[:,:-1]

        Vggx,Vggy = Gx_nex - Gx_pre,Gy_nex - Gy_pre
        Vppx,Vppy = Px_nex - Px_pre,Py_nex - Py_pre
        Vgpx,Vgpy = Gx_nex - Px_pre,Gy_nex - Py_pre
        Vpgx,Vpgy = Px_nex - Gx_pre,Py_nex - Gy_pre

        DPpgx,DPpgy = Px_pre - Gx_pre, Py_pre - Gy_pre
        DNpgx,DNpgy = Px_nex - Gx_nex, Py_nex - Gy_nex

        #STAloss
        HH = self.opt.temporal_interal
        loss_sta_sin_P = torch.sqrt(DPpgx*DPpgx + DPpgy*DPpgy + eps)/ \
                        torch.sqrt(Vgpx * Vgpx + Vgpy * Vgpy + HH)
        loss_sta_sin_N = torch.sqrt(DNpgx*DNpgx + DNpgy*DNpgy + eps)/ \
                        torch.sqrt(Vpgx * Vpgx + Vpgy * Vpgy + HH)
        loss_sta_sin = (loss_sta_sin_P + loss_sta_sin_N)/2

        loss_sta_cross = (Vgpx*Vpgx + Vgpy*Vpgy + HH)/ \
                        (torch.sqrt(Vgpx*Vgpx+Vgpy*Vgpy+ HH)*torch.sqrt(Vpgx*Vpgx+Vpgy*Vpgy+ HH))
        loss_sta_own = (Vggx*Vppx + Vggy*Vppy + HH)/ \
                      (torch.sqrt(Vggx*Vggx+Vggy*Vggy+ HH)*torch.sqrt(Vppx*Vppx+Vppy*Vppy+ HH))
        loss_sta_cos = (loss_sta_cross + loss_sta_own)/2

        loss_sta_cos = loss_sta_cos.view(-1, 1).squeeze(1)
        loss_sta_sin = loss_sta_sin.view(-1, 1).squeeze(1)
        loss_sta_cos = 1 - loss_sta_cos

        loss_sta_sin = 0.5 * loss_sta_sin
        loss_sta_cos = 0.5 * loss_sta_cos

        return loss_sta_sin,loss_sta_cos

class STAloss2(torch.nn.Module):
    def __init__(self,opt):
        super(STAloss2, self).__init__()
        self.opt = opt

    def forward(self, centerKpoints,target_wh,output_hm,output_wh, output_STA_offset,mask,index, index_all=None):
        '''
        Args:
            centerKpoints: GT box centers of K frames
            target_wh: GT box width and height of K frames
            output_hm: predicted center heatmap
            output_wh: predicted width and height of K frames
            output_STA_offset: width and height offsets of STAloss
            mask: mask indicate how many objects in this tube
            index: index is key frame's boox center position
            index_all: index_all are all frame's bbox center position
        Returns:
        '''

        #pred_wh:[B N 2*K] mask:[B N]->[B N 2*K]
        pred_wh = _tranpose_and_gather_feature(output_wh, index, index_all=index_all)
        mask = mask.unsqueeze(2).expand_as(pred_wh).float()

        #heat:[B 1 W H ] offset:[B 1 2*K]  1:only one class
        #key frame center xs:[B 1]  ys:[B 1]
        heat = loss_nms(output_hm)# B 1 72 72
        scores, index_pred, classes, ys, xs = loss_topN(heat, N=1)#
        offset = _tranpose_and_gather_feature(output_STA_offset, index_pred)#B N(1) 2*K


        #[B N 2*K]->[B 1 2*K]-> [B 2*K]
        pred = pred_wh * mask
        target = target_wh * mask
        centerKpoints = centerKpoints*mask
        #due to only one box for every frame in IOD-Video
        pred_1 = pred[:,0,:]
        offset_1 = offset[:,0,:]
        target_1 = target[:,0,:]
        centerKpoints_1 = centerKpoints[:,0,:]

        #[B 2*K]->[B*K 2]
        B, K2  =pred_1.size()
        pred_1 =pred_1.view(B,K2//2,2).contiguous()#B K 2
        pred_1 =pred_1.view(B * K2//2,1, 2)  # B*K 1 2
        pred_1 =pred_1.squeeze(1)#B*K 2
        offset_1 =offset_1.view(B,K2//2,2).contiguous()#B K 2
        offset_1 =offset_1.view(B * K2//2,1, 2)  # B*K 1 2
        offset_1 =offset_1.squeeze(1)#B*K 2
        target_1 =target_1.view(B,K2//2,2).contiguous()#B K 2
        target_1 =target_1.view(B * K2 // 2,1, 2) # B*K 1 2
        target_1 =target_1.squeeze(1)#B*K 2
        centerKpoints_1 =centerKpoints_1.view(B,K2//2,2).contiguous()#B K 2
        centerKpoints_1 =centerKpoints_1.view(B * K2 // 2,1, 2) # B*K 1 2
        centerKpoints_1 =centerKpoints_1.squeeze(1)#B*K 2

        xs = xs.repeat(1,K2//2).view(B*K2//2,1).squeeze(1)#B*K 1 -> B*K
        ys = ys.repeat(1,K2//2).view(B*K2//2,1).squeeze(1)#B*K 1 -> B*K

        #predicted boxes for K frames
        pred_x1y1x2y2 = torch.zeros(B*K2//2,4).cuda()#B*K 4
        target_x1y1x2y2 = torch.zeros(B*K2//2,4).cuda()#B*K 4

        self.opt.offset_max = 18
        #pred_x1y1x2y2:[B*K,4] target_x1y1x2y2:[B*K,4]
        pred_x1y1x2y2[:,0] = xs + offset_1[:, 0]*self.opt.offset_max - pred_1[:, 0]*0.5
        pred_x1y1x2y2[:,1] = ys + offset_1[:, 1]*self.opt.offset_max - pred_1[:, 1]*0.5
        pred_x1y1x2y2[:,2] = xs + offset_1[:, 0]*self.opt.offset_max + pred_1[:, 0]*0.5
        pred_x1y1x2y2[:,3] = ys + offset_1[:, 1]*self.opt.offset_max + pred_1[:, 1]*0.5

        target_x1y1x2y2[:,0] = centerKpoints_1[:,0] - target_1[:, 0]*0.5
        target_x1y1x2y2[:,1] = centerKpoints_1[:,1] - target_1[:, 1]*0.5
        target_x1y1x2y2[:,2] = centerKpoints_1[:,0] + target_1[:, 0]*0.5
        target_x1y1x2y2[:,3] = centerKpoints_1[:,1] + target_1[:, 1]*0.5
        # print("target_x1y1x2y2 v4",target_x1y1x2y2.sum().detach().cpu().numpy())
        # print("pred_x1y1x2y2 v4",pred_x1y1x2y2.sum().detach().cpu().numpy())

        loss_3d_sin,loss_3d_cos = self.STAloss_core2(pred_x1y1x2y2, target_x1y1x2y2,B, eps=1e-7)
        return loss_3d_sin,loss_3d_cos


    def STAloss_core2(self, pred, target, B, eps=1e-7):
        """`
        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): Corresponding gt bboxes, shape (n, 4).
            B: BATCHSIZE
        Return:
            Tensor: Loss tensor.
        """
        #predict boxes:[b1_x1,b1_y1,b1_x2,b1_y2]
        #Gt boxes[b2_x1,b2_y1,b2_x2,b2_y2]
        b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
        b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
        b2_x1, b2_y1 = target[:, 0], target[:, 1]
        b2_x2, b2_y2 = target[:, 2], target[:, 3]

        #boxes center
        Px, Py = (b1_x2 + b1_x1)/2, (b1_y2 + b1_y1)/2#B*K 1
        Gx, Gy = (b2_x2 + b2_x1)/2, (b2_y2 + b2_y1)/2#B*K 1

        Px = Px.view(B,-1)#B K
        Py = Py.view(B,-1)
        Gx = Gx.view(B,-1)
        Gy = Gy.view(B,-1)

        Px_nex =Px[:,1:]#B K-1
        Px_pre =Px[:,:-1]
        Py_nex =Py[:,1:]
        Py_pre =Py[:,:-1]

        Gx_nex =Gx[:,1:]#B K-1
        Gx_pre =Gx[:,:-1]
        Gy_nex =Gy[:,1:]
        Gy_pre =Gy[:,:-1]

        Vggx,Vggy = Gx_nex - Gx_pre,Gy_nex - Gy_pre
        Vppx,Vppy = Px_nex - Px_pre,Py_nex - Py_pre
        Vgpx,Vgpy = Gx_nex - Px_pre,Gy_nex - Py_pre
        Vpgx,Vpgy = Px_nex - Gx_pre,Py_nex - Gy_pre

        DPpgx,DPpgy = Px_pre - Gx_pre, Py_pre - Gy_pre
        DNpgx,DNpgy = Px_nex - Gx_nex, Py_nex - Gy_nex

        HH = self.opt.temporal_interal
        loss_sta_sin_P = torch.sqrt(DPpgx*DPpgx + DPpgy*DPpgy + eps)/ \
                         torch.sqrt(Vgpx * Vgpx + Vgpy * Vgpy + HH)
        loss_sta_sin_N = torch.sqrt(DNpgx*DNpgx + DNpgy*DNpgy + eps)/ \
                         torch.sqrt(Vpgx * Vpgx + Vpgy * Vpgy + HH)
        loss_sta_sin = (loss_sta_sin_P + loss_sta_sin_N)/2

        loss_sta_cross = (Vgpx*Vpgx + Vgpy*Vpgy + HH)/ \
                         (torch.sqrt(Vgpx*Vgpx+Vgpy*Vgpy+ HH)*torch.sqrt(Vpgx*Vpgx+Vpgy*Vpgy+ HH))
        loss_sta_own = (Vggx*Vppx + Vggy*Vppy + HH)/ \
                       (torch.sqrt(Vggx*Vggx+Vggy*Vggy+ HH)*torch.sqrt(Vppx*Vppx+Vppy*Vppy+ HH))
        loss_sta_cos = (loss_sta_cross + loss_sta_own)/2

        loss_sta_cos = loss_sta_cos.view(-1, 1).squeeze(1)
        loss_sta_sin = loss_sta_sin.view(-1, 1).squeeze(1)
        loss_sta_cos = 1 - loss_sta_cos

        loss_sta_sin = loss_sta_sin/2
        loss_sta_cos = loss_sta_cos/2

        return loss_sta_sin,loss_sta_cos
