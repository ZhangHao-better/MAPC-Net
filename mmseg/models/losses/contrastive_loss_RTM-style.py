# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.registry import MODELS


SMALL_NUM = np.log(1e-45)



@MODELS.register_module()
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,
                 loss_weight=1.0,
                 gather=False,
                 min_points=None):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.loss_weight = loss_weight
        self.gather = gather
        self.min_points = min_points
        if min_points is None:
            self.min_points = 2

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = features.device
        # 只有在分布式已初始化且 gather=True 时，才做跨进程聚合
        if self.gather and dist.is_available() and dist.is_initialized():
            assert mask is None
            current_rank = dist.get_rank()

            all_features = get_all_gather_with_various_shape(features)

            # 保留本 rank 的特征梯度，其他 rank 的特征作为常量参与对比
            all_features[current_rank] = features
            features = torch.cat(all_features, dim=0)

            if labels is not None:
                all_labels = get_all_gather_with_various_shape(labels)
                all_labels[current_rank] = labels
                labels = torch.cat(all_labels, dim=0)


        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        num_T = labels.sum()

        if num_T < self.min_points or (batch_size - self.min_points) < 2:
            # print("construct sample pairs failed")
            return None

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        loss = self.loss_weight * loss

        return loss



@MODELS.register_module()
class SupDCLLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.1, contrast_mode='one',
                 base_temperature=0.07, loss_weight=1.0):
        super(SupDCLLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.loss_weight = loss_weight

    def forward(self, pos_feats, neg_feats):
        """Compute loss for model.

        Args:
            pos_feats: hidden vector of shape [M, C].
            neg_feats: ground truth of shape [N, C].

        Returns:
            A loss scalar.
        """
        pos_feats = concat_all_gather(pos_feats)
        neg_feats = concat_all_gather(neg_feats)

        print(pos_feats.shape)

        M, C = pos_feats.shape
        N = neg_feats.shape[0]

        pos_feats = F.normalize(pos_feats, dim=1)
        neg_feats = F.normalize(neg_feats, dim=1)

        # compute logits
        pos1_simi = torch.div(torch.matmul(pos_feats, pos_feats.T), self.temperature)
        pos2_simi = torch.div(torch.matmul(neg_feats, neg_feats.T), self.temperature)

        neg_simi = torch.div(torch.matmul(pos_feats, neg_feats.T), self.temperature)

        exp_pos1 = torch.exp(pos1_simi)
        exp_pos2 = torch.exp(pos2_simi)
        exp_neg = torch.exp(neg_simi)

        diag_mask1 = torch.eye(M, device=exp_pos1.device, dtype=torch.bool)
        diag_mask2 = torch.eye(N, device=exp_pos2.device, dtype=torch.bool)

        exp_neg1 = torch.sum(exp_neg, dim=1, keepdim=True)
        exp_neg2 = torch.sum(exp_neg.transpose(0, 1), dim=1, keepdim=True)

        exp_neg1 = exp_pos1 + exp_neg1
        exp_neg2 = exp_pos2 + exp_neg2

        # DCL loss
        exp_pos1 = -torch.log(torch.div(exp_pos1, exp_neg1))
        exp_pos2 = -torch.log(torch.div(exp_pos2, exp_neg2))

        exp_pos1 = exp_pos1[~diag_mask1]
        exp_pos2 = exp_pos2[~diag_mask2]

        # SupDCL loss
        loss1 = torch.mean(exp_pos1)
        loss2 = torch.mean(exp_pos2)
        if loss1 < 0 or loss2 < 0:
            print("warning: pos-{}, neg-{}, loss1: {}, loss2: {}".format(M, N, loss1, loss2))
        loss = 0.5 * (loss1 + loss2) * self.loss_weight

        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def get_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return tensors_gather


@torch.no_grad()
def get_all_gather_with_various_shape(tensor):
    """支持 dim0 长度不同的 all_gather。

    说明：PyTorch/NCCL 的 all_gather 要求各 rank 的输入 Tensor 形状一致。
    本函数通过“先同步长度 -> 按最大长度补零 -> all_gather -> 再裁剪”的方式，
    让 dim0 可以不一致（例如每张卡采样点数 M 不同）。

    约束：除 dim0 外，其余维度必须在各 rank 间保持一致。
    """

    if not (dist.is_available() and dist.is_initialized()):
        return [tensor]

    world_size = dist.get_world_size()
    device = tensor.device

    # 1) 先收集各 rank 的 dim0 长度
    local_len = torch.tensor([tensor.shape[0]], device=device, dtype=torch.long)
    len_gather = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(len_gather, local_len, async_op=False)
    lens = [int(x.item()) for x in len_gather]
    max_len = max(lens)

    # 2) 补零到 max_len，保证 all_gather 输入形状一致
    if tensor.shape[0] < max_len:
        pad_shape = (max_len - tensor.shape[0],) + tuple(tensor.shape[1:])
        pad_tensor = torch.zeros(pad_shape, device=device, dtype=tensor.dtype)
        tensor_pad = torch.cat([tensor, pad_tensor], dim=0)
    else:
        tensor_pad = tensor

    # 3) all_gather（此处各 rank 输入 tensor_pad 形状一致，不会死锁）
    tensors_gather = [torch.zeros_like(tensor_pad) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor_pad, async_op=False)

    # 4) 按各自长度裁剪，恢复变长列表
    tensors_gather = [t[:lens[i]] for i, t in enumerate(tensors_gather)]
    return tensors_gather



@torch.no_grad()
def concat_other_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    current_rank = torch.distributed.get_rank()

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    tensors_gather.pop(current_rank)
    output = torch.cat(tensors_gather, dim=0)
    return output


