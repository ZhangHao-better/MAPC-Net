# Copyright (c) OpenMMLab. All rights reserved.
# 消融实验，w/o MSP
#定义：只保留最终层 mask1 预测和最终层损失；去掉 mask4/3/2 三层预测、三层辅助监督和所有跨层引导。为了更干净地只消融 MSP，这里保留最终层的 NonLocalMask。
"""
Progressive Contrastive Head with PSCC-Net Architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import FocalLoss, BinaryDiceLoss
from typing import List, Tuple, Dict


class NonLocalMask(nn.Module):
    """NonLocal Mask模块"""
    
    def __init__(self, in_channels: int, reduce_scale: int = 1):
        super(NonLocalMask, self).__init__()
        self.r = reduce_scale
        self.ic = in_channels * self.r * self.r
        self.mc = self.ic
        
        self.g = nn.Conv2d(self.ic, self.ic, kernel_size=1)
        self.theta = nn.Conv2d(self.ic, self.mc, kernel_size=1)
        self.phi = nn.Conv2d(self.ic, self.mc, kernel_size=1)
        self.W_s = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.W_c = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma_s = nn.Parameter(torch.ones(1))
        self.gamma_c = nn.Parameter(torch.ones(1))
        
        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
            # 移除Sigmoid，输出logits用于Focal Loss
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x1 = x.reshape(b, self.ic, h // self.r, w // self.r) if self.r > 1 else x
        
        g_x = self.g(x1).view(b, self.ic, -1).permute(0, 2, 1)
        theta_x = self.theta(x1).view(b, self.mc, -1)
        phi_x = self.phi(x1).view(b, self.mc, -1)
        
        # Spatial attention
        f_s = torch.matmul(theta_x.permute(0, 2, 1), phi_x)
        f_s_div = F.softmax(f_s, dim=-1)
        
        # Channel attention
        f_c = torch.matmul(theta_x, phi_x.permute(0, 2, 1))
        f_c_div = F.softmax(f_c, dim=-1)
        
        y_s = torch.matmul(f_s_div, g_x).permute(0, 2, 1).contiguous().view(b, c, h, w)
        y_c = torch.matmul(g_x, f_c_div).view(b, c, h, w)
        
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)
        mask = self.getmask(z)
        
        return mask, z


@MODELS.register_module()
class ProgressiveContrastiveHead(BaseDecodeHead):
    """Progressive Contrastive Head"""
    
    def __init__(self, crop_size=(512, 512), reduce_scales=[4, 2, 2, 1], 
                 use_contrastive=False, **kwargs):
        super(ProgressiveContrastiveHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        
        self.crop_size = crop_size
        self.reduce_scales = reduce_scales
        self.use_contrastive = use_contrastive
        
        self.getmask4 = NonLocalMask(self.in_channels[3], reduce_scales[3])
        self.getmask3 = NonLocalMask(self.in_channels[2], reduce_scales[2])
        self.getmask2 = NonLocalMask(self.in_channels[1], reduce_scales[1])
        self.getmask1 = NonLocalMask(self.in_channels[0], reduce_scales[0])
        
        # 初始化Focal Loss
        self.focal_loss = FocalLoss(
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0
        )
        
        # 初始化Binary Dice Loss (仅用于mask1)
        self.dice_loss = BinaryDiceLoss(
            smooth=1,
            exponent=2,
            reduction='mean',
            loss_weight=1.0  # Dice自身的内部权重
        )
        self.dice_loss_weight = 0.01  # 在总损失中给Dice的系数
        self.boundary_width = 2         # 边界宽度：1~3都行，先用2（对应5x5核）

        # 1) 定义 total_loss_scale（先设为2.0，保证训练强度对比loss2时匹配）
        self.total_loss_scale = 2.0  # 调整为合适的 scale
    
    def forward(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = self._transform_inputs(inputs)
        s1, s2, s3, s4 = inputs
        
        target_1 = self.crop_size
        
        
        if s1.shape[2:] != target_1:
            s1 = F.interpolate(s1, size=target_1, mode='bilinear', align_corners=False)
            
        
        # w/o MSP:
        # remove mask4 / mask3 / mask2 and all auxiliary supervision;
        # keep only the final-stage prediction head (with NLM).
        mask1, z1 = self.getmask1(s1)
        
        if self.training:
            return {'mask1': mask1}
        return {'mask1': mask1}
    
    def loss_by_feat(self, seg_logits: Dict[str, torch.Tensor], 
                     batch_data_samples) -> Dict[str, torch.Tensor]:
        seg_label = self._stack_batch_gt(batch_data_samples)
        if seg_label.dim() == 4:
            seg_label = seg_label.squeeze(1)
        B, H, W = seg_label.shape
        
        # Downsample ground truth masks to match prediction sizes
        # 转为long类型（FocalLoss要求类别索引）
        gt_mask1 = seg_label.long()
        gt_mask2 = F.interpolate(seg_label.unsqueeze(1).float(), 
                                 size=(H//2, W//2), mode='nearest').squeeze(1).long()
        gt_mask3 = F.interpolate(seg_label.unsqueeze(1).float(), 
                                 size=(H//4, W//4), mode='nearest').squeeze(1).long()
        gt_mask4 = F.interpolate(seg_label.unsqueeze(1).float(), 
                                 size=(H//8, W//8), mode='nearest').squeeze(1).long()
        
        # 使用Focal Loss + 渐进式权重 [0.6, 0.8, 1.0, 1.0]
        # mask1: Focal Loss + Binary Dice Loss
        logit1 = seg_logits['mask1']  # [B, 1, H, W]
        
        # 1) Focal Loss (和之前一样)
        loss1_focal = self.focal_loss(logit1, gt_mask1)
        
        # # 2) Binary Dice: 先sigmoid得到概率，再和0/1 GT计算(之前的整体Dice，先注释掉，改成边界Dice)
        # prob1 = torch.sigmoid(logit1)  # [B, 1, H, W], ∈[0,1]
        # dice_target = gt_mask1.float().unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        # loss1_dice = self.dice_loss(prob1, dice_target)

        # 2) Boundary Dice (只在边界带算 Dice)
        prob1 = torch.sigmoid(logit1)  # [B,1,H,W]
        tgt1  = (gt_mask1 == 1).float().unsqueeze(1)  # [B,1,H,W]，确保是0/1

        # --- 用形态学梯度生成边界带 ---
        w = self.boundary_width
        k = 2 * w + 1

        # dilation / erosion (纯 torch，无需opencv)
        dilate = F.max_pool2d(tgt1, kernel_size=k, stride=1, padding=w)
        erode  = 1.0 - F.max_pool2d(1.0 - tgt1, kernel_size=k, stride=1, padding=w)

        boundary = (dilate - erode).clamp(0, 1)  # 边界带（大约2w像素厚）

        # 如果未来真的出现 ignore=255，也可以加这行更稳（现在可选）
        # valid = (gt_mask1 != 255).float().unsqueeze(1)
        # boundary = boundary * valid

        prob_b = prob1 * boundary
        tgt_b  = tgt1  * boundary

        # 边界带为空（例如该 crop 没有篡改区域）时，bdice 置 0，避免出现无意义/不稳定的统计
        if boundary.sum() < 1:
            loss1_bdice = prob1.sum() * 0.0   # 保证是 tensor、在同设备上
        else:
            loss1_bdice = self.dice_loss(prob_b, tgt_b)

        
        # 额外统计（detach，避免参与反传）
        boundary_ratio = boundary.mean().detach()
        bdice_raw = loss1_bdice.detach()
        bdice_coef = (1.0 - bdice_raw)
        
        losses = {
            'loss_mask1_focal': loss1_focal * 0.6,
            'loss_mask1_bdice': loss1_bdice * self.dice_loss_weight,

            # === 仅用于日志观察，不参与 loss 求和 ===
            'mask1_boundary_ratio': boundary_ratio,
            'mask1_bdice_raw': bdice_raw,
            'mask1_bdice_coef': bdice_coef,
        }
        
        if 'mask2' in seg_logits:
            loss2 = self.focal_loss(seg_logits['mask2'], gt_mask2)
            losses['loss_mask2'] = loss2 * 0.8
        
        if 'mask3' in seg_logits:
            loss3 = self.focal_loss(seg_logits['mask3'], gt_mask3)
            losses['loss_mask3'] = loss3 * 1.0
        
        if 'mask4' in seg_logits:
            loss4 = self.focal_loss(seg_logits['mask4'], gt_mask4)
            losses['loss_mask4'] = loss4 * 1.0
        

        # 2) 对每个 loss 逐项乘以 total_loss_scale
        if self.total_loss_scale != 1.0:
            for k in list(losses.keys()):
                if k.startswith('loss_'):
                    losses[k] = losses[k] * self.total_loss_scale

        return losses
    
    def predict(self, inputs: List[torch.Tensor], batch_img_metas: List[dict], 
                test_cfg: dict) -> torch.Tensor:
        mask_logits = self.forward(inputs)['mask1']
        # 测试时需要sigmoid转换为概率
        return torch.sigmoid(mask_logits)
