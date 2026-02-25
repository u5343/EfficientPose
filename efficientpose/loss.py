import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import compute_rotation_matrix_from_ortho6d


class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sx = nn.Parameter(torch.tensor(0.0))
        self.sq = nn.Parameter(torch.tensor(-3.0))

    def forward(self, pred_r_6d, pred_t, gt_r, gt_t):
        pred_r_mat = compute_rotation_matrix_from_ortho6d(pred_r_6d)

        # Origin/translation loss
        loss_t = self.l1(pred_t, gt_t)

        # For axis-symmetric objects, only supervise Y-axis direction.
        pred_y = pred_r_mat[:, :, 1]
        gt_y = gt_r[:, :, 1]
        cos_sim = F.cosine_similarity(pred_y, gt_y, dim=1)
        loss_y = (1.0 - cos_sim).mean()

        total_loss = (
            torch.exp(-self.sx) * loss_t
            + self.sx
            + torch.exp(-self.sq) * loss_y
            + self.sq
        )
        return total_loss, loss_t.item(), loss_y.item()
