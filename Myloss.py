"""IECC and RGB/HVI restoration losses, paper Eqs. (6)-(8)."""
import torch
from torch import nn
from torch.nn import functional as F


class IECCLoss(nn.Module):
    """Intensity-Error-Aware Chromatic Consistency, Eqs. (6)-(7).

    Assumed HVI channel order:
        channel 0 -> H
        channel 1 -> V
        channel 2 -> I
    """

    def __init__(self, lambda_mu=1.0, lambda_var=0.5, lambda_cov=0.01):
        super(IECCLoss, self).__init__()
        self.lambda_mu = lambda_mu
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov

    @staticmethod
    def _flatten_per_sample(x):
        return x.reshape(x.shape[0], -1)

    def forward(self, pred_hvi, gt_hvi):
        pred_h = pred_hvi[:, 0:1, :, :]
        pred_v = pred_hvi[:, 1:2, :, :]
        pred_i = pred_hvi[:, 2:3, :, :]

        gt_h = gt_hvi[:, 0:1, :, :]
        gt_v = gt_hvi[:, 1:2, :, :]
        gt_i = gt_hvi[:, 2:3, :, :]

        # Per-sample H/V chroma reconstruction error.
        chroma_h = torch.mean(torch.abs(pred_h - gt_h), dim=(1, 2, 3))
        chroma_v = torch.mean(torch.abs(pred_v - gt_v), dim=(1, 2, 3))
        chroma_err = chroma_h + chroma_v

        # I-channel residual-aware weighting.
        i_res = pred_i - gt_i
        i_abs_mean = torch.mean(torch.abs(i_res), dim=(1, 2, 3))
        i_var = torch.var(i_res.reshape(i_res.shape[0], -1), dim=1, unbiased=False)
        weight = 1.0 + self.lambda_mu * i_abs_mean + self.lambda_var * i_var

        weighted_chroma = torch.mean(weight * chroma_err)

        # H/V covariance consistency.
        pred_h_flat = self._flatten_per_sample(pred_h)
        pred_v_flat = self._flatten_per_sample(pred_v)
        gt_h_flat = self._flatten_per_sample(gt_h)
        gt_v_flat = self._flatten_per_sample(gt_v)

        pred_h_centered = pred_h_flat - pred_h_flat.mean(dim=1, keepdim=True)
        pred_v_centered = pred_v_flat - pred_v_flat.mean(dim=1, keepdim=True)
        gt_h_centered = gt_h_flat - gt_h_flat.mean(dim=1, keepdim=True)
        gt_v_centered = gt_v_flat - gt_v_flat.mean(dim=1, keepdim=True)

        pred_cov = torch.mean(pred_h_centered * pred_v_centered, dim=1)
        gt_cov = torch.mean(gt_h_centered * gt_v_centered, dim=1)
        cov_loss = torch.mean(torch.abs(pred_cov - gt_cov))

        return weighted_chroma + self.lambda_cov * cov_loss


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        x = torch.arange(11, dtype=torch.float32) - 5
        g = torch.exp(-x.square() / (2 * 1.5**2))
        g = g / g.sum()
        self.register_buffer('window', (g[:, None] * g[None, :])[None, None])

    def forward(self, x, y):
        window = self.window.to(x).expand(x.shape[1], 1, 11, 11)
        def avg(z):
            return F.conv2d(z, window, padding=5, groups=x.shape[1])
        mx, my = avg(x), avg(y)
        vx, vy, cov = avg(x*x)-mx*mx, avg(y*y)-my*my, avg(x*y)-mx*my
        ssim = ((2*mx*my + 0.01**2)*(2*cov + 0.03**2) /
                ((mx*mx + my*my + 0.01**2)*(vx + vy + 0.03**2)))
        return 1 - ssim.mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([0.05, 0.25, 0.4, 0.25, 0.05])
        self.register_buffer('kernel', (k[:, None]*k[None, :])[None, None])

    def blur(self, x):
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode='replicate'),
                        self.kernel.to(x).expand(x.shape[1], 1, 5, 5), groups=x.shape[1])

    def laplacian(self, x):
        filtered = self.blur(x)
        up = torch.zeros_like(filtered)
        up[..., ::2, ::2] = filtered[..., ::2, ::2] * 4
        return x - self.blur(up)

    def forward(self, x, y):
        return F.mse_loss(self.laplacian(x), self.laplacian(y))


class RestorationLoss(nn.Module):
    def __init__(self, l1=1.0, ssim=0.5, edge=50.0, hvi=1.0, iecc=0.005,
                 lambda_mu=1.0, lambda_var=0.5, lambda_cov=0.01):
        super().__init__()
        self.weights = dict(l1=l1, ssim=ssim, edge=edge, hvi=hvi, iecc=iecc)
        self.ssim, self.edge = SSIMLoss(), EdgeLoss()
        self.iecc = IECCLoss(lambda_mu, lambda_var, lambda_cov)

    def domain(self, x, y):
        w = self.weights
        return w['l1']*F.l1_loss(x, y) + w['ssim']*self.ssim(x, y) + w['edge']*self.edge(x, y)

    def forward(self, pred_rgb, pred_hvi, gt_rgb, gt_hvi):
        rgb, hvi = self.domain(pred_rgb, gt_rgb), self.domain(pred_hvi, gt_hvi)
        iecc = self.iecc(pred_hvi, gt_hvi) if self.weights['iecc'] else rgb.new_zeros(())
        return rgb + self.weights['hvi']*hvi + self.weights['iecc']*iecc
