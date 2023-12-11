import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
from helpers import pdist, smooth


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, device='cuda', thresh=None):
        self.win = win
        self.eps = torch.tensor([eps]).float().to(device)
        self.device = torch.device(device)
        self.thresh = thresh

    def loss(self, y_true, y_pred):
        if self.thresh:
            Ii = y_true * (y_true < self.thresh).float()
            Ji = y_pred * (y_true < self.thresh).float()
        else:
            Ii = y_true
            Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.maximum(J_var, self.eps)

        cc = (cross / I_var) * (cross / J_var)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class MaskedMSE:
    """
    Mean squared error loss at masked regions
    default threshold at 0.8 is suitable for CT skull removing
    """
    def __init__(self, thresh=0.8):
        self.thresh = thresh

    def loss(self, y_true, y_pred):
        mask = (y_true < self.thresh) * (y_pred < self.thresh)
        return torch.mean(torch.masked_select((y_true - y_pred) ** 2, mask>0))


class MaskedL1:
    """
    L1 loss at masked regions
    """
    def __init__(self, low_thresh=0.01, high_thresh=0.8):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh

    def loss(self, y_true, y_pred):
        mask = (y_true > self.low_thresh) * (y_true < self.high_thresh)
        mask = mask + (~mask) * 0.05

        return torch.mean(torch.abs(y_true - y_pred) * mask)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = y_true.sum(dim=vol_axes) + y_pred.sum(dim=vol_axes) + 1e-5
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, _, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


class MIND:
    """
    Modality-Independent NeighborHood Descriptor
    """

    def __init__(self, vol_size, d, patch_size, use_ssc=False, use_gaussian_kernel=False,
                 use_fixed_var=True, device='cuda'):
        self.epsilon = 0.000001
        self.vol_size = vol_size
        self.d = d
        self.patch_size = patch_size
        self.use_ssc = use_ssc
        self.use_gaussian_kernel = use_gaussian_kernel
        self.use_fixed_var = use_fixed_var

        if use_gaussian_kernel:
            dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

            vals = torch.exp(dist.log_prob(torch.arange(-(patch_size - 1) / 2, (patch_size - 1) / 2 + 1)))
            kernel = torch.einsum('i,j,k->ijk', vals, vals, vals)
            kernel = kernel / torch.sum(kernel)
            self.kernel = kernel.unsqueeze(0).unsqueeze(0)
        else:
            self.kernel = torch.ones([1, 1, patch_size, patch_size, patch_size]) / (patch_size ** 3)
        self.kernel = self.kernel.to(device)

    def ssd_shift(self, image, direction):
        # expects a 3d image
        x, y, z = self.vol_size
        new_shift = np.clip(direction, 0, None)
        old_shift = -np.clip(direction, None, 0)

        # translate images
        new_image = image[:, :, new_shift[0]:x - old_shift[0], new_shift[1]:y - old_shift[1],
                    new_shift[2]:z - old_shift[2]]
        old_image = image[:, :, old_shift[0]:x - new_shift[0], old_shift[1]:y - new_shift[1],
                    old_shift[2]:z - new_shift[2]]
        # get squared difference
        diff = torch.square(new_image - old_image)

        # pad the diff
        # padding = np.concatenate((np.array([[0,0]]), np.transpose([old_shift, new_shift])))
        padding = (old_shift[2], new_shift[2], old_shift[1], new_shift[1], old_shift[0], new_shift[0])
        diff = F.pad(diff, padding)

        # apply convolution
        conv = F.conv3d(diff, self.kernel, padding=(self.patch_size // 2))
        return conv

    def mind_loss(self, y_true, y_pred):
        ndims = 3
        y_true = y_true
        y_pred = y_pred
        loss_tensor = 0

        if self.use_fixed_var:
            y_true_var = 0.004
            y_pred_var = 0.004
        else:
            y_true_var = 0
            y_pred_var = 0
            for i in range(ndims):
                direction = [0] * ndims
                direction[i] = self.d

                y_true_var += self.ssd_shift(y_true, direction)
                y_pred_var += self.ssd_shift(y_pred, direction)

                direction = [0] * ndims
                direction[i] = -self.d
                y_true_var += self.ssd_shift(y_true, direction)
                y_pred_var += self.ssd_shift(y_pred, direction)

            y_true_var = y_true_var / (ndims * 2) + self.epsilon
            y_pred_var = y_pred_var / (ndims * 2) + self.epsilon

        # print(y_true_var)
        for i in range(ndims):
            direction = [0] * ndims
            direction[i] = self.d

            loss_tensor += torch.mean(torch.abs(torch.exp(-self.ssd_shift(y_true, direction) / y_true_var)
                                                - torch.exp(-self.ssd_shift(y_pred, direction) / y_pred_var)))

            direction = [0] * ndims
            direction[i] = -self.d
            loss_tensor += torch.mean(torch.abs(torch.exp(-self.ssd_shift(y_true, direction) / y_true_var)
                                                - torch.exp(-self.ssd_shift(y_pred, direction) / y_pred_var)))

        return loss_tensor / (ndims * 2)

    def loss(self, y_true, y_pred):
        return self.mind_loss(y_true, y_pred)


class LocalMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(LocalMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5, 6)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5, 6)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)


class MINDSCC(torch.nn.Module):
    """
    MIND descriptor with self context
    see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    """
    def __init__(self, delta=1, sigma=0.8, device='cuda'):
        super(MINDSCC, self).__init__()
        self.delta = delta
        self.sigma = sigma
        self.device = device

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]], dtype=torch.float, device=device)

        # squared distances
        dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6, device=device), torch.arange(6, device=device))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()
        self.mshift1 = torch.zeros((12, 1, 3, 3, 3), device=device)
        self.mshift1.view(-1)[
            torch.arange(12, device=device) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        self.mshift2 = torch.zeros((12, 1, 3, 3, 3), device=device)
        self.mshift2.view(-1)[
            torch.arange(12, device=device) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        self.rpad = nn.ReplicationPad3d(delta)

    def compute(self, img):
        # compute patch-ssd
        ssd = smooth(
            ((F.conv3d(self.rpad(img), self.mshift1, dilation=self.delta) -
              F.conv3d(self.rpad(img), self.mshift2, dilation=self.delta)) ** 2),
            self.sigma)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        # mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
        clamp_min, clamp_max = mind_var.mean() * 0.001, mind_var.mean() * 1000
        mind_var = torch.where(mind_var > clamp_max, clamp_max, mind_var)
        mind_var = torch.where(mind_var < clamp_min, clamp_min, mind_var)
        mind /= mind_var
        mind = torch.exp(-mind)
        return mind

    def loss(self, x, y):
        # MSE of the MIND SSC descriptor
        return torch.mean((self.compute(x) - self.compute(y)) ** 2)

    def forward(self, x, y):
        return self.loss(x, y)
    
class MIND_metric:
    """
    Modality-Independent NeighborHood Descriptor
    """

    def __init__(self, vol_size, d, patch_size, use_ssc=False, use_gaussian_kernel=False,
                 use_fixed_var=True, device='cuda'):
        self.epsilon = 0.000001
        self.vol_size = vol_size
        self.d = d
        self.patch_size = patch_size
        self.use_ssc = use_ssc
        self.use_gaussian_kernel = use_gaussian_kernel
        self.use_fixed_var = use_fixed_var

        if use_gaussian_kernel:
            dist = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

            vals = torch.exp(dist.log_prob(torch.arange(-(patch_size - 1) / 2, (patch_size - 1) / 2 + 1)))
            kernel = torch.einsum('i,j,k->ijk', vals, vals, vals)
            kernel = kernel / torch.sum(kernel)
            self.kernel = kernel.unsqueeze(0).unsqueeze(0)
        else:
            self.kernel = torch.ones([1, 1, patch_size, patch_size, patch_size]) / (patch_size ** 3)
        self.kernel = self.kernel.to(device)

    def ssd_shift(self, image, direction):
        # expects a 3d image
        x, y, z = self.vol_size
        new_shift = np.clip(direction, 0, None)
        old_shift = -np.clip(direction, None, 0)

        # translate images
        new_image = image[:, :, new_shift[0]:x - old_shift[0], new_shift[1]:y - old_shift[1],
                    new_shift[2]:z - old_shift[2]]
        old_image = image[:, :, old_shift[0]:x - new_shift[0], old_shift[1]:y - new_shift[1],
                    old_shift[2]:z - new_shift[2]]
        # get squared difference
        diff = torch.square(new_image - old_image)

        # pad the diff
        # padding = np.concatenate((np.array([[0,0]]), np.transpose([old_shift, new_shift])))
        padding = (old_shift[2], new_shift[2], old_shift[1], new_shift[1], old_shift[0], new_shift[0])
        diff = F.pad(diff, padding)

        # apply convolution
        conv = F.conv3d(diff, self.kernel, padding=(self.patch_size // 2))
        return conv

    def mind_loss(self, y_true, y_pred):
        ndims = 3
        y_true = y_true
        y_pred = y_pred
        loss_tensor = 0

        if self.use_fixed_var:
            y_true_var = 0.004
            y_pred_var = 0.004
        else:
            y_true_var = 0
            y_pred_var = 0
            for i in range(ndims):
                direction = [0] * ndims
                direction[i] = self.d

                y_true_var += self.ssd_shift(y_true, direction)
                y_pred_var += self.ssd_shift(y_pred, direction)

                direction = [0] * ndims
                direction[i] = -self.d
                y_true_var += self.ssd_shift(y_true, direction)
                y_pred_var += self.ssd_shift(y_pred, direction)

            y_true_var = y_true_var / (ndims * 2) + self.epsilon
            y_pred_var = y_pred_var / (ndims * 2) + self.epsilon

        # print(y_true_var)
        for i in range(ndims):
            direction = [0] * ndims
            direction[i] = self.d

            loss_tensor += torch.abs(torch.exp(-self.ssd_shift(y_true, direction) / y_true_var)
                                                - torch.exp(-self.ssd_shift(y_pred, direction) / y_pred_var))

            direction = [0] * ndims
            direction[i] = -self.d
            loss_tensor += torch.abs(torch.exp(-self.ssd_shift(y_true, direction) / y_true_var)
                                                - torch.exp(-self.ssd_shift(y_pred, direction) / y_pred_var))

        return loss_tensor / (ndims * 2)

    def loss(self, y_true, y_pred):
        return self.mind_loss(y_true, y_pred)
    
class NCC_metric:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5, device='cuda', thresh=None):
        self.win = win
        self.eps = torch.tensor([eps]).float().to(device)
        self.device = torch.device(device)
        self.thresh = thresh

    def loss(self, y_true, y_pred):
        if self.thresh:
            Ii = y_true * (y_true < self.thresh).float()
            Ji = y_pred * (y_true < self.thresh).float()
        else:
            Ii = y_true
            Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(self.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = torch.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = torch.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = torch.maximum(J_var, self.eps)

        cc = (cross / I_var) * (cross / J_var)

        return -(cc)