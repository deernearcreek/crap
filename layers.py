import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.interpolation import affine_transform

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    From VoxelMorph: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        # explicitly check data types, especially under AMP
        # https://github.com/pytorch/pytorch/issues/42218
        if src.dtype != new_locs.dtype:
            new_locs = new_locs.type(src.dtype)
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

# class SpatialTransformer(nn.Module):
#     """
#     N-D Spatial Transformer
#     From VoxelMorph: https://github.com/voxelmorph/voxelmorph/blob/dev/voxelmorph/torch/layers.py
#     """

#     def __init__(self, size, mode='bilinear'):
#         super().__init__()

#         self.mode = mode

#         # create sampling grid
#         vectors = [torch.arange(0, s) for s in size]
#         grids = torch.meshgrid(vectors)
#         grid = torch.stack(grids)
#         grid = torch.unsqueeze(grid, 0)
#         grid = grid.type(torch.FloatTensor)

#         # registering the grid as a buffer cleanly moves it to the GPU, but it also
#         # adds it to the state dict. this is annoying since everything in the state dict
#         # is included when saving weights to disk, so the model files are way bigger
#         # than they need to be. so far, there does not appear to be an elegant solution.
#         # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
#         self.register_buffer('grid', grid)

#     def forward(self, src, flow,T=torch.eye(3,4),device=torch.device('cuda')):
#         # new locations
#         grid = F.affine_grid(T.unsqueeze(0), src.size()).to(device)
#         src = F.grid_sample(src, grid, align_corners=True)
#         new_locs = self.grid + flow
#         shape = flow.shape[2:]

#         # need to normalize grid values to [-1, 1] for resampler
#         for i in range(len(shape)):
#             new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

#         # move channels dim to last position
#         # also not sure why, but the channels need to be reversed
#         if len(shape) == 2:
#             new_locs = new_locs.permute(0, 2, 3, 1)
#             new_locs = new_locs[..., [1, 0]]
#         elif len(shape) == 3:
#             new_locs = new_locs.permute(0, 2, 3, 4, 1)
#             new_locs = new_locs[..., [2, 1, 0]]
#         # explicitly check data types, especially under AMP
#         # https://github.com/pytorch/pytorch/issues/42218
#         if src.dtype != new_locs.dtype:
#             new_locs = new_locs.type(src.dtype)
#         return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Exponentiate a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps, transformer=None):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class GumbelSoftmax(nn.Module):
    """
    Adapted from https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
    Convert to categorical variable
    """
    def __init__(self, temperature=1.0, hard=False):
        """
        temperature: the smaller the value, the closer to quantized categorical sample
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def _sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits):
        y = logits + self._sample_gumbel(logits.size()).to(logits.device)
        return F.softmax(y / self.temperature, dim=-1)

    def forward(self, x):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return:
        """
        logits = nn.Softmax()(x)
        y = self._gumbel_softmax_sample(logits)

        if not self.hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard