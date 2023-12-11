import os
import torch
import torch.nn.functional as F

from scipy import ndimage
import numpy as np
import pystrum.pynd.ndutils as nd

def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def normalizeImage(img, fmax, fmin):
    # normalize image to 0 to 1
    img[img>fmax] = fmax
    img[img<fmin] = fmin
    img = (img-fmin)/(fmax-fmin)
    return img


def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "./saves/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


def pdist(x, p=2):
    if p == 1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p == 2:
        xx = (x ** 2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist


def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6, )
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N // 2
    padding = padding.long().tolist()

    view = torch.ones(5, )
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B * C, 1, D, H, W), padding, mode=padding_mode),
                    weight.view(view)).view(B, C, D, H, W)

def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img

def compute_surface_metrics(mask_gt, mask_pred):
    """Computes a set of surface distance metrics using https://github.com/deepmind/surface-distance.
  Args:
    mask_gt: 3-dim bool Numpy array. The ground truth mask of a single structure.
    mask_pred: 3-dim bool Numpy array. The predicted mask of a single structure.
    spacing_mm: 3-element list-like structure. Voxel spacing
      in x0 anx x1 (resp. x0, x1 and x2) directions.
  Returns:
    A dict with:
    "dice": Dice coefficient.
    "avg_dist": average bidirection surface distance.
    "hd95": 95 percentile Hausdorff distance.
    "hd": Hausdorff distance.
    "surface_dist": a dict of surface distances and surface areas
    """
    import surface_distance
    metrics = dict()

    metrics['dice'] = surface_distance.compute_dice_coefficient(mask_gt, mask_pred)

    dist = surface_distance.compute_surface_distances(mask_gt, mask_pred, (1,1,1))
    metrics['surface_dist'] = dist

    # Average surface distance
    metrics['avg_dist'] = np.asarray(surface_distance.compute_average_surface_distance(dist)).mean()

    # Hausdorff 95 distance
    metrics['hd95'] = surface_distance.compute_robust_hausdorff(dist, 95)

    # Hausdorff distance
    metrics['hd'] = surface_distance.compute_robust_hausdorff(dist, 100)

    return metrics

def compute_TRE(mask_gt, mask_pred, spacing=[1,1,1], threshold = 1000):
    """Computes TRE with target points defined as centroids of segmentations smaller than threshold.
  Args:
    mask_gt: 3-dim bool Numpy array. The ground truth mask.
    mask_pred: 3-dim bool Numpy array. The predicted mask.
    spacing_mm: 3-element list-like structure. Voxel spacing
      in x0 anx x1 (resp. x0, x1 and x2) directions.
  Returns:
    An array of TRE measurements
    """
    TRE = []
    for i in np.unique(mask_gt):
        if np.sum(mask_gt==i)<1000:
            before = ndimage.measurements.center_of_mass(mask_gt==i)
            after = ndimage.measurements.center_of_mass(mask_pred==i)
            dist = np.asarray(after)-np.asarray(before)
            TRE.append(np.linalg.norm(np.multiply(dist, spacing)))
    return np.asarray(TRE)


def random_transform(image_size, trans_scale=5, rotation_scale=5):
    """
    compute a 1*6 transformation parameters with rotation around image center
    trans_scale: in voxels
    rotation_scale: in degrees
    """
    trans = np.random.uniform(-1, 1, 3) * trans_scale
    rot = np.random.uniform(-1, 1, 3) * rotation_scale / 180 * np.pi
    R = np.array(compose_transform([0, 0, 0], rot))
    center = np.transpose([np.append(np.array(image_size) / 2.0, 1)], (1, 0))
    trans = np.transpose(np.matmul(R, -center) + center, (1, 0))[0, :-1] + trans
    return compose_transform(trans, rot)


def compose_transform(translation, rotation):
    """
    compute 4x4 transformation matrix from translation and rotation vectors
    """
    transforms_alpha = [[np.cos(rotation[0]), -np.sin(rotation[0]), 0, 0],
                        [np.sin(rotation[0]), np.cos(rotation[0]), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    transforms_beta = [[np.cos(rotation[1]), 0, np.sin(rotation[1]), 0], [0, 1, 0, 0],
                       [-np.sin(rotation[1]), 0, np.cos(rotation[1]), 0], [0, 0, 0, 1]]
    transforms_gamma = [[1, 0, 0, 0], [0, np.cos(rotation[2]), -np.sin(rotation[2]), 0],
                        [0, np.sin(rotation[2]), np.cos(rotation[2]), 0], [0, 0, 0, 1]]
    transform = np.matmul(np.matmul(transforms_alpha, transforms_beta), transforms_gamma)
    transform = np.matmul(
        [[1, 0, 0, translation[0]], [0, 1, 0, translation[1]], [0, 0, 1, translation[2]], [0, 0, 0, 1]], transform)
    return transform


def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims], 
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else: # must be 2 
        
        dfdx = J[0]
        dfdy = J[1] 
        
        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
    
    
def compute_negative_jacobian(disp):
    J = jacobian_determinant(disp)
    return np.sum(J<0)/np.prod(J.shape)