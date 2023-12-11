import numpy as np
from layers import VecInt
from scipy.ndimage import distance_transform_edt as bwdist
import torch


class SourceWarpPoint:
    def __init__(self, point, decay_power=1, deformation_magnitude=6):

        # Default values
        self.decay_power = decay_power
        self.deformation_magnitude = deformation_magnitude
        self.point = point

    def print(self):
        print('point: ', self.point, ' decay power: ', self.decay_power,
              ' deformation magnitude: ', self.deformation_magnitude)


class SourceGen:
    """
    Randomly generate source points with:
    n_points: number of source points to drive deformation
    random decay power: min and max value
    deformation magnitude: min and max value
    Source points are randomly located with a given mask (e.g ventricle)
    """
    def __init__(self, n_points, image_shape, decay_power_range=[0.5, 2], deformation_magnitude_range=[1, 5]):
        self.image_shape = image_shape
        self.decay_power_range = decay_power_range
        self.deformation_magnitude_range = deformation_magnitude_range
        self.n_points = n_points

    def run(self, ventricle_mask=None):
        points = []
        sources = []

        if ventricle_mask is None:
            x, y, z = np.meshgrid(np.linspace(0, self.image_shape[0]-1, self.image_shape[0]),
                                  np.linspace(0, self.image_shape[1]-1, self.image_shape[1]),
                                  np.linspace(0, self.image_shape[2]-1, self.image_shape[2]))
        else:
            islice = np.argmax(np.sum(ventricle_mask > 0, axis=(0, 1)))  # find the slice with the largest masked volume
            mask_center = np.zeros(ventricle_mask.shape)
            mask_center[:, :, islice-5:islice+4] = ventricle_mask[:, :, islice - 5:islice + 4]  # only place sources around ±5 slices from islice
            x, y, z = np.where(mask_center > 0)
            loc = np.where(z == islice)
            # nloc = [loc[0][0],loc[0][-1]]

        index = np.random.randint(len(x), size=self.n_points)
        for i in range(self.n_points):
            points.append([x[index[i]], y[index[i]], z[index[i]]])
            alpha = np.random.uniform(low=self.deformation_magnitude_range[0], high=self.deformation_magnitude_range[1])
            beta_exp = np.random.uniform(low=np.log(self.decay_power_range[0]),
                                         high=np.log(np.minimum(self.decay_power_range[1], np.log2(alpha/(alpha-1)))))
            beta = np.exp(beta_exp)

            sources.append(SourceWarpPoint(point=points[i], decay_power=beta,
                                           deformation_magnitude=alpha*np.random.choice([-1, 1])))
        return sources


class Simulator:
    """
    Simulate a random deformation field based on source model D=a/r^b from multiple source points.

    Returns
    -------
    D : the overall deformation field

    """

    def __init__(self, image_size=[192, 240, 192], int_steps=7):

        self._image_size = image_size
        # self.transformer = SpatialTransformer(image_size, mode=interp_method)
        self.vectint = VecInt(image_size, int_steps)  # square and scaling layer for exponentiation

    def simulate(self, sources, brain_mask=None, thresh=30):
        """
        sources: sources to drive deformation, generated from SourceGen
        brain_mask: optional mask of the brain region, deformation outside is set to 0
        thresh: a threshold of the distance to the brain boundary (value above thresh is not affected)
        """
        [X, Y, Z] = np.mgrid[0:self._image_size[0], 0:self._image_size[1], 0:self._image_size[2]]
        D = np.zeros((3, self._image_size[0], self._image_size[1], self._image_size[2]))
        
        for source in sources:
            decay_power = source.decay_power
            deformation_mag = source.deformation_magnitude

            # R2 = np.square((X-source.point[0])*1.5) + np.square((Y-source.point[1])*1.5) + np.square((Z-source.point[2])*1.5)
            R2 = np.square((X-source.point[0])) + np.square((Y-source.point[1])) + np.square((Z-source.point[2]))
            F = deformation_mag/(np.power(R2, decay_power/2)+1e-6) # deformation magnitude (+retraction,-dilation)
            F[source.point[0], source.point[1], source.point[2]] = deformation_mag
            
            DX = (X-source.point[0])/(np.sqrt(R2)+1e-5)*F
            DY = (Y-source.point[1])/(np.sqrt(R2)+1e-5)*F
            DZ = (Z-source.point[2])/(np.sqrt(R2)+1e-5)*F

            D = D + np.stack((DX, DY, DZ), axis=0)

        if brain_mask is not None:  # set deformation outside the brain to 0 with smooth transitions
            brain_dist = bwdist(brain_mask) / 30
            brain_dist[brain_dist > 1] = 1
            D = np.multiply(D, np.repeat(brain_dist[np.newaxis, :, :], 3, axis=0))

        D = self.vectint(torch.from_numpy(D).unsqueeze(0).float())
        return D



# if __name__ == '__main__':
#     sources = [
#         SourceWarpPoint(point=[120,90,101]),
#         SourceWarpPoint(point=[112,144,101]),
#         SourceWarpPoint(point=[85,140,90], deformation_magnitude=20),
#     ]
#
#     # Instantiate the Warping Transform
#     transform = WarpTransform()
