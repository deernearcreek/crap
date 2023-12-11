import sys

import numpy as np
from scipy.ndimage import distance_transform_edt as bwdist

# sys.path.append('./ext/neuron')
# sys.path.append('./ext/pynd-lib')
# sys.path.append('./ext/pytools-lib')
# import neuron.layers as nrn_layers
import voxelmorph as vxm

class SourceWarpPoint:
    def __init__(self, point, decay_power=1, deformation_magnitude=6):

        # Default values
        self._decay_power = decay_power
        self._deformation_magnitude = deformation_magnitude

        self._point = point

    @property
    def point(self):
        return self._point

    @property
    def decay_power(self):
        return self._decay_power

    @property
    def deformation_magnitude(self):
        return self._deformation_magnitude

    @property
    def warp_point(self):
        return [*self._point, self._decay_power, self._deformation_magnitude]

class RandomSourceWarpPoint:
    # Randomly generate source points with random decay power and deformation magnitude. Source points are randomly located with a given mask (e.g ventricle)
    def __init__(self, n_points, image_shape, mask = None, decay_power_range=[0.5,2], deformation_magnitude_range=[1,5]):
        self.n_points = n_points
        #self.mask = mask
        self.decay_power_range = decay_power_range
        self.deformation_magnitude_range = deformation_magnitude_range

        self.sources = self._GenerateSources(self.n_points, image_shape, mask, self.decay_power_range, self.deformation_magnitude_range)

    def _GenerateSources(self, n_points, image_shape, mask, decay_power_range, deformation_range):
        points = []
        deformation_magnitude = []
        decay_power = []
        sources = []

        if mask is None:
            x, y, z = np.meshgrid(np.linspace(0,image_shape[0]-1,image_shape[0]),
                                 np.linspace(0,image_shape[1]-1,image_shape[1]),
                                 np.linspace(0,image_shape[2]-1,image_shape[2]))
        else:
            islice = np.argmax(np.sum(mask>0, axis=(0,1)))
            mask_center = np.zeros(np.shape(mask))
            mask_center[:,:,islice-5:islice+4] = mask[:,:,islice-5:islice+4]
            x, y, z = np.where(mask_center>0)
            #x, y, z = np.where(mask>0)
        index = np.random.randint(len(x),size=n_points)
        for i in range(n_points):
            points.append([x[index[i]], y[index[i]], z[index[i]]])
            alpha = np.random.uniform(low=deformation_range[0], high=deformation_range[1])
            beta_exp = np.random.uniform(low=np.log(decay_power_range[0]), high=np.log(np.minimum(decay_power_range[1],np.log2(alpha/(alpha-1)))))
            beta = np.exp(beta_exp)
            #beta = np.random.uniform(low=decay_power_range[0], high=np.minimum(decay_power_range[1],np.log2(alpha/(alpha-1)))) # anti-folding constraint
            deformation_magnitude.append(alpha*np.random.choice([-1,1]))
            decay_power.append(beta)
            sources.append(SourceWarpPoint(point=points[i], decay_power=decay_power[i], deformation_magnitude=deformation_magnitude[i]))
        return sources

    @property
    def print_sources(self):
        print('source point [x,y,z],  decay_power,  deformation_magnitude')
        for i in range(0,len(self.sources)):
            print(self.sources[i].warp_point)


