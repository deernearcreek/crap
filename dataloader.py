import os
import glob
import numpy as np
from skimage.transform import resize

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torchio as tio


class RadOncTrainingDataset(Dataset):
    def __init__(self, file_path, num_samples=None, supervision=False, return_segmentation=False,
                 transform=False, target_transform=ToTensor,uncertainty = False):
        if num_samples is not None:
            self.img_names = glob.glob(os.path.join(file_path, '*.npz'))[:num_samples]
        else:
            self.img_names = glob.glob(os.path.join(file_path, '*.npz'))
        self.return_segmentation = return_segmentation
        self.supervision = supervision
        self.uncertainty = uncertainty
        if transform:
            # use TorchIO to normalize MR images
            landmarks = np.array([0, 0.33792968, 0.62630105, 0.75103401, 0.88691668, 1.24087058, 1.92283445, 7.28772973,
                                  23.91458834, 32.59366753, 39.85579238, 53.92610512, 100])  # precomputed
            histogram_transform = tio.HistogramStandardization({'mr': landmarks})
            znorm_transform = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
            rescale_transform = tio.transforms.RescaleIntensity((0, 1), percentiles=(0, 99.5))
            transform = tio.Compose([histogram_transform, znorm_transform, rescale_transform])
            self.transform = transform
        else:
            self.transform = None
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        with np.load(img_path) as data:
            cbct_fixed = data['cbct']
            mr_moving = data['moving_img']
            
            if self.uncertainty:
                flow = data['flow']
            if self.supervision:
                ct_fixed = data['fixed_img']
                ct_moving = data['fixed_warp']

            if self.return_segmentation:
                seg_fixed = data['ventricle_fixed']
                seg_moving = data['ventricle_moving']

                # Extract segmentations of a few structures and stack them
                mask_val = [1, 2, 23, 5, 19, 29, 8]
                for val in [6, 20, 24, 30, 9]:
                    seg_fixed[seg_fixed == val] -= 1
                    seg_moving[seg_moving == val] -= 1
                mask_fixed = np.zeros([len(mask_val)] + list(seg_fixed.shape))
                mask_moving = np.zeros([len(mask_val)] + list(seg_moving.shape))
                for i, val in enumerate(mask_val):
                    mask_fixed[i, :, :] = seg_fixed == val
                    mask_moving[i, :, :, :] = seg_moving == val
                del seg_fixed, seg_moving

        if self.transform is not None:
            subject = tio.Subject(mr=tio.ScalarImage(tensor=np.expand_dims(mr_moving, axis=0) * 2400 - 100))
            mr_moving = np.squeeze(self.transform(subject).mr.data)
            # cbct_fixed = self.transform(cbct_fixed)
            # mr_moving = self.transform(mr_moving)
            # if self.supervision:
            #     ct_fixed = self.transform(ct_fixed)
            #     ct_moving = self.transform(ct_moving)
            # if self.return_segmentation:
            #     seg_fixed = self.transform(seg_fixed)
            #     seg_moving = self.transform(seg_moving)

        out = [cbct_fixed, mr_moving]
        if self.supervision:
            out.append(ct_fixed)
            out.append(ct_moving)
        if self.return_segmentation:
            out.append(mask_fixed)
            out.append(mask_moving)
        if self.uncertainty:
            out.append(flow)

        return out


class RadOncSynthesisDataset(Dataset):
    def __init__(self, file_path, num_samples=None, target_transform=ToTensor):
        if num_samples is not None:
            self.img_names = glob.glob(os.path.join(file_path, '*.npz'))[:num_samples]
        else:
            self.img_names = glob.glob(os.path.join(file_path, '*.npz'))

        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        with np.load(img_path) as data:
            cbct = data['cbct']
            mr = data['moving_truth']
            ct = data['fixed_img']
            flow = data['flow']

        return [cbct, mr, ct,flow]


class RadOncValidationDataset(Dataset):
    def __init__(self, file_path, num_samples=None, supervision=False, return_segmentation=False, uncert = False):
        if num_samples is not None:
            self.img_names = glob.glob(os.path.join(file_path, '*.pt'))[:num_samples]
        else:
            self.img_names = glob.glob(os.path.join(file_path, '*.pt'))
        self.return_segmentation = return_segmentation
        self.supervision = supervision
        self. uncert = uncert

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        data = torch.load(img_path)
        cbct_fixed = data['cbct_fixed']
        mr_moving = data['mr_moving']
        if self.uncert:
            flow = data['flow']
            mr_gt = data['moving_truth']

        if self.supervision:
            ct_fixed = torch.from_numpy(data['ct_fixed'])
            ct_moving = torch.from_numpy(data['ct_moving'])

        if self.return_segmentation:
            seg_fixed = data['seg_fixed']
            seg_moving = data['seg_moving']

            # Extract segmentations of a few structures and stack them
            mask_val = [1, 2, 23, 5, 19, 29, 8]
            for val in [6, 20, 24, 30, 9]:
                seg_fixed[seg_fixed == val] -= 1
                seg_moving[seg_moving == val] -= 1
            mask_fixed = torch.cat([seg_fixed == val for val in mask_val], dim=1)
            mask_moving = torch.cat([seg_moving == val for val in mask_val], dim=1)

        out = [cbct_fixed, mr_moving]
        if self.supervision:
            out.append(ct_fixed)
            out.append(ct_moving)
        if self.return_segmentation:
            out.append(mask_fixed)
            out.append(mask_moving)
        if self.uncert:
            out.append(mr_gt)
            out.append(flow)
        return out


class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):
        return resize(img, self.output_size, order=1)


class RegistrationDatasetMR(Dataset):
    """
    Only return CBCT (fixed) and MR (moving / fixed). Goal is to register in the MR domain only
    """

    def __init__(self, file_path, num_samples=None, supervision=False, return_segmentation=False,
                 target_transform=ToTensor):
        if num_samples is not None:
            self.img_names = glob.glob(os.path.join(file_path, '*.npz'))[:num_samples]
        else:
            self.img_names = glob.glob(os.path.join(file_path, '*.npz'))
        self.return_segmentation = return_segmentation
        self.supervision = supervision

        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        with np.load(img_path) as data:
            cbct_fixed = data['cbct']
            mr_moving = data['moving_img']
            mr_fixed = data['moving_truth']

            if self.return_segmentation:
                seg_fixed = data['ventricle_fixed']
                seg_moving = data['ventricle_moving']

                # Extract segmentations of a few structures and stack them
                mask_val = [1, 2, 23, 5, 19, 29, 8]
                for val in [6, 20, 24, 30, 9]:
                    seg_fixed[seg_fixed == val] -= 1
                    seg_moving[seg_moving == val] -= 1
                mask_fixed = np.zeros([len(mask_val)] + list(seg_fixed.shape))
                mask_moving = np.zeros([len(mask_val)] + list(seg_moving.shape))
                for i, val in enumerate(mask_val):
                    mask_fixed[i, :, :] = seg_fixed == val
                    mask_moving[i, :, :, :] = seg_moving == val
                del seg_fixed, seg_moving

        out = [cbct_fixed, mr_moving, mr_fixed]
        if self.return_segmentation:
            out.append(mask_fixed)
            out.append(mask_moving)

        return out


def headscanner_training_dataset(file_path, corrected=False, augmentation=False, leave=None):
    # leave: leave out a validation id, 2,4,5,208,212,222
    if leave is not None:
        MR_paths = sorted(glob.glob(os.path.join(file_path, '*_MR.nii.gz')))
        CT_paths = sorted(glob.glob(os.path.join(file_path, '*_CT.nii.gz')))
        if corrected:
            CBCT_paths = sorted(glob.glob(os.path.join(file_path, '*_CBCT_corr.nii.gz')))
        else:
            CBCT_paths = sorted(glob.glob(os.path.join(file_path, '*_CBCT_uncorr.nii.gz')))
    else:
        MR_paths = set(glob.glob(os.path.join(file_path, '*_MR.nii.gz'))) - \
                   set([os.path.join(file_path, '{}_MR.nii.gz'.format(leave))])
        CT_paths = set(glob.glob(os.path.join(file_path, '*_CT.nii.gz'))) - \
                   set([os.path.join(file_path, '{}_CT.nii.gz'.format(leave))])
        if corrected:
            CBCT_paths = set(glob.glob(os.path.join(file_path, '*_CBCT_corr.nii.gz'))) - \
                         set([os.path.join(file_path, '{}_CBCT_corr.nii.gz'.format(leave))])
        else:
            CBCT_paths = set(glob.glob(os.path.join(file_path, '*_CBCT_uncorr.nii.gz'))) - \
                         set([os.path.join(file_path, '{}_CBCT_uncorr.nii.gz'.format(leave))])

    subjects = []
    subjects_id = []
    for (MR_path, CT_path, CBCT_path) in zip(MR_paths, CT_paths, CBCT_paths):
        subject = tio.Subject(
            MR=tio.ScalarImage(MR_path),
            CBCT=tio.ScalarImage(CBCT_path),
            CT=tio.ScalarImage(CT_path),
        )
        subjects.append(subject)
        if os.name == 'nt':
            subjects_id.append(int(MR_path.split('\\')[-1][:-10]))
        else:
            subjects_id.append(int(MR_path.split('/')[-1][:-10]))
    if augmentation:
        training_transform = tio.Compose([
            tio.RandomAffine(scales=(0.8, 1.0), degrees=5, translation=5, isotropic=True, p=0.7),
        ])
    else:
        training_transform = tio.Compose([])

    training_set = tio.SubjectsDataset(
        subjects, transform=training_transform)
    return training_set, subjects_id


def headscanner_validation_dataset(file_path, segmentation_path, corrected=False, leave=2):
    # leave: leave out a validation id, 2,4,5,208,212,222

    MR_paths = [os.path.join(file_path, '{}_MR.nii.gz'.format(leave))]
    CT_paths = [os.path.join(file_path, '{}_CT.nii.gz'.format(leave))]
    segmentation_moving_paths = [os.path.join(segmentation_path, '{}_segmentation_moving.nii.gz'.format(leave))]
    segmentation_fixed_paths = [os.path.join(segmentation_path, '{}_segmentation_fixed.nii.gz'.format(leave))]
    if corrected:
        CBCT_paths = [os.path.join(file_path, '{}_CBCT_corr.nii.gz'.format(leave))]
    else:
        CBCT_paths = [os.path.join(file_path, '{}_CBCT_uncorr.nii.gz'.format(leave))]

    subjects = []
    subjects_id = []
    for (MR_path, CT_path, CBCT_path,
         segmentation_moving_path, segmentation_fixed_path) in zip(MR_paths, CT_paths, CBCT_paths,
                                                                   segmentation_moving_paths, segmentation_fixed_paths):
        subject = tio.Subject(
            MR=tio.ScalarImage(MR_path),
            CBCT=tio.ScalarImage(CBCT_path),
            CT=tio.ScalarImage(CT_path),
            segmentation_moving=tio.ScalarImage(segmentation_moving_path),
            segmentation_fixed=tio.ScalarImage(segmentation_fixed_path)
        )
        subjects.append(subject)
        if os.name == 'nt':
            subjects_id.append(int(MR_path.split('\\')[-1][:-10]))
        else:
            subjects_id.append(int(MR_path.split('/')[-1][:-10]))

    valid_set = tio.SubjectsDataset(subjects)
    return valid_set, subjects_id


def simulation_metal_training_dataset(file_path, augmentation=True):
    MR_paths = sorted(glob.glob(os.path.join(file_path, '*T1*.nii.gz')))
    CT_paths = sorted(glob.glob(os.path.join(file_path, '*_CT_*.nii.gz')))
    CBCT_paths = sorted(glob.glob(os.path.join(file_path, '*CBCT*.nii.gz')))

    subjects = []
    for (MR_path, CT_path, CBCT_path) in zip(MR_paths, CT_paths, CBCT_paths):
        subject = tio.Subject(
            MR=tio.ScalarImage(MR_path),
            CBCT=tio.ScalarImage(CBCT_path),
            CT=tio.ScalarImage(CT_path),
        )
        subjects.append(subject)

    if augmentation:
        training_transform = tio.Compose([
            tio.RandomAffine(scales=(0.8, 1.0), degrees=5, translation=5, isotropic=True, p=0.7),
        ])
    else:
        training_transform = tio.Compose([])

    training_set = tio.SubjectsDataset(
        subjects, transform=training_transform)
    return training_set, []


def simulation_metal_validation_dataset(file_path):
    MR_paths = sorted(glob.glob(os.path.join(file_path, '*T1*.nii.gz')))
    CT_paths = sorted(glob.glob(os.path.join(file_path, '*_CT_*.nii.gz')))
    CBCT_paths = sorted(glob.glob(os.path.join(file_path, '*CBCT*.nii.gz')))

    subjects = []
    for (MR_path, CT_path, CBCT_path) in zip(MR_paths, CT_paths, CBCT_paths):
        subject = tio.Subject(
            MR=tio.ScalarImage(MR_path),
            CBCT=tio.ScalarImage(CBCT_path),
            CT=tio.ScalarImage(CT_path),
        )
        subjects.append(subject)

    valid_set = tio.SubjectsDataset(subjects)
    return valid_set, []


def sample_transform(image_size, trans_scale=5, rotation_scale=5):
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
    theta = np.concatenate((trans, rot), axis=None)
    return theta


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
