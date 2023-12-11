import os
import random
import argparse
import time
from tqdm import tqdm
import numpy as np
import torch

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

from torch.utils.data import DataLoader
from dataloader import ImageDataset
from networks_dualstream import DualStreamReg

args = dict()
# args['train_path'] = r"//istar-blackhole/data2/RadOnc_Brain/NonRigid_Training_CBCT"
# args['test_path'] = r"//istar-blackhole/data2/RadOnc_Brain/Validation"
args['train_path'] = '/mnt/blackhole-data2/RadOnc_Brain/NonRigid_Training_CBCT'
args['test_path'] = '/mnt/blackhole-data2/RadOnc_Brain/NonRigid_Valid_CBCT'
args['save_path'] = './checkpoint'
args['image_size'] = (128, 160, 128)
args['lr'] = 5e-4
args['lambda_flow'] = 2  # registration flow smoothness penalty
args['start_epoch'] = 0
args['num_epoch'] = 200
args['num_synth'] = 150
args['batch_size'] = 2
args['disc_channels'] = 2
args['bidir'] = False
args['int_downsize'] = 1
args['load_model'] = None
device = torch.device('cuda')

# Build dataset
train_dataset = ImageDataset(args['train_path'], num_samples=None, supervision=True, return_segmentation=False)
train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=2)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = True

# unet architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

# otherwise configure new model
model = DualStreamReg(
    channels=(16, 32, 64, 128, 128),
    skip_connect=True
)

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

image_loss_func = vxm.losses.MSE().loss
smooth_loss_func = vxm.losses.Grad('l2', loss_mult=args['int_downsize']).loss


# training loops
for epoch in range(args['start_epoch'], args['num_epoch']):

    # save model checkpoint
    if epoch % 20 == 0:
        torch.save({'state_dict': model.state_dict()},
                   os.path.join(args['save_path'], 'DualStream_CT_ep{}.pt'.format(epoch)))

    sim_loss = 0
    smoothness_loss = 0
    total_loss = 0
    num_samples = 0

    for i, batch in tqdm(enumerate(train_dataloader), 'reg phase:'):
        _, _, ct_fixed, ct_moving = batch
        batch_size = ct_fixed.size(0)
        num_samples += batch_size
        ct_fixed = ct_fixed.float().unsqueeze(1).to(device)
        ct_moving = ct_moving.float().unsqueeze(1).to(device)

        # run inputs through the model to produce a warped image and flow field
        ct_reg, flow = model(ct_moving, ct_fixed)

        # calculate total loss
        loss_sim = image_loss_func(ct_reg, ct_fixed)
        loss_smooth = smooth_loss_func(_, flow)
        loss = loss_sim + 2 * loss_smooth

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sim_loss += loss_sim.item() * batch_size
        smoothness_loss += loss_smooth * batch_size
        total_loss += loss.item() * batch_size

        # tqdm.write('sim_loss: {}, smooth_loss: {}'.format(loss_sim.item(), loss_smooth.item()))

    print('epoch: {}, total_loss: {}, sim_loss: {}, smoothness_loss: {}'.format(epoch, total_loss / num_samples,
                                                                                sim_loss / num_samples,
                                                                                smoothness_loss / num_samples))

# final model save
model.save(os.path.join(args['save_path'], 'Dualstream_CT_ep{}.pt'.format(epoch)))