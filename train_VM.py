import os
import random
from argparse import ArgumentParser
import time
from tqdm import tqdm
import torch
import losses
import torchio as tio

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8
from networks_gan import VoxelMorph

from torch.utils.data import DataLoader
from dataloader import RadOncTrainingDataset, RadOncValidationDataset, headscanner_training_dataset, headscanner_validation_dataset
from torch.utils.tensorboard import SummaryWriter


# args['train_path'] = r"//istar-blackhole/data2/RadOnc_Brain/NonRigid_Training_CBCT"
# args['test_path'] = r"//istar-blackhole/data2/RadOnc_Brain/Validation"

def train_epoch(args, model, train_dataloader, opt, device):
    model.train()
    sim_loss = 0
    smooth_loss = 0
    total_loss = 0
    num_samples = 0

    for batch in tqdm(train_dataloader):
        if args['dataset'] == 'RadOnc':
            cbct, mr = batch  # mr, cbct, ct_preop, ct_intraop
            cbct = cbct.float().unsqueeze(1).to(device)
            mr = mr.float().unsqueeze(1).to(device)
        else:
            mr = batch['MR'][tio.DATA].float().to(device)
            cbct = batch['CBCT'][tio.DATA].float().to(device)

        batch_size = mr.size(0)
        num_samples += batch_size

        opt.zero_grad()
        mr_reg, flow = model(mr, cbct)

        loss_sim = losses.LocalMutualInformation().local_mi(mr_reg, cbct)
        # loss_sim = losses.MIND((128, 160, 128), d=1, patch_size=7, use_gaussian_kernel=True).loss(mr_reg, cbct)
        loss_smooth = vxm.losses.Grad('l2', loss_mult=1).loss(None, flow)
        loss = loss_sim + args['lambda_smooth'] * loss_smooth
        loss.backward()
        opt.step()

        sim_loss += loss_sim.item() * batch_size
        smooth_loss += loss_smooth.item() * batch_size
        total_loss += loss.item() * batch_size

    return total_loss / num_samples, sim_loss / num_samples, smooth_loss / num_samples


def valid_epoch(model, valid_dataloader, device):
    sim_loss = 0
    seg_loss = 0
    smooth_loss = 0
    num_samples = 0

    for i, batch in tqdm(enumerate(valid_dataloader)):
        if args['dataset'] == 'RadOnc':
            cbct, mr, seg_fixed, seg_moving = batch
            cbct = cbct.float().to(device)
            mr = mr.float().to(device)
            seg_fixed = seg_fixed.float().to(device)
            seg_moving = seg_moving.float().to(device)
        else:
            cbct = batch['CBCT'][tio.DATA].float().unsqueeze(0).to(device)
            mr = batch['MR'][tio.DATA].float().unsqueeze(0).to(device)
            seg_fixed = batch['segmentation_fixed'][tio.DATA].float().unsqueeze(0).to(device)
            seg_moving = batch['segmentation_moving'][tio.DATA].float().unsqueeze(0).to(device)

        batch_size = mr.size(0)
        num_samples += batch_size

        with torch.no_grad():
            mr_reg, flow = model(mr, cbct)
            warped_seg = model.transformer(seg_moving, flow)

            loss_sim = losses.LocalMutualInformation().local_mi(mr_reg, cbct)
            loss_smooth = vxm.losses.Grad('l2', loss_mult=1).loss(None, flow)
            loss_seg = losses.Dice().loss(warped_seg, seg_fixed)

        sim_loss += loss_sim.item()
        seg_loss += loss_seg.item()
        smooth_loss += loss_smooth.item()

    return sim_loss / num_samples, seg_loss / num_samples, smooth_loss / num_samples


def get_config(train_path, test_path, exp_name, save_path='./checkpoint/', pretrained=None,
               lr=1e-4, lambda_smooth=20, num_epoch=100, start_epoch=0, batch_size=1, dataset='RadOnc', id=2):
    args = dict()
    args['train_path'] = train_path
    args['test_path'] = test_path
    args['exp_name'] = exp_name
    args['save_path'] = save_path
    args['pretrained'] = pretrained

    args['lr'] = lr  # learning rate
    args['lambda_smooth'] = lambda_smooth  # smoothness regularization
    args['num_epoch'] = num_epoch
    args['start_epoch'] = start_epoch
    args['batch_size'] = batch_size
    args['dataset'] = dataset
    args['id'] = id

    return args


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("train_path", type=str,
                        default=r"\\istar-tauceti\c_users\U01_BrainRegistration\NormalAnatomy_Normalized",
                        help="data folder")
    parser.add_argument("test_path", type=str,
                        default=r"\\istar-tauceti\c_users\U01_BrainRegistration\RadOnc_NonRigid_Valid",
                        help="valid data folder")
    parser.add_argument("exp_name", type=str, default="0",
                        help="experiment name")
    parser.add_argument("--save_path", type=str, default="./checkpoint",
                        help="model saved path")
    parser.add_argument("--pretrained", type=str, default=None, help='dir to pretrain model')
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--lambda_smooth", type=float,
                        dest="lambda_smooth", default=1, help="smoothness regularization")
    parser.add_argument("--epochs", type=int,
                        dest="num_epoch", default=100, help="number of epochs")
    parser.add_argument("--start_epoch", type=int,
                        dest="start_epoch", default=0, help="starting epoch")
    parser.add_argument("--batch", type=int,
                        dest="batch_size", default=1, help="batch_size")
    parser.add_argument("--dataset", type=str, default='RadOnc', dest="dataset", help='RadOnc or HeadScanner')
    parser.add_argument("--id", type=int, dest="id", default=2, help="leave out id in headscanner")

    args = parser.parse_args()
    args = get_config(**vars(args))
    print(args)
    device = torch.device('cuda')

    # Build dataset
    if args['dataset'] == 'RadOnc':
        train_dataset = RadOncTrainingDataset(args['train_path'], num_samples=None, transform=False,
                                              supervision=False, return_segmentation=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=1, pin_memory=True)
        valid_dataset = RadOncValidationDataset(args['test_path'], num_samples=None, supervision=False,
                                                return_segmentation=True)
    else:
        train_dataset, _ = headscanner_training_dataset(args['train_path'], corrected=True,
                                                     augmentation=True, leave=args['id'])
        train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=1, pin_memory=True)
        valid_dataset, _ = headscanner_validation_dataset(args['train_path'], args['test_path'],
                                                       corrected=True, leave=args['id'])

    # Build Network
    model = VoxelMorph(res=False).to(device)

    # Load pretrained model
    if args['pretrained'] is not None:
        if os.path.exists(args['pretrained']):
            checkpoint = torch.load(args['pretrained'])
            model.load_state_dict(checkpoint['state_dict'])
            print('pretrained model loaded...')
            del checkpoint

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    boardio = SummaryWriter(log_dir='./log/VM_MR_CBCT_{}_{}'.format(args['lr'], args['lambda_smooth']))

    for epoch in range(args['start_epoch'], args['start_epoch'] + args['num_epoch']):
        total_loss, sim_loss, smooth_loss = train_epoch(args, model, train_dataloader, optimizer, device)
        boardio.add_scalar('total_loss', total_loss, epoch)
        boardio.add_scalar('sim_loss', sim_loss, epoch)
        boardio.add_scalar('smooth_loss', smooth_loss, epoch)
        print('epoch: {}, total_loss: {}, sim_loss: {}, smoothness_loss: {}'.format(epoch, total_loss, sim_loss,
                                                                                    smooth_loss))

        valid_sim_loss, valid_seg_loss, valid_smooth_loss = valid_epoch(model, valid_dataset, device)
        boardio.add_scalar('valid_sim_loss', valid_sim_loss, epoch)
        boardio.add_scalar('valid_seg_loss', valid_seg_loss, epoch)
        boardio.add_scalar('valid_smooth_loss', valid_smooth_loss, epoch)
        # final model save
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'args': args
            }, os.path.join(args['save_path'], 'VM_MR_CBCT_lr{}_smooth{}_ep{}.pt'.format(args['lr'],
                                                                                         args['lambda_smooth'], epoch)))
