import os
import numpy as np
from tqdm import tqdm
import itertools
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import RadOncTrainingDataset, RadOncValidationDataset, headscanner_training_dataset, headscanner_validation_dataset
from networks_gan import UNet, MultiResDiscriminator
import losses
from torch.utils.tensorboard import SummaryWriter
import torchio as tio

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def train_epoch(args, G, F, D_src, D_tgt, train_dataloader,
                opt_G, opt_D_src, opt_D_tgt, device, scaler=None):
    G.train()
    F.train()
    D_src.train()
    D_tgt.train()

    # freeze specified layers for transfer learning
    G.set_required_grad(level=args['freeze_levels'])

    disc_loss = 0
    gan_loss = 0
    cycle_loss = 0
    structure_loss = 0
    total_loss = 0
    num_samples = 0

    for batch in tqdm(train_dataloader, 'synthesis phase:'):
        if args['dataset'] == 'RadOnc':
            if args['src'] == 'mr':
                src, _, tgt, _ = batch  # mr, cbct, ct_preop, ct_intraop
            else:
                _, src, _, tgt = batch
            src = src.float().unsqueeze(1).to(device)
            tgt = tgt.float().unsqueeze(1).to(device)
        else:
            if args['src'] == 'mr':
                src = batch['MR'][tio.DATA].float().to(device)
                tgt = batch['CT'][tio.DATA].float().to(device)
            else:
                src = batch['CBCT'][tio.DATA].float().to(device)
                tgt = batch['CT'][tio.DATA].float().to(device)

        batch_size = src.size(0)
        num_samples += batch_size

        opt_G.zero_grad()
        with torch.cuda.amp.autocast():  # mix precision overhead
            # =========== Update G ================
            tgt_fake = G(src)
            src_cycled = F(tgt_fake)
            src_fake = F(tgt)
            tgt_cycled = G(src_fake)

            D_fake_tgt = D_tgt(torch.cat((tgt_fake, src), dim=1))
            D_fake_src = D_src(torch.cat((src_fake, tgt), dim=1))

            # Compute adversarial loss
            loss_G_GAN = 0
            loss_F_GAN = 0
            for d_fake_src, d_fake_tgt in zip(D_fake_src, D_fake_tgt):
                loss_G_GAN += torch.mean((torch.ones_like(d_fake_tgt) - d_fake_tgt) ** 2)
                loss_F_GAN += torch.mean((torch.ones_like(d_fake_src) - d_fake_src) ** 2)
            loss_G_GAN = loss_G_GAN / len(D_fake_tgt)
            loss_F_GAN = loss_F_GAN / len(D_fake_src)

            # Compute cycle-consistency loss
            loss_G_Cycle = torch.mean(torch.abs(src_cycled - src))
            loss_F_Cycle = torch.mean(torch.abs(tgt_cycled - tgt))

            # Compute MIND loss
            loss_G_Structure = losses.MIND((128, 160, 128), 1, 7, device=device).loss(tgt_fake, src)
            loss_F_Structure = losses.MIND((128, 160, 128), 1, 7, device=device).loss(src_fake, tgt)

            # Compute L1 loss (if matching src / tgt)
            # loss_G_L1 = torch.mean(torch.abs(fake_tgt - tgt))
            # loss_F_L1 = torch.mean(torch.abs(fake_src - src))
            # loss_G_L1 = losses.MaskedL1(low_thresh=0.8, high_thresh=1.0).loss(fake_tgt, tgt)
            # loss_F_L1 = 0

            loss_G = loss_G_GAN + args['lambda_cycle'] * loss_G_Cycle + args['lambda_mind'] * loss_G_Structure + \
                     loss_F_GAN + args['lambda_cycle'] * loss_F_Cycle + args['lambda_mind'] * loss_F_Structure

        scaler.scale(loss_G).backward()
        scaler.step(opt_G)

        # =========== Update D ================
        for param in D_src.parameters():
            param.requires_grad = True
        for param in D_tgt.parameters():
            param.requires_grad = True
        opt_D_src.zero_grad()
        opt_D_tgt.zero_grad()

        with torch.cuda.amp.autocast():
            D_real_src = D_src(torch.cat((src, tgt), dim=1))
            D_real_tgt = D_tgt(torch.cat((tgt, src), dim=1))
            D_fake_src = D_src(torch.cat((src_fake.detach(), tgt), dim=1))
            D_fake_tgt = D_tgt(torch.cat((tgt_fake.detach(), src), dim=1))

            loss_D_real_src = 0
            loss_D_real_tgt = 0
            loss_D_fake_src = 0
            loss_D_fake_tgt = 0

            for d_real_src, d_real_tgt, d_fake_src, d_fake_tgt in zip(D_real_src, D_real_tgt,
                                                                      D_fake_src, D_fake_tgt):
                loss_D_real_src += torch.mean((torch.ones_like(d_real_src) - d_real_src) ** 2)
                loss_D_real_tgt += torch.mean((torch.ones_like(d_real_tgt) - d_real_tgt) ** 2)
                loss_D_fake_src += torch.mean((torch.zeros_like(d_fake_src) - d_fake_src) ** 2)
                loss_D_fake_tgt += torch.mean((torch.zeros_like(d_fake_tgt) - d_fake_tgt) ** 2)
            loss_D_src = (loss_D_real_src + loss_D_fake_src) / 2 / len(D_real_src)
            loss_D_tgt = (loss_D_real_tgt + loss_D_fake_tgt) / 2 / len(D_real_tgt)

        scaler.scale(loss_D_src).backward()
        scaler.step(opt_D_src)
        scaler.scale(loss_D_tgt).backward()
        scaler.step(opt_D_tgt)
        scaler.update()

        # Record losses
        disc_loss += (loss_D_src.item() + loss_D_tgt.item()) * batch_size / 2
        gan_loss += (loss_G_GAN.item() + loss_F_GAN.item()) * batch_size / 2
        cycle_loss += (loss_G_Cycle.item() + loss_F_Cycle.item()) * batch_size / 2
        structure_loss += (loss_G_Structure.item() + loss_F_Structure.item()) * batch_size / 2
        total_loss += loss_G.item() * batch_size

    return total_loss / num_samples, gan_loss / num_samples, cycle_loss / num_samples, \
           structure_loss / num_samples, disc_loss / num_samples


def valid_epoch(args, G, F, valid_dataloader, device):
    l1_loss = 0
    cycle_loss = 0
    num_samples = 0

    for i, batch in tqdm(enumerate(valid_dataloader)):
        if args['dataset'] == 'RadOnc':
            if args['src'] == 'mr':
                src, _, tgt, _, _, _ = batch  # mr, cbct, ct_preop, ct_intraop
            else:
                _, src, _, tgt, _, _ = batch
            src = src.float().to(device)
            tgt = tgt.float().to(device)
        else:
            if args['src'] == 'mr':
                src = batch['MR'][tio.DATA].float().unsqueeze(0).to(device)
                tgt = batch['CT'][tio.DATA].float().unsqueeze(0).to(device)
            else:
                src = batch['CBCT'][tio.DATA].float().unsqueeze(0).to(device)
                tgt = batch['CT'][tio.DATA].float().unsqueeze(0).to(device)

        batch_size = src.size(0)
        num_samples += batch_size

        with torch.no_grad():
            tgt_fake = G(src)
            src_cycled = F(tgt_fake)
            src_fake = F(tgt)
            tgt_cycled = G(src_fake)

            loss_L1 = torch.mean(torch.abs(tgt - tgt_fake)) + torch.mean(torch.abs(src - src_fake))
            loss_cycle = torch.mean(torch.abs(tgt - tgt_cycled)) + torch.mean(torch.abs(src - src_cycled))

        l1_loss += loss_L1.item() * batch_size
        cycle_loss += loss_cycle.item() * batch_size

    return l1_loss / num_samples, cycle_loss / num_samples


def get_config(train_path, test_path, exp_name, save_path='./checkpoint/', pretrained=None, dataset='RadOnc', id=2,
               src='mr', lr_G=1e-4, lr_D=1e-4, lambda_l1=20, lambda_mind=20, lambda_cycle=10, freeze_levels=-1,
               start_epoch=0, num_epoch=200, num_synth=150, batch_size=1, disc_channels=1, D_norm=True):
    args = dict()
    args['train_path'] = train_path
    args['test_path'] = test_path
    args['exp_name'] = exp_name
    args['save_path'] = save_path
    args['pretrained'] = pretrained
    args['dataset'] = dataset
    args['id'] = id
    args['src'] = src

    args['lr_G'] = lr_G  # initial learning rate for generator
    args['lr_D'] = lr_D  # initial learning rate for discriminator
    args['lambda_l1'] = lambda_l1  # synthesis L1
    args['lambda_cycle'] = lambda_cycle
    args['lambda_mind'] = lambda_mind
    args['start_epoch'] = start_epoch
    args['num_epoch'] = num_epoch
    args['num_synth'] = num_synth  # first # epochs is used for synthesis only
    args['batch_size'] = batch_size
    args['disc_channel'] = disc_channels
    args['D_norm'] = D_norm
    args['freeze_levels'] = freeze_levels

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
    parser.add_argument("--lr_G", type=float,
                        dest="lr_G", default=1e-4, help="generator learning rate")
    parser.add_argument("--lr_D", type=float,
                        dest="lr_D", default=1e-4, help="discriminator learning rate")
    parser.add_argument("--lambda_l1", type=float,
                        dest="lambda_l1", default=20, help="lambda L1 loss")
    parser.add_argument("--lambda_cycle", type=float,
                        dest="lambda_cycle", default=10, help="lambda Cycle loss")
    parser.add_argument("--lambda_mind", type=float,
                        dest="lambda_mind", default=20, help="lambda synthesis structural consistency loss")
    parser.add_argument("--epochs", type=int,
                        dest="num_epoch", default=200, help="number of epochs")
    parser.add_argument("--epochs_synth", type=int,
                        dest="num_synth", default=150, help="number of synthesis epochs")
    parser.add_argument("--start_epoch", type=int,
                        dest="start_epoch", default=0, help="starting epoch")
    parser.add_argument("--batch", type=int,
                        dest="batch_size", default=1, help="batch_size")
    parser.add_argument("--disc_channels", type=int,
                        dest="disc_channels", default=2, help="batch_size")
    parser.add_argument("--D_norm", default=False, action='store_true', dest="D_norm",
                        help="apply instance normalization in discriminator")
    parser.add_argument("--dataset", type=str, default='RadOnc', dest="dataset", help='RadOnc or HeadScanner')
    parser.add_argument("--id", type=int, dest="id", default=2, help="leave out id in headscanner")
    parser.add_argument("--src", type=str, default='mr', dest="src", help='mr or cbct to ct synthesis')
    parser.add_argument("--freeze_levels", type=int,
                        dest="freeze_levels", default=-1, help="-1 no freeze, -2 freeze encoder, 0 lowest level, "
                                                               "1 lowest two levels, .......")

    args = parser.parse_args()
    args = get_config(**vars(args))
    print(args)
    device = torch.device('cuda')

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = True

    # Build dataset
    if args['dataset'] == 'RadOnc':
        train_dataset = RadOncTrainingDataset(args['train_path'], num_samples=None, transform=False,
                                              supervision=True, return_segmentation=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=1, pin_memory=True)
        valid_dataset = RadOncValidationDataset(args['test_path'], num_samples=None, supervision=True,
                                                return_segmentation=True)
    else:
        train_dataset, _ = headscanner_training_dataset(args['train_path'], corrected=True,
                                                     augmentation=True, leave=args['id'])
        train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                      num_workers=1, pin_memory=True)
        valid_dataset, _ = headscanner_validation_dataset(args['train_path'], args['test_path'],
                                                       corrected=True, leave=args['id'])

    # Build model
    G = UNet(cdim=1, channels=(16, 32, 64, 128, 128), res=True).to(device)
    F = UNet(cdim=1, channels=(16, 32, 64, 128, 128), res=True).to(device)
    D_src = MultiResDiscriminator(cdim=args['disc_channel'], num_channels=16, norm='IN').to(device)
    D_tgt = MultiResDiscriminator(cdim=args['disc_channel'], num_channels=16, norm='IN').to(device)

    # Load pretrained model
    if args['pretrained'] is not None:
        if os.path.exists(args['pretrained']):
            checkpoint = torch.load(args['pretrained'])
            G.load_state_dict(checkpoint['G_state_dict'])
            F.load_state_dict(checkpoint['F_state_dict'])
            D_src.load_state_dict(checkpoint['D_src_state_dict'])
            D_tgt.load_state_dict(checkpoint['D_tgt_state_dict'])
            print('pretrained model loaded...')
            del checkpoint

    # Build Optimizer
    opt_G = optim.Adam(itertools.chain(G.parameters(), F.parameters()), lr=args['lr_G'])  # chain the forward/backward networks together
    # opt_G = optim.Adam(G.parameters(), lr=args['lr_G'])
    # opt_F = optim.Adam(F.parameters(), lr=args['lr_G'])
    opt_D_src = optim.Adam(D_src.parameters(), lr=args['lr_D'])
    opt_D_tgt = optim.Adam(D_tgt.parameters(), lr=args['lr_D'])

    if args['start_epoch'] > args['num_synth']:
        scheduler = optim.lr_scheduler.StepLR(opt_G, step_size=50, gamma=0.2)
    #     scheduler_warmup = GradualWarmupScheduler(opt_G, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    # Prepare Tensorboard
    boardio = SummaryWriter(log_dir='./log/CycleGAN_{}_{}_{}'.format(args['exp_name'], args['lambda_l1'],
                                                                        args['lambda_cycle']))
    scaler = torch.cuda.amp.GradScaler()  # mix precision -- reduce memory footprint at expense of lower precision

    # Start training
    for epoch in range(args['start_epoch'], args['start_epoch']+args['num_epoch']):
        total_loss, gan_loss, cycle_loss, structure_loss, disc_loss = train_epoch(args, G, F, D_src, D_tgt,
                                                                                  train_dataloader, opt_G, opt_D_src,
                                                                                  opt_D_tgt, device, scaler)
        print('epoch: {}, total_loss: {}, gan_loss: {}, cycle_loss: {}, '
              'structure_loss: {}, disc_loss: {}'.format(epoch, total_loss, gan_loss, cycle_loss,
                                                         structure_loss, disc_loss))
        boardio.add_scalar('total_loss', total_loss, epoch)
        boardio.add_scalar('gan_loss', gan_loss, epoch)
        boardio.add_scalar('cycle_loss', cycle_loss, epoch)
        boardio.add_scalar('structure_loss', structure_loss, epoch)
        boardio.add_scalar('disc_loss', disc_loss, epoch)

        valid_l1_loss, valid_cycle_loss = valid_epoch(args, G, F, valid_dataset, device)
        boardio.add_scalar('valid_l1_loss', valid_l1_loss, epoch)
        boardio.add_scalar('valid_cycle_loss', valid_cycle_loss, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'F_state_dict': F.state_dict(),
                'D_src_state_dict': D_src.state_dict(),
                'D_tgt_state_dict': D_tgt.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                # 'opt_F_state_dict': opt_F.state_dict(),
                'opt_D_src_state_dict': opt_D_src.state_dict(),
                'opt_D_tgt_state_dict': opt_D_tgt.state_dict(),
                'args': args
            }, os.path.join(args['save_path'], 'CycleGAN_{}_ep{}.pt'.format(args['exp_name'], epoch)))
