import os
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import RadOncTrainingDataset, RadOncValidationDataset, headscanner_training_dataset, headscanner_validation_dataset
from networks_gan import UNet, MultiResDiscriminator, Discriminator
import losses
from torch.utils.tensorboard import SummaryWriter
import torchio as tio

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def train_epoch(args, G, D, train_dataloader, opt_G, opt_D, device):
    G.train()
    D.train()

    disc_loss = 0
    gan_loss = 0
    l1_loss = 0
    total_loss = 0
    num_samples = 0

    for batch in tqdm(train_dataloader, 'synthesis phase:'):
        if args['dataset'] == 'RadOnc':
            if args['src'] == 'mr':
                _, src, _, tgt = batch
            else:
                src, _, tgt, _ = batch
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

        # =========== Update G ================
        opt_G.zero_grad()

        tgt_fake = G(src)
        if args['disc_channel'] == 1:  # normal discriminator
            D_fake = D(tgt_fake)
        else:  # discriminator conditioned on the input to the synthesis network
            D_fake = D(torch.cat((tgt_fake, src), dim=1))

        # Compute adversarial loss
        loss_G_GAN = 0
        for d_fake in D_fake:
            loss_G_GAN += torch.mean((torch.ones_like(d_fake) - d_fake) ** 2)
        loss_G_GAN = loss_G_GAN / len(D_fake)

        # Compute L1 loss
        loss_G_L1 = torch.mean(torch.abs(tgt - tgt_fake))  # over the entire image
        # loss_G_L1 = losses.MaskedL1(high_thresh=0.8).loss(tgt, tgt_fake)  # exclude skull (>0.8)

        # Compute MIND loss
        loss_G_MIND = losses.MIND((128, 160, 128), 1, 7, device=device).loss(src, tgt_fake)

        loss_G = loss_G_GAN + args['lambda_l1'] * loss_G_L1 + args['lambda_mind'] * loss_G_MIND
        loss_G.backward()
        opt_G.step()

        # =========== Update D ================
        for param in D.parameters():
            param.requires_grad = True
        opt_D.zero_grad()

        if args['disc_channel'] == 1:
            D_real = D(tgt)
            D_fake = D(tgt_fake.detach())
        else:
            D_real = D(torch.cat((tgt, src), dim=1))
            D_fake = D(torch.cat((tgt_fake.detach(), src), dim=1))

        loss_D_real = 0
        loss_D_fake = 0

        for d_real, d_fake in zip(D_real, D_fake):
            loss_D_real += torch.mean((torch.ones_like(d_real) - d_real) ** 2)
            loss_D_fake += torch.mean((torch.zeros_like(d_fake) - d_fake) ** 2)
        loss_D = (loss_D_real + loss_D_fake) / len(D_real) / 2

        loss_D.backward()
        opt_D.step()

        disc_loss += loss_D.item() * batch_size

        gan_loss += loss_G_GAN.item() * batch_size
        l1_loss += loss_G_L1.item() * args['lambda_l1'] * batch_size
        total_loss += loss_G.item() * batch_size

    return total_loss / num_samples, gan_loss / num_samples, l1_loss / num_samples, disc_loss / num_samples


def valid_epoch(args, G, valid_dataloader, device):
    l1_loss = 0
    num_samples = 0

    for i, batch in tqdm(enumerate(valid_dataloader)):
        if args['dataset'] == 'RadOnc':
            if args['src'] == 'mr':
                _, src, _, tgt, _, _ = batch
            else:
                src, _, tgt, _, _, _ = batch
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

            loss_G_L1 = losses.MaskedL1().loss(tgt, tgt_fake)

        l1_loss += loss_G_L1.item() * batch_size

    return l1_loss / num_samples


def get_config(train_path, test_path, exp_name, save_path='./checkpoint/', pretrained=None, dataset='RadOnc', src='mr',
               id=2, lr_G=1e-4, lr_D=1e-4, lambda_l1=20, lambda_mind=20, start_epoch=0, num_epoch=200,
               batch_size=1, disc_channels=1, D_norm=True):
    """
    lambda_l1: L1 loss weight
    disc_channels: if 1, use GAN discriminator; if 2, use conditional GAN discriminator
    D_norm: if True, apply normalization in the discriminator;
    """
    args = dict()
    args['train_path'] = train_path
    args['test_path'] = test_path
    args['exp_name'] = exp_name
    args['save_path'] = save_path
    args['pretrained'] = pretrained
    args['dataset'] = dataset
    args['src'] = src
    args['id'] = id

    args['lr_G'] = lr_G  # initial learning rate for generator
    args['lr_D'] = lr_D  # initial learning rate for discriminator
    args['lambda_l1'] = lambda_l1  # synthesis L1
    args['lambda_mind'] = lambda_mind # synthesis structural consistency loss
    args['start_epoch'] = start_epoch
    args['num_epoch'] = num_epoch
    args['batch_size'] = batch_size
    args['disc_channel'] = disc_channels
    args['D_norm'] = D_norm

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
    parser.add_argument("--lambda_mind", type=float,
                        dest="lambda_mind", default=20, help="lambda MIND loss")
    parser.add_argument("--epochs", type=int,
                        dest="num_epoch", default=200, help="number of epochs")
    parser.add_argument("--start_epoch", type=int,
                        dest="start_epoch", default=0, help="starting epoch")
    parser.add_argument("--batch", type=int,
                        dest="batch_size", default=1, help="batch_size")
    parser.add_argument("--disc_channels", type=int,
                        dest="disc_channels", default=1, help="batch_size")
    parser.add_argument("--dataset", type=str, default='RadOnc', dest="dataset", help='RadOnc or HeadScanner')
    parser.add_argument("--src", type=str, default='mr', dest="src", help='mr or cbct to ct synthesis')
    parser.add_argument("--id", type=int, dest="id", default=2, help="leave out id in headscanner")
    parser.add_argument("--D_norm", default=False, action='store_true', dest="D_norm",
                        help="apply instance normalization in discriminator")

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
    G = UNet(cdim=1, channels=(16, 32, 64, 128, 128), res=True, gumbel=False).to(device)
    # D = Discriminator(cdim=args['disc_channel'], num_channels=16, skip_connect=False).to(device)
    D = MultiResDiscriminator(cdim=args['disc_channel'], num_channels=16, norm='IN').to(device)

    # Load pretrained model
    if args['pretrained'] is not None:
        if os.path.exists(args['pretrained']):
            checkpoint = torch.load(args['pretrained'])
            G.load_state_dict(checkpoint['G_state_dict'], strict=False)
            D.load_state_dict(checkpoint['D_state_dict'])
            print('pretrained model loaded...')
            del checkpoint

    # Build Optimizer
    opt_G = optim.Adam(G.parameters(), lr=args['lr_G'])
    opt_D = optim.Adam(D.parameters(), lr=args['lr_D'])
    # opt_D = optim.Adam(D.parameters(), lr=args['lr_D'])

    # Prepare Tensorboard
    boardio = SummaryWriter(log_dir='./log/Pix2Pix_{}_{}'.format(args['exp_name'], args['lambda_l1']))

    # Start training
    for epoch in range(args['start_epoch'], args['num_epoch']):
        # =========== Training first few epochs with only synthesis ================
        total_loss, gan_loss, l1_loss, disc_loss = train_epoch(args, G, D, train_dataloader, opt_G, opt_D, device)
        print('epoch: {}, total_loss: {}, gan_loss: {}, l1_loss: {}, disc_loss: {}'.format(epoch, total_loss,
                                                                                           gan_loss, l1_loss,
                                                                                           disc_loss))
        boardio.add_scalar('total_loss', total_loss, epoch)
        boardio.add_scalar('gan_loss', gan_loss, epoch)
        boardio.add_scalar('l1_loss', l1_loss, epoch)
        boardio.add_scalar('disc_loss', disc_loss, epoch)

        valid_l1_loss = valid_epoch(args, G, valid_dataset, device)
        boardio.add_scalar('valid_l1_loss', valid_l1_loss, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'args': args
            }, os.path.join(args['save_path'], 'Pix2Pix_{}_ep{}.pt'.format(args['exp_name'], epoch)))
