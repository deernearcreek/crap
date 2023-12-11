import os
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from dataloader import RadOncTrainingDataset, RadOncValidationDataset
from networks_gan import JSRCascade, Discriminator, MultiResDiscriminator, JSRCascade_v3
import losses
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

def train_reg_epoch(args, G, train_dataloader, opt_G, device, scaler=None):
    checkpoint = torch.load('JSRCascade_synthesis_l1100_multi0.3_reg100_seg6_ep99.pt',map_location=device)
    for name, module in G.named_parameters():
        if name in checkpoint['G_state_dict'].keys():
            if name[0:11]!='decoder_reg':
                module.data.copy_(checkpoint['G_state_dict'][name])
                module.requires_grad = False
            # if name != 'decoder_reg.flow4.weight' and name != 'decoder_reg.flow4.bias':
            #     if name != 'decoder_syn_src.dec_final.weight' and name != 'decoder_syn_src.dec_final.bias':
            #         if name != 'decoder_syn_tgt.dec_final.weight' and name != 'decoder_syn_tgt.dec_final.bias':
            #             module.requires_grad = False
    G.train()
    sim_loss = 0
    smooth_loss = 0
    total_loss = 0
    num_samples = 0

    for _, batch in tqdm(enumerate(train_dataloader), 'reg phase:'):
        opt_G.zero_grad()
        if args['weak_supervision']:
            cbct_fixed, mr_moving, ct_fixed, ct_moving, seg_fixed, seg_moving = batch
            seg_fixed = seg_fixed.float().to(device)
            seg_moving = seg_moving.float().to(device)
        else:
            cbct_fixed, mr_moving, ct_fixed, ct_moving = batch
        batch_size = cbct_fixed.size(0)
        num_samples += batch_size
        cbct_fixed = cbct_fixed.float().unsqueeze(1).to(device)
        mr_moving = mr_moving.float().unsqueeze(1).to(device)
        ct_fixed = ct_fixed.float().unsqueeze(1).to(device)
        ct_moving = ct_moving.float().unsqueeze(1).to(device)
 

        if scaler is not None:
            with torch.cuda.amp.autocast():
                
                # z_src = G.encoder_src(mr_moving)
                # z_tgt = G.encoder_tgt(cbct_fixed)
                
                ct_fixed1 = torch.nn.AvgPool3d(2)(ct_fixed)
                ct_fixed2 = torch.nn.AvgPool3d(2)(ct_fixed1)
                ct_fixed3 = torch.nn.AvgPool3d(2)(ct_fixed2)
                
                ct_moving1 = torch.nn.AvgPool3d(2)(ct_moving)
                ct_moving2 = torch.nn.AvgPool3d(2)(ct_moving1)
                ct_moving3 = torch.nn.AvgPool3d(2)(ct_moving2)
                
                # input_dec = [torch.cat((z_src[0], z_tgt[0]), dim=1),
                #                   [ct_moving3, ct_fixed3],
                #                   [ct_moving2, ct_fixed2],
                #                   [ct_moving1, ct_fixed1],
                #                   [ct_moving, ct_fixed]]
                # flows = G.decoder_reg(input_dec)
                flows, ct_moving_synths, ct_fixed_synths = G(mr_moving, cbct_fixed)
                ct_fixed_synth = ct_fixed, ct_fixed1, ct_fixed2, ct_fixed3
                ct_moving_synth = ct_moving, ct_moving1, ct_moving2, ct_moving3
                
                synths_warped = [G.STN4(ct_moving_synths[0].detach(), flows[0]),
                                 G.STN3(ct_moving_synths[1].detach(), flows[1]),
                                 G.STN2(ct_moving_synths[2].detach(), flows[2]),
                                 G.STN1(ct_moving_synths[3].detach(), flows[3])]
                if args['lambda_multi'] > 0:
                    mrs_warped = [G.decoder_reg.STN4(mr_moving, flows[0]),
                                  G.decoder_reg.STN3(torch.nn.AvgPool3d(2)(mr_moving), flows[1]),
                                  G.decoder_reg.STN2(torch.nn.AvgPool3d(4)(mr_moving), flows[2]),
                                  G.decoder_reg.STN1(torch.nn.AvgPool3d(8)(mr_moving), flows[3])]
                if args['weak_supervision']:
                    segs_fixed = [seg_fixed, torch.nn.AvgPool3d(2)(seg_fixed),
                                  torch.nn.AvgPool3d(4)(seg_fixed),
                                  torch.nn.AvgPool3d(8)(seg_fixed)]
                    segs_warped = [G.decoder_reg.STN4(seg_moving, flows[0]),
                                   G.decoder_reg.STN3(torch.nn.AvgPool3d(2)(seg_moving), flows[1]),
                                   G.decoder_reg.STN2(torch.nn.AvgPool3d(4)(seg_moving), flows[2]),
                                   G.decoder_reg.STN1(torch.nn.AvgPool3d(8)(seg_moving), flows[3])]

                loss_sim = 0
                loss_smooth = 0
                loss_seg = 0
                weights_per_level = [1, 0.75, 0.5, 0.25]
                for level in range(len(flows)):
                    weight = weights_per_level[level]
                    if args['lambda_multi'] > 0:
                        loss_sim += losses.MIND((128 // 2 ** level, 160 // 2 ** level, 128 // 2 ** level),
                                                d=1, patch_size=7, use_gaussian_kernel=True).loss(mrs_warped[level],
                                                                                                  ct_fixed_synths[
                                                                                                      level].detach()) * \
                                    args['lambda_multi'] * weight
                        loss_sim += (1 + losses.NCC(device='cuda').loss(
                            synths_warped[level], ct_fixed_synths[level].detach())) \
                                    * (1 - args['lambda_multi']) * weight
                    else:
                        loss_sim += 1 + losses.NCC(device='cuda').loss(synths_warped[level],
                                                                       ct_fixed_synths[level].detach()) * weight
                    loss_smooth += vxm.losses.Grad('l2', loss_mult=1).loss(None, flows[level]) * args['lambda_flow'][
                        level]

                    if args['weak_supervision']:
                        loss_seg += losses.Dice().loss(segs_warped[level],
                                                       segs_fixed[level]) * args['lambda_seg'] * weight

                loss_reg = (loss_sim + loss_smooth + loss_seg) / len(flows)

            scaler.scale(loss_reg).backward()
            scaler.step(opt_G)
            scaler.update()

        else:
            ct_fixed1 = torch.nn.AvgPool3d(2)(ct_fixed)
            ct_fixed2 = torch.nn.AvgPool3d(2)(ct_fixed1)
            ct_fixed3 = torch.nn.AvgPool3d(2)(ct_fixed2)
            
            ct_moving1 = torch.nn.AvgPool3d(2)(ct_moving)
            ct_moving2 = torch.nn.AvgPool3d(2)(ct_moving1)
            ct_moving3 = torch.nn.AvgPool3d(2)(ct_moving2)
            
            flows, ct_moving_synths, ct_fixed_synths = G(mr_moving, cbct_fixed)
            
            ct_fixed_synth = ct_fixed, ct_fixed1, ct_fixed2, ct_fixed3
            ct_moving_synth = ct_moving, ct_moving1, ct_moving2, ct_moving3
            synths_warped = [G.decoder_reg.STN4(ct_moving_synths[0].detach(), flows[0]),
                             G.decoder_reg.STN3(ct_moving_synths[1].detach(), flows[1]),
                             G.decoder_reg.STN2(ct_moving_synths[2].detach(), flows[2]),
                             G.decoder_reg.STN1(ct_moving_synths[3].detach(), flows[3])]
            if args['lambda_multi'] > 0:
                mrs_warped = [G.decoder_reg.STN4(mr_moving, flows[0]),
                              G.decoder_reg.STN3(torch.nn.AvgPool3d(2)(mr_moving), flows[1]),
                              G.decoder_reg.STN2(torch.nn.AvgPool3d(4)(mr_moving), flows[2]),
                              G.decoder_reg.STN1(torch.nn.AvgPool3d(8)(mr_moving), flows[3])]
            if args['weak_supervision']:
                segs_fixed = [seg_fixed, torch.nn.AvgPool3d(2)(seg_fixed),
                              torch.nn.AvgPool3d(4)(seg_fixed),
                              torch.nn.AvgPool3d(8)(seg_fixed)]
                segs_warped = [G.decoder_reg.STN4(seg_moving, flows[0]),
                               G.decoder_reg.STN3(torch.nn.AvgPool3d(2)(seg_moving), flows[1]),
                               G.decoder_reg.STN2(torch.nn.AvgPool3d(4)(seg_moving), flows[2]),
                               G.decoder_reg.STN1(torch.nn.AvgPool3d(8)(seg_moving), flows[3])]

            loss_sim = 0
            loss_smooth = 0
            loss_seg = 0
            weights_per_level = [1, 0.75, 0.5, 0.25]
            for level in range(len(flows)):
                weight = weights_per_level[level]
                if args['lambda_multi'] > 0:
                    loss_sim += losses.MIND((128 // 2 ** level, 160 // 2 ** level, 128 // 2 ** level),
                                            d=1, patch_size=7, use_gaussian_kernel=True).loss(mrs_warped[level],
                                                                                              ct_fixed_synths[
                                                                                                  level].detach()) * \
                                args['lambda_multi'] * weight
                    loss_sim += (1 + losses.NCC(device='cuda').loss(
                        synths_warped[level], ct_fixed_synths[level].detach())) \
                                * (1 - args['lambda_multi']) * weight
                else:
                    loss_sim += 1 + losses.NCC(device='cuda').loss(synths_warped[level],
                                                                   ct_fixed_synths[level].detach()) * weight
                loss_smooth += vxm.losses.Grad('l2', loss_mult=1).loss(None, flows[level]) * args['lambda_flow'][
                    level]

                if args['weak_supervision']:
                    loss_seg += losses.Dice().loss(segs_warped[level],
                                                   segs_fixed[level]) * args['lambda_seg'] * weight

            loss_reg = (loss_sim + loss_smooth + loss_seg) / len(flows)

            loss_reg.backward()
            opt_G.step()

        sim_loss += loss_sim.item() * batch_size
        smooth_loss += loss_smooth.item() * batch_size
        total_loss += loss_reg.item() * batch_size

    return total_loss / num_samples, sim_loss / num_samples, smooth_loss / num_samples

def valid_epoch(G, valid_dataloader, device):
    loss_L1 = 0
    loss_ncc = 0
    loss_seg = 0
    num_samples = 0

    for i, batch in tqdm(enumerate(valid_dataloader), 'joint phase:'):
        cbct_fixed, mr_moving, ct_fixed, ct_moving, seg_fixed, seg_moving = batch
        cbct_fixed = cbct_fixed.float().to(device)
        mr_moving = mr_moving.float().to(device)
        ct_fixed = ct_fixed.unsqueeze(0).unsqueeze(0).float().to(device)
        ct_moving = ct_moving.unsqueeze(0).unsqueeze(0).float().to(device)
        seg_fixed = seg_fixed.float().to(device)
        seg_moving = seg_moving.float().to(device)

        batch_size = cbct_fixed.size(0)
        num_samples += batch_size

        with torch.no_grad():
            # z_src = G.encoder_src(mr_moving)
            # z_tgt = G.encoder_tgt(cbct_fixed)
            
            ct_fixed1 = torch.nn.AvgPool3d(2)(ct_fixed)
            ct_fixed2 = torch.nn.AvgPool3d(2)(ct_fixed1)
            ct_fixed3 = torch.nn.AvgPool3d(2)(ct_fixed2)
            
            ct_moving1 = torch.nn.AvgPool3d(2)(ct_moving)
            ct_moving2 = torch.nn.AvgPool3d(2)(ct_moving1)
            ct_moving3 = torch.nn.AvgPool3d(2)(ct_moving2)
            
            # input_dec = [torch.cat((z_src[0], z_tgt[0]), dim=1),
            #                 [ct_moving3, ct_fixed3],
            #                 [ct_moving2, ct_fixed2],
            #                 [ct_moving1, ct_fixed1],
            #                 [ct_moving, ct_fixed]]
            # flows = G.decoder_reg(input_dec)
            flows, ct_moving_synths, ct_fixed_synths = G(mr_moving, cbct_fixed)
            ct_fixed_synth = ct_fixed, ct_fixed1, ct_fixed2, ct_fixed3
            ct_moving_synth = ct_moving, ct_moving1, ct_moving2, ct_moving3

            synth_warped = G.decoder_reg.STN4(ct_moving_synths[0].detach(), flows[0])
            seg_warped = G.decoder_reg.STN4(seg_moving, flows[0])

            loss_G_L1 = torch.mean(torch.abs(ct_fixed - ct_fixed_synths[0])) + torch.mean(
                torch.abs(ct_moving - ct_moving_synths[0]))

            # Registration losses: multi-resolution image similarity and smoothness regularization
            loss_L1 += losses.MaskedL1().loss(ct_moving_synths[0], ct_moving).item() + \
                       losses.MaskedL1().loss(ct_fixed_synths[0], ct_fixed).item()

            loss_ncc += 1 + losses.NCC(device='cuda').loss(synth_warped, ct_fixed_synths[0]).item()
            loss_seg += losses.Dice().loss(seg_warped, seg_fixed).item()

    return loss_L1 / num_samples, loss_ncc / num_samples, loss_seg / num_samples


def save_model(G, D_MR, D_CBCT, opt_G, opt_D_MR, opt_D_CBCT, args, epoch):
    torch.save({
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_MR_state_dict': D_MR.state_dict(),
        'D_CBCT_state_dict': D_CBCT.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_MR_state_dict': opt_D_MR.state_dict(),
        'opt_D_CBCT_state_dict': opt_D_CBCT.state_dict(),
        'args': args
    }, os.path.join(args['save_path'], 'JSRDecoder_gtCT_{}_ep{}.pt'.format(args['exp_name'], epoch)))
    #}, os.path.join(args['save_path'], 'JSRReg_{}_ep{}.pt'.format(args['exp_name'], epoch)))


def get_config(train_path, test_path, exp_name, save_path='./checkpoint/', pretrained=None, amp=False,
               lr_G=1e-4, lr_D=1e-4, lambda_l1=20, lambda_mind=10, lambda_seg=1, lambda_flow=2, lambda_reg=1,
               lambda_multi=0.3, start_epoch=0, batch_size=1, disc_channels=2, levels=0,
               separate_decoders=False, weak_supervision=False, schedule=[10, 0, 10],
               synthesis_checkpoint_mod=10, registration_checkpoint_mod=10, jsr_checkpoint_mod=10):
    args = dict()
    args['train_path'] = train_path
    args['test_path'] = test_path
    args['exp_name'] = exp_name
    args['save_path'] = os.getcwd()
    args['pretrained'] = pretrained

    args['lr_G'] = lr_G  # initial learning rate for generator
    args['lr_D'] = lr_D  # initial learning rate for discriminator
    args['lambda_l1'] = lambda_l1  # synthesis L1
    args['lambda_mind'] = lambda_mind  # synthesis structural consistency
    args['lambda_flow'] = lambda_flow  # registration flow smoothness penalty
    args['lambda_multi'] = lambda_multi  # multimodal similarity metric
    args['lambda_reg'] = lambda_reg  # registration vs synthesis
    args['lambda_seg'] = lambda_seg
    args['start_epoch'] = start_epoch
    args['schedule'] = schedule
    args['batch_size'] = batch_size
    args['disc_channel'] = disc_channels
    args['levels'] = levels
    args['separate_decoders'] = separate_decoders
    args['weak_supervision'] = weak_supervision
    args['amp'] = amp
    args['synthesis_checkpoint_mod'] = synthesis_checkpoint_mod
    args['registration_checkpoint_mod'] = registration_checkpoint_mod
    args['jsr_checkpoint_mod'] = jsr_checkpoint_mod

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
                        dest="lambda_mind", default=20, help="lambda synthesis structural consistency loss")
    parser.add_argument("--lambda_flow", type=int, nargs='+',
                        dest="lambda_flow", default=2, help="smoothness regularization")
    parser.add_argument("--lambda_reg", type=float,
                        dest="lambda_reg", default=1, help="registration vs synthesis strength")
    parser.add_argument("--lambda_seg", type=float,
                        dest="lambda_seg", default=1, help="Dice loss")
    parser.add_argument("--lambda_multi", type=float,
                        dest="lambda_multi", default=0.3, help="multimodal similarity loss")
    parser.add_argument("--batch", type=int,
                        dest="batch_size", default=1, help="batch_size")
    parser.add_argument("--disc_channels", type=int,
                        dest="disc_channels", default=2, help="batch_size")
    parser.add_argument("--separate_dec", default=False, action='store_true', dest="separate_decoders",
                        help="whether to use separate synthesis decoders")
    parser.add_argument("--weak_supervision", default=False, action='store_true', dest="weak_supervision",
                        help="weak supervision using segmentation labels")
    parser.add_argument("--levels", type=int,
                        dest="levels", default=0, help="number of resolution levels")
    parser.add_argument("--schedule", type=int, nargs='+',
                        dest="schedule", default=[10, 0, 10], help="training schedule: synthesis+regitration+joint")
    parser.add_argument("--start_epoch", type=int,
                        dest="start_epoch", default=0, help="starting epoch")
    parser.add_argument("--amp", default=False, action='store_true', dest="amp",
                        help="whether to use automatic mixed precision training")
    parser.add_argument("--synthesis_checkpoint_mod", type=int,
                        dest="synthesis_checkpoint_mod", default=10, help="Save synthesis checkpoint every...")
    parser.add_argument("--registration_checkpoint_mod", type=int,
                        dest="registration_checkpoint_mod", default=10, help="Save registration checkpoint every...")
    parser.add_argument("--jsr_checkpoint_mod", type=int,
                        dest="jsr_checkpoint_mod", default=10, help="Save JSR checkpoint every...")

    args = parser.parse_args()
    args = get_config(**vars(args))
    print(args)
    device = torch.device('cuda')

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = True

    # Build dataset
    train_dataset = RadOncTrainingDataset(args['train_path'], num_samples=None, transform=False,
                                          supervision=True, return_segmentation=args['weak_supervision'])
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                  num_workers=1, pin_memory=True)
    valid_dataset = RadOncValidationDataset(args['test_path'], num_samples=None, supervision=True,
                                            return_segmentation=True)

    # Build model
    G = JSRCascade(cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                   separate_decoders=args['separate_decoders'], res=True, version='v3').to(device)
    # D = Discriminator(cdim=args['disc_channel'], num_layers=3, num_channels=32,
    #                   kernel_size=3, apply_norm=args['D_norm'], norm='BN', skip_connect=False).to(device)
    D_MR = MultiResDiscriminator(cdim=args['disc_channel'], num_channels=16, norm='IN').to(device)
    D_CBCT = MultiResDiscriminator(cdim=args['disc_channel'], num_channels=16, norm='IN').to(device)

    # Load pretrained model
    if args['pretrained'] is not None:
        if os.path.exists(args['pretrained']):
            checkpoint = torch.load(args['pretrained'])
            G.load_state_dict(checkpoint['G_state_dict'])
            D_MR.load_state_dict(checkpoint['D_MR_state_dict'])
            D_CBCT.load_state_dict(checkpoint['D_CBCT_state_dict'])
            print('pretrained model loaded...')
            del checkpoint

    # Build Optimizer
    opt_G = optim.Adam(G.parameters(), lr=args['lr_G'])
    opt_D_MR = optim.Adam(D_MR.parameters(), lr=args['lr_D'])
    opt_D_CBCT = optim.Adam(D_CBCT.parameters(), lr=args['lr_D'])
    # opt_G = optim.SGD(G.parameters(), lr=args['lr_G'])
    # opt_D_MR = optim.SGD(D_MR.parameters(), lr=args['lr_D'])
    # opt_D_CBCT = optim.SGD(D_CBCT.parameters(), lr=args['lr_D'])
    scheduler = optim.lr_scheduler.StepLR(opt_G, step_size=40, gamma=0.1)
    # scheduler_warmup = GradualWarmupScheduler(opt_G, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    # Optimizer Scheduler
    scheduler = optim.lr_scheduler.StepLR(opt_G, step_size=50, gamma=0.2)
    # scheduler_warmup = GradualWarmupScheduler(opt_G, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    # Prepare Tensorboard
    boardio = SummaryWriter(log_dir='./log/JSRCascade_{}_{}_{}_{}'.format(args['exp_name'], args['lambda_l1'],
                                                                          args['lambda_flow'], args['lambda_reg']))

    if args['amp']:
        scaler = torch.cuda.amp.GradScaler()  # Cuda Mix Precision--save gpu memory
    else:
        scaler = None

    # Start training
    epoch = args['start_epoch']
    # =========== Synthesis Training Phase ================
    for i in range(args['schedule'][0]):

        total_loss, gan_loss, l1_loss, disc_loss = train_synth_epoch(args, G, D_MR, D_CBCT, train_dataloader,
                                                                     opt_G, opt_D_MR, opt_D_CBCT, device, scaler)
        print('synthesis epoch: {}, total_loss: {}, gan_loss: {}, l1_loss: {}, disc_loss: {}'.format(epoch, total_loss,
                                                                                                     gan_loss, l1_loss,
                                                                                                     disc_loss))
        valid_l1_loss, valid_ncc_loss, valid_seg_loss = valid_epoch(G, valid_dataset, device)
        boardio.add_scalar('total_loss', total_loss, epoch)
        boardio.add_scalar('gan_loss', gan_loss, epoch)
        boardio.add_scalar('l1_loss', l1_loss, epoch)
        boardio.add_scalar('disc_loss', disc_loss, epoch)
        boardio.add_scalar('valid_l1_loss', valid_l1_loss, epoch)
        boardio.add_scalar('valid_ncc_loss', valid_ncc_loss, epoch)
        boardio.add_scalar('valid_seg_loss', valid_seg_loss, epoch)

        if (epoch + 1) % args['synthesis_checkpoint_mod'] == 0:
            save_model(G, D_MR, D_CBCT, opt_G, opt_D_MR, opt_D_CBCT, args, epoch)
        epoch += 1

    # =========== Registration Training Phase ================
    for i in range(args['schedule'][1]):
        reg_loss, sim_loss, smoothness_loss = train_reg_epoch(args, G, train_dataloader, opt_G, device, scaler)
        print('registration epoch: {}, reg_loss: {}, sim_loss: {}, smoothness_loss: {}'.format(epoch, reg_loss,
                                                                                               sim_loss,
                                                                                               smoothness_loss))
        valid_l1_loss, valid_ncc_loss, valid_seg_loss = valid_epoch(G, valid_dataset, device)
        boardio.add_scalar('reg_loss', reg_loss, epoch)
        boardio.add_scalar('sim_loss', sim_loss, epoch)
        boardio.add_scalar('smoothness_loss', smoothness_loss, epoch)
        boardio.add_scalar('valid_l1_loss', valid_l1_loss, epoch)
        boardio.add_scalar('valid_ncc_loss', valid_ncc_loss, epoch)
        boardio.add_scalar('valid_seg_loss', valid_seg_loss, epoch)

        if (epoch + 1) % args['registration_checkpoint_mod'] == 0:
            save_model(G, D_MR, D_CBCT, opt_G, opt_D_MR, opt_D_CBCT, args, epoch)
        scheduler.step()
        epoch += 1

    # =========== Joint Training Phase ================
    for i in range(args['schedule'][2]):

        total_loss, synth_loss, reg_loss, sim_loss, \
        smoothness_loss, disc_loss = train_joint_epoch(args, G, D_MR, D_CBCT, train_dataloader,
                                                       opt_G, opt_D_MR, opt_D_CBCT, device, scaler)
        valid_l1_loss, valid_ncc_loss, valid_seg_loss = valid_epoch(G, valid_dataset, device)
        scheduler.step()
        print('joint epoch: {}, total_loss: {}, synthesis_loss: {}, reg_loss: {}, sim_loss: {}, smoothness_loss: {}, '
              'disc_loss: {}'.format(epoch, total_loss, synth_loss, reg_loss, sim_loss, smoothness_loss, disc_loss))
        boardio.add_scalar('total_loss', total_loss, epoch)
        boardio.add_scalar('synth_loss', synth_loss, epoch)
        boardio.add_scalar('reg_loss', reg_loss, epoch)
        boardio.add_scalar('sim_loss', sim_loss, epoch)
        boardio.add_scalar('smoothness_loss', smoothness_loss, epoch)
        boardio.add_scalar('disc_loss', disc_loss, epoch)
        boardio.add_scalar('valid_l1_loss', valid_l1_loss, epoch)
        boardio.add_scalar('valid_ncc_loss', valid_ncc_loss, epoch)
        boardio.add_scalar('valid_seg_loss', valid_seg_loss, epoch)

        if (epoch + 1) % args['jsr_checkpoint_mod'] == 0:
            save_model(G, D_MR, D_CBCT, opt_G, opt_D_MR, opt_D_CBCT, args, epoch)
        scheduler.step()
        epoch += 1
