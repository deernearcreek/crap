import os
from tqdm import tqdm
from argparse import ArgumentParser
import math
from skimage.transform import resize
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import RadOncTrainingDataset, RadOncValidationDataset
from networks_gan import JSRCascade, Discriminator, MultiResDiscriminator
import losses
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True

def train_reg_epoch(args, G, train_dataloader, opt_G, device, scaler=None,scheduler_lambda = False):
    checkpoint = torch.load('JSRCascade_rnewflow_l1100_multi0.3_reg100_seg6_ep99.pt',map_location=device)
    for name, module in G.named_parameters():
        if name in checkpoint['G_state_dict'].keys():
            # if name[0:11]!='decoder_reg':
            # module.data.copy_(checkpoint['G_state_dict'][name])
            # if name != 'decoder_reg.flow4.weight' and name != 'decoder_reg.flow4.bias':
            # #     if name != 'decoder_syn_src.dec_final.weight' and name != 'decoder_syn_src.dec_final.bias':
            # #         if name != 'decoder_syn_tgt.dec_final.weight' and name != 'decoder_syn_tgt.dec_final.bias':
            module.data.copy_(checkpoint['G_state_dict'][name])
                # module.requires_grad = False
    # G.set_required_grad(mode='reg')
    G.train()
    sim_loss = 0
    smooth_loss = 0
    seg_loss = 0
    total_loss = 0
    unc_loss = 0
    log1_xloss = 0
    log2_xloss = 0
    log3_xloss = 0
    log1_yloss = 0
    log2_yloss = 0
    log3_yloss = 0
    log1_zloss = 0
    log2_zloss = 0
    log3_zloss = 0
    reg_xloss = 0
    reg_yloss = 0
    reg_zloss = 0
    
    num_samples = 0
    lunc = args['lambda_inunc']
    for _, batch in tqdm(enumerate(train_dataloader), 'reg phase:'):
        opt_G.zero_grad()
        if scheduler_lambda:
            lambd = 0.3*math.exp(0.115*(num_samples+101)/400-13.8)
        else:
            lambd = args['lambda_unc']
        print(lambd)
        if args['weak_supervision']:
            cbct_fixed, mr_moving, _, _, seg_fixed, seg_moving,flow = batch
            seg_fixed = seg_fixed.float().to(device)
            seg_moving = seg_moving.float().to(device)
        else:
            cbct_fixed, mr_moving, _, _, flow= batch
        batch_size = cbct_fixed.size(0)
        num_samples += batch_size
        cbct_fixed = cbct_fixed.float().unsqueeze(1).to(device)
        mr_moving = mr_moving.float().unsqueeze(1).to(device)
        gtx = flow[0,0,0,:,:,:].to(device)
        gty = flow[0,0,1,:,:,:].to(device)
        gtz = flow[0,0,2:,:,:].to(device)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                flows, ct_moving_synths, ct_fixed_synths,_,_ = G(mr_moving, cbct_fixed)
                # uncert = flows[4]
                uncert = torch.clamp(flows[4],min = 1e-6)
                alpha_x = uncert[0,0,:,:,:]
                beta_x = uncert[0,1,:,:,:]
                mu_x = uncert[0,2,:,:,:]
                omega_x = 2*beta_x*(1+mu_x)
                alpha_y = uncert[0,3,:,:,:]
                beta_y = uncert[0,4,:,:,:]
                mu_y = uncert[0,5,:,:,:]
                omega_y = 2*beta_y*(1+mu_y)
                alpha_z = uncert[0,6,:,:,:]
                beta_z = uncert[0,7,:,:,:]
                mu_z = uncert[0,8,:,:,:]
                omega_z = 2*beta_z*(1+mu_z)
                
                log1x = torch.log(math.pi/mu_x)
                log1y = torch.log(math.pi/mu_y)
                log1z = torch.log(math.pi/mu_z)
                log2x = torch.log(omega_x)
                log2y = torch.log(omega_y)
                log2z = torch.log(omega_z)
                log3x = torch.log((flows[0][0,0,:,:,:]+gtx)**2*mu_x+omega_x)
                log3y = torch.log((flows[0][0,1,:,:,:]+gty)**2*mu_y+omega_y)
                log3z = torch.log((flows[0][0,2,:,:,:]+gtz)**2*mu_z+omega_z)
                log4x = torch.lgamma(alpha_x)-torch.lgamma(alpha_x+0.5)
                log4y = torch.lgamma(alpha_y)-torch.lgamma(alpha_y+0.5)
                log4z = torch.lgamma(alpha_z)-torch.lgamma(alpha_z+0.5)
                
                unc_x = 0.5*log1x-alpha_x*log2x+(alpha_x+0.5)*log3x+log4x
                unc_y = 0.5*log1y-alpha_y*log2y+(alpha_y+0.5)*log3y+log4y
                unc_z = 0.5*log1z-alpha_z*log2z+(alpha_z+0.5)*log3z+log4z
                reg_x = (2*mu_x+alpha_x)*torch.abs((flows[0][0,0,:,:,:]+gtx))
                reg_y = (2*mu_y+alpha_y)*torch.abs((flows[0][0,1,:,:,:]+gty))
                reg_z = (2*mu_z+alpha_z)*torch.abs((flows[0][0,2,:,:,:]+gtz))
                
                loss_uncx = unc_x+lunc*reg_x
                loss_uncy = unc_y+lunc*reg_y
                loss_uncz = unc_z+lunc*reg_z
                            
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
                for level in range(4):
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

                loss_reg = (loss_sim + loss_smooth + loss_seg) / 4
                loss_unc = (loss_uncx + loss_uncy + loss_uncz).mean()
            # scaler.scale(loss_reg.mean()).backward()
            # scaler.scale(loss_unc).backward()
            scaler.scale(loss_reg+lambd*loss_unc).backward()
            scaler.step(opt_G)
            scaler.update()
            #print(loss_reg, loss_unc.mean())

        else:
            flows, ct_moving_synths, ct_fixed_synths = G(mr_moving, cbct_fixed)
            # uncert = flows[4]
            uncert = torch.clamp(flows[4],min = 1e-6)
            alpha_x = uncert[0,0,:,:,:]+1
            beta_x = uncert[0,1,:,:,:]
            mu_x = uncert[0,2,:,:,:]
            omega_x = 2*beta_x*(1+mu_x)
            alpha_y = uncert[0,3,:,:,:]+1
            beta_y = uncert[0,4,:,:,:]
            mu_y = uncert[0,5,:,:,:]
            omega_y = 2*beta_y*(1+mu_y)
            alpha_z = uncert[0,6,:,:,:]+1
            beta_z = uncert[0,7,:,:,:]
            mu_z = uncert[0,8,:,:,:]
            omega_z = 2*beta_z*(1+mu_z)
            
            log1x = torch.log(math.pi/mu_x)
            log1y = torch.log(math.pi/mu_y)
            log1z = torch.log(math.pi/mu_z)
            log2x = torch.log(omega_x)
            log2y = torch.log(omega_y)
            log2z = torch.log(omega_z)
            log3x = torch.log((flows[0][0,0,:,:,:]+gtx)**2*mu_x+omega_x)
            log3y = torch.log((flows[0][0,1,:,:,:]+gty)**2*mu_y+omega_y)
            log3z = torch.log((flows[0][0,2,:,:,:]+gtz)**2*mu_z+omega_z)
            log4x = torch.lgamma(alpha_x)-torch.lgamma(alpha_x+0.5)
            log4y = torch.lgamma(alpha_y)-torch.lgamma(alpha_y+0.5)
            log4z = torch.lgamma(alpha_z)-torch.lgamma(alpha_z+0.5)
            
            unc_x = 0.5*log1x-alpha_x*log2x+(alpha_x+0.5)*log3x+log4x
            unc_y = 0.5*log1y-alpha_y*log2y+(alpha_y+0.5)*log3y+log4y
            unc_z = 0.5*log1z-alpha_z*log2z+(alpha_z+0.5)*log3z+log4z
            
            reg_x = (2*mu_x+alpha_x)*torch.abs((flows[0][0,0,:,:,:]+gtx))
            reg_y = (2*mu_y+alpha_y)*torch.abs((flows[0][0,1,:,:,:]+gty))
            reg_z = (2*mu_z+alpha_z)*torch.abs((flows[0][0,2,:,:,:]+gtz))
            
            loss_uncx = unc_x+lunc*reg_x
            loss_uncy = unc_y+lunc*reg_y
            loss_uncz = unc_z+lunc*reg_z
                                                   
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
            for level in range(4):
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

            loss_reg = (loss_sim + loss_smooth + loss_seg) / 4
            loss_unc = (loss_uncx+loss_uncy+loss_uncz).mean()

            (loss_reg+lambd*loss_unc).backward()
            # loss_unc.backward()
            #loss_reg.backward()
            
            opt_G.step()
            
        sim_loss += loss_sim.item() * batch_size
        smooth_loss += loss_smooth.item() * batch_size
        unc_loss += loss_unc.item()*batch_size
        total_loss += loss_reg.item() * batch_size
        seg_loss += loss_seg.item() * batch_size

        log1_xloss += log1x.mean().item() * batch_size
        log2_xloss += log2x.mean().item() * batch_size
        log3_xloss += log3x.mean().item() * batch_size
        log1_yloss += log1y.mean().item() * batch_size
        log2_yloss += log2y.mean().item() * batch_size
        log3_yloss += log3y.mean().item() * batch_size
        log1_zloss += log1z.mean().item() * batch_size
        log2_zloss += log2z.mean().item() * batch_size
        log3_zloss += log3z.mean().item() * batch_size
        reg_xloss += reg_x.mean().item() * batch_size
        reg_yloss += reg_y.mean().item() * batch_size
        reg_zloss += reg_z.mean().item() * batch_size
        
        if total_loss != total_loss:
            print(total_loss / num_samples, sim_loss / num_samples, smooth_loss / num_samples, unc_loss/num_samples, seg_loss/num_samples,\
                  log1_xloss/num_samples,log2_xloss/num_samples,log3_xloss/num_samples,log1_yloss/num_samples,log2_yloss/num_samples,\
                  log3_yloss/num_samples,log1_zloss/num_samples,log2_zloss/num_samples,log3_zloss/num_samples,reg_xloss,reg_yloss,reg_zloss)
            break


    return total_loss / num_samples, sim_loss / num_samples, smooth_loss / num_samples, unc_loss/num_samples, seg_loss/num_samples,\
                  log1_xloss/num_samples,log2_xloss/num_samples,log3_xloss/num_samples,log1_yloss/num_samples,log2_yloss/num_samples,\
                  log3_yloss/num_samples,log1_zloss/num_samples,log2_zloss/num_samples,log3_zloss/num_samples,reg_xloss/num_samples,reg_yloss/num_samples,reg_zloss/num_samples



def train_joint_epoch(args, G, D_MR, D_CBCT, train_dataloader, opt_G, opt_D_MR, opt_D_CBCT, device, scaler=None,scheduler_lambda = False):
    # checkpoint = torch.load('JSRCascade_test_l1100_multi0.3_reg100_seg6_ep99.pt',map_location=device)
    # for name, module in G.named_parameters():
    #     if name in checkpoint['G_state_dict'].keys():
    #         module.data.copy_(checkpoint['G_state_dict'][name])
            # if name != 'decoder_reg.flow4.weight' and name != 'decoder_reg.flow4.bias':
            #     if name != 'decoder_syn_src.dec_final.weight' and name != 'decoder_syn_src.dec_final.bias':
            #         if name != 'decoder_syn_tgt.dec_final.weight' and name != 'decoder_syn_tgt.dec_final.bias':
            #             module.requires_grad = False
    G.train()
    D_MR.train()
    D_CBCT.train()
    
    

    disc_loss = 0
    synth_loss = 0
    reg_loss = 0
    sim_loss = 0
    smooth_loss = 0
    total_loss = 0
    
    seg_loss = 0
    unc_loss = 0
    log1_xloss = 0
    log2_xloss = 0
    log3_xloss = 0
    log1_yloss = 0
    log2_yloss = 0
    log3_yloss = 0
    log1_zloss = 0
    log2_zloss = 0
    log3_zloss = 0
    reg_xloss = 0
    reg_yloss = 0
    reg_zloss = 0
    num_samples = 0
    if scheduler_lambda:
        lambd = np.exp(0.115*(num_samples+1)-13.8)
    else:
        lambd = 0.1
    lunc = 1

    for _, batch in tqdm(enumerate(train_dataloader), 'joint phase:'):
        if args['weak_supervision']:
            cbct_fixed, mr_moving, ct_fixed, ct_moving, seg_fixed, seg_moving, flow = batch
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
        gtx = flow[0,0,0,:,:,:].to(device)
        gty = flow[0,0,1,:,:,:].to(device)
        gtz = flow[0,0,2:,:,:].to(device)

        # =========== Update G ================
        G.set_required_grad(mode='joint')
        opt_G.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                flows, ct_moving_synths, ct_fixed_synths = G(mr_moving, cbct_fixed)
                uncert = flows[4]
                alpha_x = uncert[0,0,:,:,:]
                beta_x = uncert[0,1,:,:,:]
                mu_x = uncert[0,2,:,:,:]
                omega_x = 2*beta_x*(1+mu_x)
                alpha_y = uncert[0,3,:,:,:]
                beta_y = uncert[0,4,:,:,:]
                mu_y = uncert[0,5,:,:,:]
                omega_y = 2*beta_y*(1+mu_y)
                alpha_z = uncert[0,6,:,:,:]
                beta_z = uncert[0,7,:,:,:]
                mu_z = uncert[0,8,:,:,:]
                omega_z = 2*beta_z*(1+mu_z)
                
                log1x = torch.log(math.pi/mu_x)
                log1y = torch.log(math.pi/mu_y)
                log1z = torch.log(math.pi/mu_z)
                log2x = torch.log(omega_x)
                log2y = torch.log(omega_y)
                log2z = torch.log(omega_z)
                log3x = torch.log((flows[0][0,0,:,:,:]+gtx)**2*mu_x+omega_x)
                log3y = torch.log((flows[0][0,1,:,:,:]+gty)**2*mu_y+omega_y)
                log3z = torch.log((flows[0][0,2,:,:,:]+gtz)**2*mu_z+omega_z)
                log4x = torch.lgamma(alpha_x)-torch.lgamma(alpha_x+0.5)
                log4y = torch.lgamma(alpha_y)-torch.lgamma(alpha_y+0.5)
                log4z = torch.lgamma(alpha_z)-torch.lgamma(alpha_z+0.5)
                
                unc_x = 0.5*log1x-alpha_x*log2x+(alpha_x+0.5)*log3x+log4x
                unc_y = 0.5*log1y-alpha_y*log2y+(alpha_y+0.5)*log3y+log4y
                unc_z = 0.5*log1z-alpha_z*log2z+(alpha_z+0.5)*log3z+log4z
                reg_x = (2*mu_x+alpha_x)*torch.abs((flows[0][0,0,:,:,:]+gtx))
                reg_y = (2*mu_y+alpha_y)*torch.abs((flows[0][0,1,:,:,:]+gty))
                reg_z = (2*mu_z+alpha_z)*torch.abs((flows[0][0,2,:,:,:]+gtz))
                
                loss_uncx = unc_x+lunc*reg_x
                loss_uncy = unc_y+lunc*reg_y
                loss_uncz = unc_z+lunc*reg_z
                
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

                D_fake_fixed = D_CBCT(torch.cat((ct_fixed_synths[0], cbct_fixed), dim=1))
                D_fake_moving = D_MR(torch.cat((ct_moving_synths[0], mr_moving), dim=1))

                # Synthesis Losses
                loss_G_GAN = 0
                for d_fake_fixed, d_fake_moving in zip(D_fake_fixed, D_fake_moving):
                    loss_G_GAN += torch.mean((torch.ones_like(d_fake_fixed) - d_fake_fixed) ** 2) \
                                  + torch.mean((torch.ones_like(d_fake_moving) - d_fake_moving) ** 2)
                loss_G_GAN = loss_G_GAN / len(D_fake_fixed)

                loss_G_L1 = torch.mean(torch.abs(ct_fixed - ct_fixed_synths[0])) \
                            + torch.mean(torch.abs(ct_moving - ct_moving_synths[0]))
                loss_G_MIND = losses.MIND((128, 160, 128), 1, 7, device=device).loss(cbct_fixed, ct_fixed_synths[0]) + \
                              losses.MIND((128, 160, 128), 1, 7, device=device).loss(mr_moving, ct_moving_synths[0])

                loss_synth = loss_G_GAN + args['lambda_l1'] * loss_G_L1 + args['lambda_mind'] * loss_G_MIND

                # Registration losses
                loss_sim = 0
                loss_smooth = 0
                loss_seg = 0
                weights_per_level = [1, 0.75, 0.5, 0.25]
                for level in range(4):
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

                loss_reg = (loss_sim + loss_smooth + loss_seg) / 4
                loss_unc = (loss_uncx+loss_uncy+loss_uncz).mean()

                # Total Loss
                loss_G = loss_synth + args['lambda_reg'] * (loss_reg+lambd*loss_unc.mean())

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)

        else:
            flows, ct_moving_synths, ct_fixed_synths = G(mr_moving, cbct_fixed)
            uncert = flows[4]
            alpha_x = uncert[0,0,:,:,:]
            beta_x = uncert[0,1,:,:,:]
            mu_x = uncert[0,2,:,:,:]
            omega_x = 2*beta_x*(1+mu_x)
            alpha_y = uncert[0,3,:,:,:]
            beta_y = uncert[0,4,:,:,:]
            mu_y = uncert[0,5,:,:,:]
            omega_y = 2*beta_y*(1+mu_y)
            alpha_z = uncert[0,6,:,:,:]
            beta_z = uncert[0,7,:,:,:]
            mu_z = uncert[0,8,:,:,:]
            omega_z = 2*beta_z*(1+mu_z)
            
            log1x = torch.log(math.pi/mu_x)
            log1y = torch.log(math.pi/mu_y)
            log1z = torch.log(math.pi/mu_z)
            log2x = torch.log(omega_x)
            log2y = torch.log(omega_y)
            log2z = torch.log(omega_z)
            log3x = torch.log((flows[0][0,0,:,:,:]+gtx)**2*mu_x+omega_x)
            log3y = torch.log((flows[0][0,1,:,:,:]+gty)**2*mu_y+omega_y)
            log3z = torch.log((flows[0][0,2,:,:,:]+gtz)**2*mu_z+omega_z)
            log4x = torch.lgamma(alpha_x)-torch.lgamma(alpha_x+0.5)
            log4y = torch.lgamma(alpha_y)-torch.lgamma(alpha_y+0.5)
            log4z = torch.lgamma(alpha_z)-torch.lgamma(alpha_z+0.5)
            
            unc_x = 0.5*log1x-alpha_x*log2x+(alpha_x+0.5)*log3x+log4x
            unc_y = 0.5*log1y-alpha_y*log2y+(alpha_y+0.5)*log3y+log4y
            unc_z = 0.5*log1z-alpha_z*log2z+(alpha_z+0.5)*log3z+log4z
            reg_x = (2*mu_x+alpha_x)*torch.abs((flows[0][0,0,:,:,:]+gtx))
            reg_y = (2*mu_y+alpha_y)*torch.abs((flows[0][0,1,:,:,:]+gty))
            reg_z = (2*mu_z+alpha_z)*torch.abs((flows[0][0,2,:,:,:]+gtz))
            
            loss_uncx = unc_x+lunc*reg_x
            loss_uncy = unc_y+lunc*reg_y
            loss_uncz = unc_z+lunc*reg_z
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

            D_fake_fixed = D_CBCT(torch.cat((ct_fixed_synths[0], cbct_fixed), dim=1))
            D_fake_moving = D_MR(torch.cat((ct_moving_synths[0], mr_moving), dim=1))

            # Synthesis Losses
            loss_G_GAN = 0
            for d_fake_fixed, d_fake_moving in zip(D_fake_fixed, D_fake_moving):
                loss_G_GAN += torch.mean((torch.ones_like(d_fake_fixed) - d_fake_fixed) ** 2) \
                              + torch.mean((torch.ones_like(d_fake_moving) - d_fake_moving) ** 2)
            loss_G_GAN = loss_G_GAN / len(D_fake_fixed)

            loss_G_L1 = torch.mean(torch.abs(ct_fixed - ct_fixed_synths[0])) \
                        + torch.mean(torch.abs(ct_moving - ct_moving_synths[0]))
            loss_G_MIND = losses.MIND((128, 160, 128), 1, 7, device=device).loss(cbct_fixed, ct_fixed_synths[0]) + \
                          losses.MIND((128, 160, 128), 1, 7, device=device).loss(mr_moving, ct_moving_synths[0])

            loss_synth = loss_G_GAN + args['lambda_l1'] * loss_G_L1 + args['lambda_mind'] * loss_G_MIND

            # Registration losses
            loss_sim = 0
            loss_smooth = 0
            loss_seg = 0
            weights_per_level = [1, 0.75, 0.5, 0.25]
            for level in range(4):
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

            loss_reg = (loss_sim + loss_smooth + loss_seg) / 4
            loss_unc = (loss_uncx+loss_uncy+loss_uncz).mean()

            # Total Loss
            loss_G = loss_synth + args['lambda_reg'] * (loss_reg+lambd*loss_unc)
            print(loss_unc.mean())
            loss_G.backward()
            opt_G.step()

        # =========== Update D ================
        opt_D_MR.zero_grad()
        opt_D_CBCT.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                D_real_fixed = D_CBCT(torch.cat((ct_fixed, cbct_fixed), dim=1))
                D_real_moving = D_MR(torch.cat((ct_moving, mr_moving), dim=1))
                D_fake_fixed = D_CBCT(torch.cat((ct_fixed_synths[0].detach(), cbct_fixed), dim=1))
                D_fake_moving = D_MR(torch.cat((ct_moving_synths[0].detach(), mr_moving), dim=1))

                loss_D_real_MR = 0
                loss_D_real_CBCT = 0
                loss_D_fake_MR = 0
                loss_D_fake_CBCT = 0

                for d_real_fixed, d_real_moving, d_fake_fixed, d_fake_moving in zip(D_real_fixed, D_real_moving,
                                                                                    D_fake_fixed,
                                                                                    D_fake_moving):
                    loss_D_real_CBCT += torch.mean((torch.ones_like(d_real_fixed) - d_real_fixed) ** 2)
                    loss_D_real_MR += torch.mean((torch.ones_like(d_real_moving) - d_real_moving) ** 2)
                    loss_D_fake_CBCT += torch.mean((torch.zeros_like(d_fake_fixed) - d_fake_fixed) ** 2)
                    loss_D_fake_MR += torch.mean((torch.zeros_like(d_fake_moving) - d_fake_moving) ** 2)
                loss_D_MR = (loss_D_real_MR + loss_D_fake_MR) / 2 / len(D_real_fixed)
                loss_D_CBCT = (loss_D_real_CBCT + loss_D_fake_CBCT) / 2 / len(D_real_fixed)

            scaler.scale(loss_D_MR).backward()
            scaler.step(opt_D_MR)
            scaler.scale(loss_D_CBCT).backward()
            scaler.step(opt_D_CBCT)
            scaler.update()

        else:
            D_real_fixed = D_CBCT(torch.cat((ct_fixed, cbct_fixed), dim=1))
            D_real_moving = D_MR(torch.cat((ct_moving, mr_moving), dim=1))
            D_fake_fixed = D_CBCT(torch.cat((ct_fixed_synths[0].detach(), cbct_fixed), dim=1))
            D_fake_moving = D_MR(torch.cat((ct_moving_synths[0].detach(), mr_moving), dim=1))

            loss_D_real_MR = 0
            loss_D_real_CBCT = 0
            loss_D_fake_MR = 0
            loss_D_fake_CBCT = 0

            for d_real_fixed, d_real_moving, d_fake_fixed, d_fake_moving in zip(D_real_fixed, D_real_moving,
                                                                                D_fake_fixed,
                                                                                D_fake_moving):
                loss_D_real_CBCT += torch.mean((torch.ones_like(d_real_fixed) - d_real_fixed) ** 2)
                loss_D_real_MR += torch.mean((torch.ones_like(d_real_moving) - d_real_moving) ** 2)
                loss_D_fake_CBCT += torch.mean((torch.zeros_like(d_fake_fixed) - d_fake_fixed) ** 2)
                loss_D_fake_MR += torch.mean((torch.zeros_like(d_fake_moving) - d_fake_moving) ** 2)
            loss_D_MR = (loss_D_real_MR + loss_D_fake_MR) / 2 / len(D_real_fixed)
            loss_D_CBCT = (loss_D_real_CBCT + loss_D_fake_CBCT) / 2 / len(D_real_fixed)

            loss_D_MR.backward()
            opt_D_MR.step()
            loss_D_CBCT.backward()
            opt_D_CBCT.step()

        disc_loss += (loss_D_MR.item() + loss_D_CBCT.item()) * batch_size
        synth_loss += loss_synth.item() * batch_size
        reg_loss += loss_reg.item() * args['lambda_reg'] * batch_size
        sim_loss += loss_sim.item() * batch_size
        smooth_loss += loss_smooth.item() * batch_size
        total_loss += loss_G.item() * batch_size
        unc_loss += loss_unc.item()*batch_size
        seg_loss += loss_seg.item() * batch_size

        log1_xloss += log1x.mean().item() * batch_size
        log2_xloss += log2x.mean().item() * batch_size
        log3_xloss += log3x.mean().item() * batch_size
        log1_yloss += log1y.mean().item() * batch_size
        log2_yloss += log2y.mean().item() * batch_size
        log3_yloss += log3y.mean().item() * batch_size
        log1_zloss += log1z.mean().item() * batch_size
        log2_zloss += log2z.mean().item() * batch_size
        log3_zloss += log3z.mean().item() * batch_size
        reg_xloss += reg_x.mean().item() * batch_size
        reg_yloss += reg_y.mean().item() * batch_size
        reg_zloss += reg_z.mean().item() * batch_size

    return total_loss / num_samples, synth_loss / num_samples, reg_loss / num_samples, \
           sim_loss / num_samples, smooth_loss / num_samples, disc_loss / num_samples,\
            unc_loss/num_samples, seg_loss/num_samples,log1_xloss/num_samples,\
            log2_xloss/num_samples,log3_xloss/num_samples,log1_yloss/num_samples,log2_yloss/num_samples,\
            log3_yloss/num_samples,log1_zloss/num_samples,log2_zloss/num_samples,log3_zloss/num_samples,\
            reg_xloss/num_samples,reg_yloss/num_samples,reg_zloss/num_samples


def valid_epoch(G, valid_dataloader, device):
    loss_L1 = 0
    loss_ncc = 0
    loss_seg = 0
    num_samples = 0

    for i, batch in tqdm(enumerate(valid_dataloader), 'validation phase:'):
        cbct_fixed, mr_moving, ct_fixed, ct_moving, seg_fixed, seg_moving = batch
        cbct_fixed = cbct_fixed.float().to(device)
        mr_moving = mr_moving.float().to(device)
        ct_fixed = ct_fixed.float().to(device)
        ct_moving = ct_moving.float().to(device)
        seg_fixed = seg_fixed.float().to(device)
        seg_moving = seg_moving.float().to(device)

        batch_size = cbct_fixed.size(0)
        num_samples += batch_size

        with torch.no_grad():
            flows, ct_moving_synths, ct_fixed_synths = G(mr_moving, cbct_fixed)

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
    }, os.path.join(args['save_path'], 'JSREDL_all_max_sup_{}_ep{}.pt'.format(args['exp_name'], epoch)))


def get_config(train_path, test_path, exp_name, save_path='./checkpoint/', pretrained=None, amp=False,
               lr_G=1e-4, lr_D=1e-4, lambda_l1=20, lambda_mind=10, lambda_seg=1, lambda_flow=2, lambda_reg=1,
               lambda_multi=0.3, start_epoch=0, batch_size=1, disc_channels=2, levels=0,
               separate_decoders=False, weak_supervision=False, schedule=[10, 0, 10],
               synthesis_checkpoint_mod=10, registration_checkpoint_mod=10, jsr_checkpoint_mod=10,lambda_unc = 1, lambda_inunc=1):
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
    args['lambda_unc'] = lambda_unc
    args['lambda_inunc'] = lambda_inunc
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
    parser.add_argument("--lambda_inunc", type=float,
                        dest="lambda_inunc", default=1, help="Lambda inside uncertainty loss function")
    parser.add_argument("--lambda_unc", type=float,
                        dest="lambda_unc", default=1, help="uncertainty loss weight")
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
                                          supervision=True, return_segmentation=args['weak_supervision'],uncertainty = True)
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                  num_workers=1, pin_memory=True)
    valid_dataset = RadOncValidationDataset(args['test_path'], num_samples=None, supervision=True,
                                            return_segmentation=True)

    # Build model
    G = JSRCascade(cdim=1, channels=(16, 32, 64, 128, 128), image_size=(128, 160, 128),
                   separate_decoders=args['separate_decoders'], res=True, version='v4').to(device)
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
    boardio = SummaryWriter(log_dir='./log/JSRCascade_unconly_{}_{}_{}_{}_{}_{}'.format(args['exp_name'], args['lambda_l1'],
                                                                          args['lambda_flow'], args['lambda_reg'],
                                                                          args['lambda_unc'],args['lambda_inunc']))

    if args['amp']:
        scaler = torch.cuda.amp.GradScaler()  # Cuda Mix Precision--save gpu memory
    else:
        scaler = None

    # Start training
    epoch = args['start_epoch']
    # =========== Synthesis Training Phase ================
    for i in range(args['schedule'][0]):

        total_loss, gan_loss, l1_loss, disc_loss = train_synth_epoch(args, G, D_MR, D_CBCT, train_dataloader,
                                                                     opt_G, opt_D_MR, opt_D_CBCT, device, scaler,scheduler_lambda=True)
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
                  
        reg_loss, sim_loss, smoothness_loss,unc_loss, seg_loss, log1_xloss,log2_xloss,log3_xloss,log1_yloss,log2_yloss,\
        log3_yloss,log1_zloss,log2_zloss,log3_zloss,reg_xloss,reg_yloss,reg_zloss= train_reg_epoch(args, G, train_dataloader, opt_G, device, scaler,scheduler_lambda=False)
        print('registration epoch: {}, reg_loss: {}, sim_loss: {}, smoothness_loss: {},uncertainty_loss: {}'.format(epoch, reg_loss,
                                                                                               sim_loss,
                                                                                               smoothness_loss,unc_loss))
        valid_l1_loss, valid_ncc_loss, valid_seg_loss = valid_epoch(G, valid_dataset, device)
        boardio.add_scalar('reg_loss', reg_loss, epoch)
        boardio.add_scalar('sim_loss', sim_loss, epoch)
        boardio.add_scalar('smoothness_loss', smoothness_loss, epoch)
        boardio.add_scalar('unc_loss', unc_loss, epoch)
        boardio.add_scalar('seg_loss', seg_loss, epoch)
        boardio.add_scalar('log1_xloss', log1_xloss, epoch)
        boardio.add_scalar('log2_xloss', log2_xloss, epoch)
        boardio.add_scalar('log3_xloss', log3_xloss, epoch)
        boardio.add_scalar('log1_yloss', log1_yloss, epoch)
        boardio.add_scalar('log2_yloss', log2_yloss, epoch)
        boardio.add_scalar('log3_yloss', log3_yloss, epoch)
        boardio.add_scalar('log1_zloss', log1_zloss, epoch)
        boardio.add_scalar('log2_zloss', log2_zloss, epoch)
        boardio.add_scalar('log3_zloss', log3_zloss, epoch)
        boardio.add_scalar('reg_xloss', reg_xloss, epoch)
        boardio.add_scalar('reg_yloss', reg_yloss, epoch)
        boardio.add_scalar('reg_zloss', reg_zloss, epoch)
        
        
        boardio.add_scalar('valid_l1_loss', valid_l1_loss, epoch)
        boardio.add_scalar('valid_ncc_loss', valid_ncc_loss, epoch)
        boardio.add_scalar('valid_seg_loss', valid_seg_loss, epoch)

        if (epoch + 1) % args['registration_checkpoint_mod'] == 0:
            save_model(G, D_MR, D_CBCT, opt_G, opt_D_MR, opt_D_CBCT, args, epoch)
        scheduler.step()
        epoch += 1

    # =========== Joint Training Phase ================
    for i in range(args['schedule'][2]):

        total_loss, synth_loss, reg_loss, sim_loss,smoothness_loss, disc_loss,\
        unc_loss, seg_loss, log1_xloss,log2_xloss,log3_xloss,log1_yloss,log2_yloss,\
        log3_yloss,log1_zloss,log2_zloss,log3_zloss,reg_xloss,reg_yloss,reg_zloss= train_joint_epoch(args, G, D_MR, D_CBCT, train_dataloader,
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
        boardio.add_scalar('unc_loss', unc_loss, epoch)
        boardio.add_scalar('seg_loss', seg_loss, epoch)
        boardio.add_scalar('log1_xloss', log1_xloss, epoch)
        boardio.add_scalar('log2_xloss', log2_xloss, epoch)
        boardio.add_scalar('log3_xloss', log3_xloss, epoch)
        boardio.add_scalar('log1_yloss', log1_yloss, epoch)
        boardio.add_scalar('log2_yloss', log2_yloss, epoch)
        boardio.add_scalar('log3_yloss', log3_yloss, epoch)
        boardio.add_scalar('log1_zloss', log1_zloss, epoch)
        boardio.add_scalar('log2_zloss', log2_zloss, epoch)
        boardio.add_scalar('log3_zloss', log3_zloss, epoch)
        boardio.add_scalar('reg_xloss', reg_xloss, epoch)
        boardio.add_scalar('reg_yloss', reg_yloss, epoch)
        boardio.add_scalar('reg_zloss', reg_zloss, epoch)
        boardio.add_scalar('valid_l1_loss', valid_l1_loss, epoch)
        boardio.add_scalar('valid_ncc_loss', valid_ncc_loss, epoch)
        boardio.add_scalar('valid_seg_loss', valid_seg_loss, epoch)

        if (epoch + 1) % args['jsr_checkpoint_mod'] == 0:
            save_model(G, D_MR, D_CBCT, opt_G, opt_D_MR, opt_D_CBCT, args, epoch)
        scheduler.step()
        epoch += 1
