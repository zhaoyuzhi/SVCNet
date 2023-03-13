import time
import datetime
import itertools
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import datasets.ssnet_DAVISVidevo as ssnet_dataset
import networks.pwcnet as pwcnet
import utils

def Trainer(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    folder_name = '%s_l1%d_seg%d_flow%d' % \
        (opt.pre_train_ssnet_type, opt.lambda_l1, opt.lambda_seg, opt.lambda_flow_short)
    save_path = os.path.join(opt.save_path, folder_name)
    sample_path = os.path.join(opt.sample_path, folder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    
    # Initialize Generator
    cpnet, ssnet = utils.create_generator(opt)
    flownet = utils.create_pwcnet(opt)

    # To device
    if opt.multi_gpu:
        cpnet = nn.DataParallel(cpnet)
        cpnet = cpnet.cuda()
        ssnet = nn.DataParallel(ssnet)
        ssnet = ssnet.cuda()
        flownet = nn.DataParallel(flownet)
        flownet = flownet.cuda()
    else:
        cpnet = cpnet.cuda()
        ssnet = ssnet.cuda()
        flownet = flownet.cuda()

    # Optimizers
    optimizer_cpnet = torch.optim.Adam(
        cpnet.parameters(), lr = opt.lr_g_cpnet, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay
    )
    optimizer_ssnet = torch.optim.Adam(
        ssnet.parameters(), lr = opt.lr_g_ssnet, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay
    )
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, lr, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = lr * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, ssnet):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == 'epoch':
            ssnet_name = 'ssnet_epoch%d_bs%d.pth' % (epoch, opt.batch_size)
        if opt.save_mode == 'iter':
            ssnet_name = 'ssnet_iter%d_bs%d.pth' % (iteration, opt.batch_size)
        save_ssnet_name = os.path.join(save_path, ssnet_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(ssnet.module.state_dict(), save_ssnet_name)
                    print('The trained model is saved as %s' % (save_ssnet_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(ssnet.module.state_dict(), save_ssnet_name)
                    print('The trained model is saved as %s' % (save_ssnet_name))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(ssnet.state_dict(), save_ssnet_name)
                    print('The trained model is saved as %s' % (save_ssnet_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(ssnet.state_dict(), save_ssnet_name)
                    print('The trained model is saved as %s' % (save_ssnet_name))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    imglist = utils.text_readlines(opt.video_imagelist_txt)
    classlist = utils.text_readlines(opt.video_class_txt)
    trainset = ssnet_dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = utils.create_dataloader(trainset, opt)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, out_part, color_scribble_part) in enumerate(dataloader):
            
            # ----------------------------------------
            # To cuda
            input_frames = []
            gt_frames = []
            scribble_frames = []
            for i in range(opt.iter_frames):
                input_frames.append(in_part[i].cuda())
                gt_frames.append(out_part[i].cuda())
                scribble_frames.append(color_scribble_part[i].cuda())
            
            # Forward CPNet first to obtain the colorized frames
            cpnet_frames = []
            for j in range(opt.iter_frames):
                x_t = input_frames[j][:, [0], :, :]
                color_scribble = scribble_frames[j]
                cpnet_out, _ = cpnet(x_t, color_scribble)
                cpnet_frames.append(cpnet_out)
            # save the first CPNet's output
            cpnet_out_0 = cpnet_frames[0]
            
            # Prepare all the frames
            overall_frames = opt.iter_frames
            num_of_input_frames_for_ssnet = 7
            ssnet_training_iter_per_batch = overall_frames - num_of_input_frames_for_ssnet + 1
            # ----------------------------------------

            # ----------------------------------------
            # Train Generator
            optimizer_cpnet.zero_grad()
            optimizer_ssnet.zero_grad()
            states = None
            
            # Define loss functions
            loss_L1 = 0
            loss_flow_short = 0
            loss_flow_long = 0

            # Forward SSNet and compute loss functions
            # 0123456 - 1234567 - 2345678 - ... - 6789101112 (overall 7 groups)
            for ii in range(ssnet_training_iter_per_batch):
                # warp previous and leading frames through optical flows
                center_id = ii + 3
                flow_minus3_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii], input_frames[center_id], drange = True, reshape = False)
                flow_minus2_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+1], input_frames[center_id], drange = True, reshape = False)
                flow_minus1_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+2], input_frames[center_id], drange = True, reshape = False)
                flow_add1_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+4], input_frames[center_id], drange = True, reshape = False)
                flow_add2_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+5], input_frames[center_id], drange = True, reshape = False)
                flow_add3_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+6], input_frames[center_id], drange = True, reshape = False)
                if ii > 0:
                    flow_zero_to_current = pwcnet.PWCEstimate(flownet, input_frames[0], input_frames[center_id], drange = True, reshape = False)
                else:
                    flow_zero_to_current = flow_minus3_to_current
                
                # compute visible mask
                x_t_minus3_warp = pwcnet.PWCNetBackward(input_frames[ii], flow_minus3_to_current)
                x_t_minus2_warp = pwcnet.PWCNetBackward(input_frames[ii+1], flow_minus2_to_current)
                x_t_minus1_warp = pwcnet.PWCNetBackward(input_frames[ii+2], flow_minus1_to_current)
                x_t_add1_warp = pwcnet.PWCNetBackward(input_frames[ii+4], flow_add1_to_current)
                x_t_add2_warp = pwcnet.PWCNetBackward(input_frames[ii+5], flow_add2_to_current)
                x_t_add3_warp = pwcnet.PWCNetBackward(input_frames[ii+6], flow_add3_to_current)
                x_t_zero_warp = pwcnet.PWCNetBackward(input_frames[0], flow_zero_to_current)
                mask_minus3_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_minus3_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_minus2_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_minus2_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_minus1_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_minus1_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_add1_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_add1_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_add2_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_add2_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_add3_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_add3_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)
                mask_zero_to_current = torch.exp(- opt.mask_para * torch.sum(x_t_zero_warp - input_frames[center_id], dim = 1).pow(2)).unsqueeze(1)

                # warp CPNet outputs to current position
                cpnet_t_minus3_warp = pwcnet.PWCNetBackward(cpnet_frames[ii], flow_minus3_to_current)
                cpnet_t_minus2_warp = pwcnet.PWCNetBackward(cpnet_frames[ii+1], flow_minus2_to_current)
                cpnet_t_minus1_warp = pwcnet.PWCNetBackward(cpnet_frames[ii+2], flow_minus1_to_current)
                cpnet_t_add1_warp = pwcnet.PWCNetBackward(cpnet_frames[ii+4], flow_add1_to_current)
                cpnet_t_add2_warp = pwcnet.PWCNetBackward(cpnet_frames[ii+5], flow_add2_to_current)
                cpnet_t_add3_warp = pwcnet.PWCNetBackward(cpnet_frames[ii+6], flow_add3_to_current)
                
                # save SSNet's last output and warp it
                if ii > 0:
                    ssnet_t_minus1 = ssnet_t.detach()
                    if ii == 1:
                        ssnet_first_output = ssnet_t_minus1
                    ssnet_first_output_warp = pwcnet.PWCNetBackward(ssnet_first_output, flow_zero_to_current)
                    ssnet_t_minus1_warp = pwcnet.PWCNetBackward(ssnet_t_minus1, flow_minus1_to_current)
                else:
                    ssnet_t_minus1_warp = cpnet_frames[center_id]
                
                # arrange all inputs for SSNet
                # x_t = input_frames[center_id]
                # cpnet_t = cpnet_frames[center_id]
                if ii == 0:
                    cpnet_t_0_toRGB = utils.ab_to_rgb(input_frames[0][:, [0], :, :], cpnet_frames[0])
                mask_warp_list = [mask_minus3_to_current, mask_minus2_to_current, mask_minus1_to_current, mask_add1_to_current, mask_add2_to_current, mask_add3_to_current]
                cpnet_warp_list = [cpnet_t_minus3_warp, cpnet_t_minus2_warp, cpnet_t_minus1_warp, cpnet_t_add1_warp, cpnet_t_add2_warp, cpnet_t_add3_warp]

                # SSNet forward propagation
                ssnet_t, states = ssnet(input_frames[center_id], cpnet_t_0_toRGB, cpnet_frames[center_id], ssnet_t_minus1_warp, cpnet_warp_list, mask_warp_list, states)
                states = utils.repackage_hidden(states)
                
                # Pixel-level Colorization loss
                loss_L1 += criterion_L1(ssnet_t, gt_frames[center_id])

                # Compute the short-term loss
                if ii > 0:
                    loss_flow_short += criterion_L1(mask_minus1_to_current * ssnet_t, mask_minus1_to_current * ssnet_t_minus1_warp)
                
                # Compute the long-term loss
                if ii > 1:
                    loss_flow_long += criterion_L1(mask_zero_to_current * ssnet_t, mask_zero_to_current * ssnet_first_output_warp)
                
            # Overall Loss and optimize
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_flow_short * loss_flow_short + opt.lambda_flow_long * loss_flow_long
            loss.backward()
            optimizer_cpnet.step()
            optimizer_ssnet.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, iteration, len(dataloader), \
                    loss_L1.item() / (opt.iter_frames - 0), \
                        loss_flow_short.item() / (opt.iter_frames - 1), \
                            loss_flow_long.item() / (opt.iter_frames - 2), time_left))

        # Save model at certain epochs or iterations
        save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), ssnet)

        # Learning rate decrease at certain epochs
        adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), opt.lr_g_cpnet, optimizer_cpnet)
        adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), opt.lr_g_ssnet, optimizer_ssnet)
        utils.sample(input_frames[center_id], gt_frames[center_id], color_scribble, cpnet_frames[center_id], ssnet_t, sample_path, (epoch + 1))
