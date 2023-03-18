import time
import datetime
import itertools
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import logging

import datasets.data_utils as dutils
import datasets.ssnet_DAVISVidevo as ssnet_dataset
import networks.pwcnet as pwcnet
import networks.loss as loss_def
import utils

def Trainer(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    folder_name = opt.run_name

    # Initialize logging
    logging_path = os.path.join(opt.logging_path, folder_name)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    logging_file_path = os.path.join(logging_path, 'train.log')
    logging.basicConfig(level = logging.INFO, filename = logging_file_path, filemode = 'w')

    # Initialize saving folders
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
    criterion_TV = loss_def.TVLoss().cuda()
    
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
    optimizer_G = torch.optim.Adam(
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
    
    if opt.use_lmdb:
        trainset = ssnet_dataset.MultiFramesDataset_lmdb(opt, imglist, classlist)
    else:
        trainset = ssnet_dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    #dataloader = utils.create_dataloader(trainset, opt)
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Define whether using the long-range connection
    use_long_connection = False
    if hasattr(ssnet, 'module'):
        if hasattr(ssnet.module, 'corr'):
            use_long_connection = True
    else:
        if hasattr(ssnet, 'corr'):
            use_long_connection = True

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, in_transform_part, out_part, color_scribble_part, out_large_part) in enumerate(dataloader):
            
            # ----------------------------------------
            # To cuda
            input_frames = []
            input_transform_frames = []
            gt_frames = []
            gt_large_frames = []
            scribble_frames = []
            for i in range(opt.iter_frames):
                input_frames.append(in_part[i].cuda())
                input_transform_frames.append(in_transform_part[i].cuda())
                gt_frames.append(out_part[i].cuda())
                gt_large_frames.append(out_large_part[i].cuda())
                scribble_frames.append(color_scribble_part[i].cuda())
            
            # Forward CPNet first to obtain the colorized frames
            cpnet_frames = []
            with torch.no_grad():
                for j in range(opt.iter_frames):
                    x_t = input_frames[j][:, [0], :, :]
                    color_scribble = scribble_frames[j]
                    cpnet_out, _ = cpnet(x_t, color_scribble)
                    cpnet_frames.append(cpnet_out)
            
            # Prepare all the frames
            overall_frames = opt.iter_frames * 2 - 1    # e.g., 13 = 7 * 2 - 1
            num_of_input_frames_for_ssnet = 7
            ssnet_training_iter_per_batch = overall_frames - num_of_input_frames_for_ssnet + 1

            input_frames_supp = []
            input_transform_frames_supp = []
            gt_frames_supp = []
            gt_large_frames_supp = []
            cpnet_frames_supp = []
            for k in range(opt.iter_frames):
                input_frames_supp.append(input_frames[opt.iter_frames - 1 - k])
                input_transform_frames_supp.append(input_transform_frames[opt.iter_frames - 1 - k])
                gt_frames_supp.append(gt_frames[opt.iter_frames - 1 - k])
                gt_large_frames_supp.append(gt_large_frames[opt.iter_frames - 1 - k])
                cpnet_frames_supp.append(cpnet_frames[opt.iter_frames - 1 - k])
            input_frames = input_frames + input_frames_supp
            input_transform_frames = input_transform_frames + input_transform_frames_supp
            gt_frames = gt_frames + gt_frames_supp
            gt_large_frames = gt_large_frames + gt_large_frames_supp
            cpnet_frames = cpnet_frames + cpnet_frames_supp
            # ----------------------------------------

            # ----------------------------------------
            # Train Generator
            optimizer_G.zero_grad()
            
            # Define loss functions
            loss_L1 = 0
            loss_TV = 0
            loss_flow_short = 0
            loss_flow_long = 0
            loss_L1_large = 0
            loss_reg = 0

            # Forward SSNet and compute loss functions
            # 0123456 - 1234567 - 2345678 - ... - 6789101112 (overall 7 groups)
            for ii in range(ssnet_training_iter_per_batch):
                # warp previous and leading frames through optical flows
                center_id = ii + 3
                flow_minus3_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii], input_frames[center_id], drange = True, reshape = True)
                flow_minus2_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+1], input_frames[center_id], drange = True, reshape = True)
                flow_minus1_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+2], input_frames[center_id], drange = True, reshape = True)
                flow_add1_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+4], input_frames[center_id], drange = True, reshape = True)
                flow_add2_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+5], input_frames[center_id], drange = True, reshape = True)
                flow_add3_to_current = pwcnet.PWCEstimate(flownet, input_frames[ii+6], input_frames[center_id], drange = True, reshape = True)
                if ii > 0:
                    flow_zero_to_current = pwcnet.PWCEstimate(flownet, input_frames[0], input_frames[center_id], drange = True, reshape = True)
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
                
                # Define mask list and cpnet list
                mask_warp_list = [mask_minus3_to_current, mask_minus2_to_current, mask_minus1_to_current, mask_add1_to_current, mask_add2_to_current, mask_add3_to_current]
                cpnet_warp_list = [cpnet_t_minus3_warp, cpnet_t_minus2_warp, cpnet_t_minus1_warp, cpnet_t_add1_warp, cpnet_t_add2_warp, cpnet_t_add3_warp]

                # arrange all inputs for SSNet
                if use_long_connection and ii == 0:
                    # cpnet_ab_to_PIL_rgb recieves: cv2 format grayscale tensor + cpnet format tensor
                    cpnet_t_0_PIL_rgb = dutils.cpnet_ab_to_PIL_rgb(input_frames[0][:, [0], :, :], cpnet_frames[0])
                    for batch in range(len(cpnet_t_0_PIL_rgb)):
                        cpnet_t_0_PIL_rgb_batch = cpnet_t_0_PIL_rgb[batch]
                        cpnet_t_0_PIL_rgb_batch = trainset.transform(cpnet_t_0_PIL_rgb_batch).unsqueeze(0).cuda()
                        if batch == 0:
                            IB_lab = cpnet_t_0_PIL_rgb_batch
                        else:
                            IB_lab = torch.cat((IB_lab, cpnet_t_0_PIL_rgb_batch), 0)
                    with torch.no_grad():
                        I_reference_l = IB_lab[:, 0:1, :, :]
                        I_reference_ab = IB_lab[:, 1:3, :, :]
                        I_reference_rgb = dutils.tensor_lab2rgb(torch.cat((dutils.uncenter_l(I_reference_l), I_reference_ab), dim = 1))
                        if opt.multi_gpu:
                            features_B = ssnet.module.corr.vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess = True)
                        else:
                            features_B = ssnet.corr.vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess = True)

                # SSNet forward propagation
                if use_long_connection:
                    ssnet_t, ssnet_sr_t, residual = ssnet(input_transform_frames[center_id], IB_lab, features_B, cpnet_frames[center_id], ssnet_t_minus1_warp, cpnet_warp_list, mask_warp_list)
                else:
                    ssnet_t, ssnet_sr_t, residual = ssnet(input_transform_frames[center_id], cpnet_frames[center_id], ssnet_t_minus1_warp, cpnet_warp_list, mask_warp_list)
                
                # Pixel-level Colorization loss
                loss_L1 += criterion_L1(ssnet_t, gt_frames[center_id])
                loss_L1_large += criterion_L1(ssnet_sr_t, gt_large_frames[center_id])
                
                # Total variation loss
                loss_TV += criterion_TV(ssnet_t)
                
                # Compute the short-term loss
                if ii > 0:
                    loss_flow_short += criterion_L1(mask_minus1_to_current * ssnet_t, mask_minus1_to_current * ssnet_t_minus1_warp)
                
                # Compute the long-term loss
                if ii > 1:
                    loss_flow_long += criterion_L1(mask_zero_to_current * ssnet_t, mask_zero_to_current * ssnet_first_output_warp)
                
                # Compute the regularization loss
                loss_reg += criterion_L1(residual, torch.zeros_like(residual).cuda())
                
            # Overall Loss and optimize
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_tv * loss_TV + opt.lambda_flow_short * loss_flow_short + opt.lambda_flow_long * loss_flow_long + opt.lambda_reg * loss_reg
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            log_str = "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [L1 Loss (SR): %.4f] [TV Loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] [Regularization Loss: %.8f] Time_left: %s" % \
                ((epoch + 1), opt.epochs, iteration, len(dataloader), loss_L1.item() / (ssnet_training_iter_per_batch), \
                        loss_L1_large.item() / (ssnet_training_iter_per_batch), loss_TV.item() / (ssnet_training_iter_per_batch), \
                            loss_flow_short.item() / (ssnet_training_iter_per_batch - 1), \
                                loss_flow_long.item() / (ssnet_training_iter_per_batch - 2), \
                                    loss_reg.item() / (ssnet_training_iter_per_batch), time_left)
            
            print(log_str)
            logging.info(log_str)
            
            # Sample by iteration
            if iters_done % opt.sample_iter == 0:
                img_list = [input_frames[center_id], gt_frames[center_id], cpnet_frames[center_id], ssnet_t]
                name_list = ['grayscale', 'gt', 'cpnet_out', 'ssnet_out']
                #img_list += [p_t[:, 2:4, :, :], p_t[:, 4:6, :, :], p_t[:, 6:8, :, :], p_t[:, 8:10, :, :], p_t[:, 10:12, :, :], p_t[:, 12:14, :, :], p_t[:, 14:16, :, :]]
                #name_list += ['pt_lastout', 'refined1', 'refined2', 'refined3', 'refined4', 'refined5', 'refined6']
                utils.sample(sample_folder = sample_path, sample_name = str(iters_done + 1), img_list = img_list, name_list = name_list)

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), ssnet)

            # Learning rate decrease at certain epochs or iterations
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), opt.lr_g_ssnet, optimizer_G)
