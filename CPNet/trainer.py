import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import itertools

import network
import dataset
import utils

def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    save_path = os.path.join(opt.save_path, opt.pre_train_cpnet_type)
    sample_path = os.path.join(opt.sample_path, opt.pre_train_cpnet_type)
    utils.check_path(save_path)
    utils.check_path(sample_path)

    # Build networks
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'cpnet_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ScribbleColorizationDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, img, color_scribble, seg) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            img = img.cuda()                                                # out: [B, 2, 256, 256]
            color_scribble = color_scribble.cuda()                          # out: [B, 2, 256, 256]

            optimizer_g.zero_grad()

            # forward propagation
            img_out = generator(grayscale, color_scribble)

            ### second stage: jointly denoising and colorization
            # Color L1 Loss
            ColorL1Loss = L1Loss(img_out, img)

            # Compute losses
            loss = ColorL1Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Color L1 Loss: %.5f] time_left: %s" %
                ((epoch + opt.epochs_overhead + 1), opt.epochs, batch_idx, len(dataloader), ColorL1Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + opt.epochs_overhead + 1), opt, opt.lr_g)

        # Save the model
        save_model(generator, (epoch + opt.epochs_overhead + 1), opt)
        utils.sample(grayscale, img, color_scribble, img_out, sample_path, (epoch + opt.epochs_overhead + 1))

def Trainer_Seg(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    save_path = os.path.join(opt.save_path, opt.pre_train_cpnet_type)
    sample_path = os.path.join(opt.sample_path, opt.pre_train_cpnet_type)
    utils.check_path(save_path)
    utils.check_path(sample_path)

    # Build networks
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'cpnet_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ScribbleColorizationDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, img, color_scribble, seg) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            img = img.cuda()                                                # out: [B, 2, 256, 256]
            color_scribble = color_scribble.cuda()                          # out: [B, 2, 256, 256]
            seg = seg.cuda()                                                # out: [B, 1, 256, 256]

            optimizer_g.zero_grad()

            # forward propagation
            img_out, seg_out = generator(grayscale, color_scribble)

            ### second stage: jointly denoising and colorization
            # Color L1 Loss
            ColorL1Loss = L1Loss(img_out, img)

            # Sal L1 Loss
            SalL1Loss = L1Loss(seg_out, seg)

            # Compute losses
            loss = ColorL1Loss + opt.lambda_seg * SalL1Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Color L1 Loss: %.5f] [Sal L1 Loss: %.5f] time_left: %s" %
                ((epoch + opt.epochs_overhead + 1), opt.epochs, batch_idx, len(dataloader), ColorL1Loss.item(), SalL1Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + opt.epochs_overhead + 1), opt, opt.lr_g)

        # Save the model
        save_model(generator, (epoch + opt.epochs_overhead + 1), opt)
        utils.sample(grayscale, img, color_scribble, img_out, sample_path, (epoch + opt.epochs_overhead + 1))
