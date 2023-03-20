import argparse

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General training parameters
    parser.add_argument('--pre_train_cpnet_type', type = str, default = 'CPNet_VGG16_Seg', help = 'pre_train_cpnet_type')
    parser.add_argument('--pre_train_ssnet_type', type = str, default = 'SSNet', help = 'pre_train_ssnet_type')
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'nn.Parallel needs or not')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--save_path', type = str, default = './models', help = 'save the pre-trained model to certain path')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'save the training samples to certain path')
    parser.add_argument('--logging_path', type = str, default = './logs', help = 'save the training logs to certain path')
    parser.add_argument('--run_name', type = str, default = '', help = 'the name of runned file')
    parser.add_argument('--save_mode', type = str, default = 'epoch', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 1, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--sample_iter', type = int, default = 50, help = 'interval between saving samples')
    # Pre-trained model parameters
    parser.add_argument('--cpnet_path', type = str, default = './trained_models/CPNet/models_2nd_dv_256p/CPNet_VGG16_Seg/cpnet_epoch20_batchsize32.pth', help = 'the load name of models')
    parser.add_argument('--ssnet_path', type = str, default = '', help = 'the load name of models')
    parser.add_argument('--warn_path', type = str, default = './trained_models/SSNet/WARN/Warp_Artifact_Removal_Net_256p_in_epoch2000_bs16.pth', help = 'the load name of models')
    parser.add_argument('--corrnet_vgg_path', type = str, default = './trained_models/SSNet/CorrNet/vgg19_conv.pth', help = 'the load name of models')
    parser.add_argument('--corrnet_nonlocal_path', type = str, default = './trained_models/SSNet/CorrNet/nonlocal_net_iter_76000.pth', help = 'the load name of models')
    parser.add_argument('--srnet_path', type = str, default = './trained_models/SSNet/SRNet/Color_Embedding_SR_Net_normnone_epoch40_bs4.pth', help = 'the load name of models')
    parser.add_argument('--pwcnet_path', type = str, default = './trained_models/PWCNet/pwcNet-default.pytorch', help = 'the load name of models')
    parser.add_argument('--perceptual_path', type = str, default = './trained_models/Others/vgg16_pretrained.pth', help = 'the load name of models')
    parser.add_argument('--vgg_name', type = str, default = './trained_models/Others/vgg16_pretrained.pth', help = 'the load name of models')
    # Optimization parameters
    parser.add_argument('--epochs', type = int, default = 41, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g_cpnet', type = float, default = 0, help = 'Adam: learning rate')
    parser.add_argument('--lr_g_ssnet', type = float, default = 5e-5, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'epoch', help = 'lr decrease mode')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 10000, help = 'lr decrease at certain iteration and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    # Loss balancing parameters
    parser.add_argument('--lambda_l1', type = float, default = 1, help = 'coefficient for L1 Loss')
    parser.add_argument('--lambda_seg', type = float, default = 0.1, help = 'coefficient for Segmentation Loss')
    parser.add_argument('--lambda_tv', type = float, default = 0, help = 'coefficient for TV Loss')
    parser.add_argument('--lambda_flow_short', type = float, default = 0, help = 'coefficient for Temporal Loss')
    parser.add_argument('--lambda_flow_long', type = float, default = 0, help = 'coefficient for Temporal Loss')
    parser.add_argument('--mask_para', type = float, default = 200, help = 'coefficient for visible mask')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'visible mask channel')
    parser.add_argument('--seg_channels', type = int, default = 1, help = 'output segmentation image')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--start_channels_warn', type = int, default = 16, help = 'warn channel')
    parser.add_argument('--start_channels_comb', type = int, default = 32, help = 'combination net channels')
    parser.add_argument('--start_channels_sr', type = int, default = 32, help = 'super resolution net channel')
    parser.add_argument('--lambda_value', type = float, default = 500, help = 'lambda_value of WLS')
    parser.add_argument('--sigma_color', type = float, default = 4, help = 'sigma_color of WLS')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_warn', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_comb', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_sr', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_warn', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--norm_comb', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_sr', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = './data/DAVIS_videvo/DAVIS_videvo_train', help = 'the base training folder')
    parser.add_argument('--segroot', type = str, default = './data/DAVIS_videvo/DAVIS_videvo_fusion_saliency_maps_and_segmentations', help = 'the base training folder')
    parser.add_argument('--video_class_txt', type = str, default = './txt/DAVIS_videvo_train_class.txt', help = 'the path that contains DAVIS_videvo_train_class.txt')
    parser.add_argument('--video_imagelist_txt', type = str, default = './txt/DAVIS_videvo_train_imagelist.txt', help = 'the path that contains DAVIS_videvo_train_imagelist.txt')
    parser.add_argument('--use_lmdb', type = bool, default = False, help = 'whether use the lmdb to read data')
    parser.add_argument('--iter_frames', type = int, default = 7, help = 'number of iter_frames in one iteration; +1 since the first frame is not counted')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'single patch size') # second stage (128p, 256p, 448p): 128, 256, 448
    parser.add_argument('--crop_size_w', type = int, default = 448, help = 'single patch size') # second stage (128p, 256p, 448p): 256, 448, 832
    parser.add_argument('--upsample_ratio', type = int, default = 2, help = 'the upsampling ratio for the srnet')
    # Color scribble parameters
    parser.add_argument('--color_point_prob', type = float, default = 0.5, help = 'probability of color scribbles')
    parser.add_argument('--color_point', type = int, default = 40, help = 'number of color scribbles')
    parser.add_argument('--color_width', type = int, default = 5, help = 'width of each color scribble')
    parser.add_argument('--color_blur_width', type = int, default = 11, help = 'Gaussian blur width of each color scribble')
    opt = parser.parse_args()
    print(opt)
    
    # Enter main function
    #import trainer_fixcpnet as trainer
    import trainer as trainer
    trainer.Trainer(opt)
    