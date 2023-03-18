import argparse

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train_cpnet_type', type = str, default = 'CPNet_VGG16_Seg', help = 'pre_train_cpnet_type')
    parser.add_argument('--save_path', type = str, default = 'models', help = 'the save path for trained models')
    parser.add_argument('--sample_path', type = str, default = 'samples', help = 'the save path for trained models')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--finetune_path', type = str, default = '', help = 'the load name of models')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 40, help = 'number of epochs of training')
    parser.add_argument('--epochs_overhead', type = int, default = 0, help = 'number of trained epochs')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 1, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_seg', type = float, default = 0.1, help = 'the parameter of saliency map loss')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--scribble_channels', type = int, default = 2, help = 'input scribble image')
    parser.add_argument('--out_channels', type = int, default = 2, help = 'output RGB image')
    parser.add_argument('--seg_channels', type = int, default = 1, help = 'output segmentation image')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--base_root', type = str, default = "C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256", \
        help = 'the base training folder')
    parser.add_argument('--seg_root', type = str, default = "C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256", \
        help = 'the base training folder')
    parser.add_argument('--vgg_name', type = str, default = "./trained_models/vgg16_pretrained.pth", \
        help = 'load the pre-trained vgg model with certain epoch')
    parser.add_argument('--crop_size_w', type = int, default = 256, help = 'size of image')
    parser.add_argument('--crop_size_h', type = int, default = 256, help = 'size of image')
    # color scribble parameters
    parser.add_argument('--color_point_prob', type = float, default = 0.5, help = 'probability of color scribbles')
    parser.add_argument('--color_point', type = int, default = 40, help = 'number of color scribbles')
    parser.add_argument('--color_width', type = int, default = 5, help = 'width of each color scribble')
    parser.add_argument('--color_blur_width', type = int, default = 11, help = 'Gaussian blur width of each color scribble')
    opt = parser.parse_args()
    print(opt)
    
    # Enter main function
    import trainer
    if 'seg' in opt.pre_train_cpnet_type or 'Seg' in opt.pre_train_cpnet_type:
        trainer.Trainer_Seg(opt)
    else:
        trainer.Trainer(opt)
    