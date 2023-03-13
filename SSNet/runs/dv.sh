CUDA_VISIBLE_DEVICES=0 python -u train.py \
--pre_train_cpnet_type 'CPNet_VGG16_Seg' \
--pre_train_ssnet_type 'SSNet' \
--multi_gpu False \
--cudnn_benchmark True \
--save_path './models' \
--sample_path './samples' \
--logging_path './logs' \
--run_name 'dv' \
--save_mode 'epoch' \
--save_by_epoch 500 \
--sample_iter 50 \
--cpnet_path './trained_models/CPNet/models_2nd_vimeo_256p/CPNet_VGG16_Seg/cpnet_epoch10_batchsize16.pth' \
--ssnet_path '' \
--warn_path './trained_models/WARN/Warp_Artifact_Removal_Net_256p_in_epoch2000_bs16.pth' \
--corrnet_vgg_path './trained_models/CorrNet/vgg19_conv.pth' \
--corrnet_nonlocal_path './trained_models/CorrNet/nonlocal_net_iter_76000.pth' \
--srnet_path './trained_models/SRNet/Color_Embedding_SR_Net_normnone_epoch40_bs4.pth' \
--pwcnet_path './trained_models/PWCNet/pwcNet-default.pytorch' \
--perceptual_path './trained_models/vgg16_pretrained.pth' \
--vgg_name './trained_models/vgg16_pretrained.pth' \
--epochs 2001 \
--batch_size 1 \
--lr_g_cpnet 0 \
--lr_g_ssnet 5e-5 \
--lr_decrease_mode 'epoch' \
--lr_decrease_epoch 1000 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--lambda_l1 1 \
--lambda_seg 0 \
--lambda_tv 0 \
--lambda_flow_short 0 \
--lambda_flow_long 0 \
--lambda_reg 1 \
--mask_para 200 \
--baseroot '/home/zyz/Documents/dataset/DAVIS_videvo_train_lmdb' \
--segroot '/home/zyz/Documents/dataset/DAVIS_videvo_train_lmdb' \
--video_class_txt './txt/DAVIS_videvo_train_class.txt' \
--video_imagelist_txt './txt/DAVIS_videvo_train_imagelist.txt' \
--use_lmdb True \
--iter_frames 7 \
--crop_size_h 256 \
--crop_size_w 448 \
--upsample_ratio 2 \
--color_point_prob 0.5 \
--color_point 40 \
--color_width 5 \
--color_blur_width 11