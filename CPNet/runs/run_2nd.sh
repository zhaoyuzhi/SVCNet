opt=CPNet_VGG16_Seg
num_of_scribbles=40

python train.py \
--pre_train_cpnet_type ${opt} \
--base_root ./data/DAVIS_videvo/DAVIS_videvo_train \
--seg_root ./data/DAVIS_videvo/DAVIS_videvo_fusion_saliency_maps_and_segmentations \
--save_path models_2nd_dv_256p \
--sample_path samples_2nd_dv_256p \
--multi_gpu True \
--checkpoint_interval 100 \
--finetune_path models_1st/${opt}/cpnet_epoch20_batchsize32.pth \
--epochs 1000 \
--batch_size 4 \
--lr_g 5e-5 \
--lambda_l1 1 \
--lambda_seg 0.1 \
--lr_decrease_epoch 500 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--crop_size_w 448 \
--crop_size_h 256 \
--color_point_prob 0.5 \
--color_point ${num_of_scribbles} \
--color_width 5 \
--color_blur_width 5 \