opt=CPNet_VGG16_Seg
num_of_scribbles=40

python train.py \
--pre_train_cpnet_type ${opt} \
--base_root ./dataset/ILSVRC2012_train_256 \
--seg_root ./dataset/ILSVRC2012_train_256_saliencymap \
--save_path models_1st \
--sample_path samples_1st \
--multi_gpu True \
--checkpoint_interval 1 \
--epochs 41 \
--batch_size 4 \
--lr_g 1e-4 \
--lambda_l1 1 \
--lambda_seg 0.1 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--crop_size_w 256 \
--crop_size_h 256 \
--color_point_prob 0.5 \
--color_point ${num_of_scribbles} \
--color_width 5 \
--color_blur_width 5 \
