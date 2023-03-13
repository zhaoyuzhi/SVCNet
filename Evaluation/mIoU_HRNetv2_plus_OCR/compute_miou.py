import os
import cv2
import numpy as np

#gt_path = 'gt_512/Annotations/1080p/'
#pred_path = 'output/train_val_out_img_mark/Annotations/1080p/'

gt_path = '/media/zyz/Elements/submitted papers/SVCNet/evaluation/mIoU_HRNet_DAVIS/data/davis/Annotations/480p'
pred_path = '/media/zyz/Elements/submitted papers/SVCNet/evaluation/mIoU_HRNet_DAVIS/data/davis/Annotations/480p'

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    # output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    # seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    # seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)
    # import pdb; pdb.set_trace()
    seg_pred = pred[None]
    seg_gt = label[None]

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

fn_list = './data/list/davis/val.lst'

list_f = open(fn_list, 'r')
confusion_matrix = np.zeros((2, 2))

for im_p in list_f.readlines():
    
    relative_name = im_p.split('Annotations/1080p/')[-1].strip()
    imp_gt = os.path.join(gt_path, relative_name)
    imp_pred = os.path.join(pred_path, relative_name)

    gt = cv2.imread(imp_gt, cv2.IMREAD_GRAYSCALE) // 255
    pred = cv2.imread(imp_pred, cv2.IMREAD_GRAYSCALE) // 255

    confusion_matrix[...] += get_confusion_matrix(gt, pred, gt.shape, 2, 255)

pos = confusion_matrix.sum(1)
res = confusion_matrix.sum(0)
tp = np.diag(confusion_matrix)
IoU_array = (tp / np.maximum(1.0, pos + res - tp))
mean_IoU = IoU_array.mean()
print('mean IOU: ', mean_IoU)
