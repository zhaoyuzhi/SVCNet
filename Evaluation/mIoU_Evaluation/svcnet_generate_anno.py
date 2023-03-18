from config import config
from model.mpn import MaskPyramids
from data.davis import Davis
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int)

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

def main(savepath, savepath2):
    config.merge_from_file('svcnet_hr_davis_sematic_1_config.yml')
    config.freeze()

    model = MaskPyramids(config)
    model = model.cuda()
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = Davis(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=config.TEST.NUM_SAMPLES,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TEST.MULTI_SCALE,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=None)

    checkpoint = torch.load('checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    check_path(savepath)
    check_path(savepath2)

    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    model.eval()
    with torch.no_grad():
        for i_iter, batch in enumerate(val_loader):
            images, labels, instances, [h, w], _ = batch
            images = images.cuda()
            targets = {'label': labels.long().cuda()}
            out = model(images, targets)['sematic_out']
            confusion_matrix[...] += get_confusion_matrix(
                labels,
                out,
                labels.size(),
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL
            )
            for img_t, pred_t, name in zip(images, out, batch[4]):
                img = img_t.permute(1,2,0).cpu().numpy()
                img = (img - img.min())/(img.max() - img.min())
                img_mark = img.copy()
                mark = pred_t.max(0)[1].cpu().numpy()
                img_mark[mark>0] += (1,1,0)
                img_mark /= 2
                # save annotions
                savefolder = os.path.join(savepath, name.split('/')[-2])
                check_path(savefolder)
                savename = os.path.join(savepath, name.split('/')[-2], name.split('/')[-1].split('.')[0] + '.png')
                print(savename)
                binary_mark = img.copy()
                binary_mark[:, :, 0] = mark
                binary_mark[:, :, 1] = mark
                binary_mark[:, :, 2] = mark
                binary_mark = cv2.resize(binary_mark, (int(w[0].numpy()), int(h[0].numpy())), interpolation=cv2.INTER_NEAREST) # w[0].numpy(), h[0].numpy()
                plt.imsave(savename, binary_mark)
                # save img_mark
                savefolder = os.path.join(savepath2, name.split('/')[-2])
                check_path(savefolder)
                savename = os.path.join(savepath2, name.split('/')[-2], name.split('/')[-1].split('.')[0] + '.jpg')
                print(savename)
                img_mark = cv2.resize(img_mark, (int(w[0].numpy()), int(h[0].numpy())), interpolation=cv2.INTER_NEAREST)
                plt.imsave(savename, img_mark)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IoU = IoU_array.mean()
        print('mean IOU: ', mean_IoU)
        
if __name__ == "__main__":
    
    savepath = './data/svcnet/Annotations'
    savepath2 = './data/svcnet/ImageMarks'
    main(savepath, savepath2)
    