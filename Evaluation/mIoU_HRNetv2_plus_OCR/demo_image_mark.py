import os
import cv2
import numpy as np

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if '.png' in filespath:
                ret.append(os.path.join(root, filespath))
    return ret

if __name__ == "__main__":
    
    '''
    binary_mark = cv2.imread('./data/davis/Annotations/480p/blackswan/00000.png').astype(np.float64)
    img = cv2.imread('./data/davis/JPEGImages/480p/blackswan/00000.jpg').astype(np.float64)

    binary_mark[:, :, 0] *= 0
    img[binary_mark>0] += 255
    img /= 2

    cv2.imwrite('ttt.jpg', img.astype(np.uint8))
    '''

    def get_save_path(imgpath):
        save_name = imgpath.replace('Annotations', 'ImageMarks').replace('.png', '.jpg')
        last_len = save_name.split('\\')[-1]
        save_folder = save_name[:-len(last_len)]
        return save_folder, save_name

    readpath1 = 'data/davis/Annotations'
    readpath2 = 'data/davis/JPEGImages'
    savepath = 'data/davis/ImageMarks'
    readpath_imglist = get_files(readpath1)
    
    for i in range(len(readpath_imglist)):
        imgpath = readpath_imglist[i]
        print(i, len(readpath_imglist), imgpath)

        binary_mark = cv2.imread(imgpath).astype(np.float64)
        img = cv2.imread(imgpath.replace('Annotations', 'JPEGImages').replace('.png', '.jpg')).astype(np.float64)

        save_folder, save_name = get_save_path(imgpath)
        check_path(save_folder)
        
        binary_mark[:, :, 0] *= 0
        img[binary_mark>0] += 255
        img /= 2

        cv2.imwrite(save_name, img.astype(np.uint8))
