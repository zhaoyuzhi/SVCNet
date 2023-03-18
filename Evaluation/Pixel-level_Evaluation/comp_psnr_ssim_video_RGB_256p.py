import argparse
import os
import cv2
import numpy as np

import skimage.measure as measure

# ----------------------------------------
#               Read image
# ----------------------------------------
# Find the tail of generated images (e.g., .jpg or .png)
def get_files_specific_tail(path):
    '''
    tail = get_files_specific_tail(opt.generated_root)
    tail = '.' + tail[0].split('__')[-1]
    '''
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if '00000' in filespath:
                method = root.split('\\')[-3]
                tail = filespath.split('.')[-1]
                method_with_tail = method + '__' + tail     # double __
                if method_with_tail not in ret:
                    ret.append(method_with_tail)
    return ret

# Read images: srcpath is the generated image; dstpath is the ground truth image
def read_DAVIS_videvo(srcpath, dstpath):
    # read images
    scr = cv2.imread(srcpath)
    dst = cv2.imread(dstpath)
    scr = cv2.resize(scr, (448, 256))
    dst = cv2.resize(dst, (448, 256))
    # convert grayscale to 3 channels
    if len(scr.shape) == 2:
        scr = scr[:, :, np.newaxis]
        scr = np.concatenate((scr, scr, scr), axis = 2)
    # DVP-based and FAVC will crop images, so we need to crop ground truth...
    scr_h = scr.shape[0]
    scr_w = scr.shape[1]
    dst = dst[:scr_h, :scr_w, :]
    return scr, dst

def define_dataset(opt):
    # Inference for color scribbles
    #imglist = text_readlines(opt.tag + '_test_imagelist.txt')
    imglist = text_readlines(opt.tag + '_test_imagelist_without_first_frame.txt')
    classlist = text_readlines(opt.tag + '_test_class.txt')
    imgroot = [list() for i in range(len(classlist))]

    for i, classname in enumerate(classlist):
        for j, imgname in enumerate(imglist):
            if imgname.split('/')[-2] == classname:
                imgroot[i].append(imgname)

    print('There are %d videos in the test set.' % (len(imgroot)))
    return imgroot

# ----------------------------------------
#                Evaluation
# ----------------------------------------
# Compute the mean-squared error between two images
def MSE(scr, dst):
    mse = measure.compare_mse(scr, dst)
    return mse

# Compute the normalized root mean-squared error (NRMSE) between two images
def NRMSE(scr, dst, mse_type = 'Euclidean'):
    nrmse = measure.compare_nrmse(scr, dst, norm_type = mse_type)
    return nrmse

# Compute the peak signal to noise ratio (PSNR) for an image
def PSNR(scr, dst):
    psnr = measure.compare_psnr(scr, dst)
    return psnr

# Compute the mean structural similarity index between two images
def SSIM(scr, dst, multichannel_input = True):
    ssim = measure.compare_ssim(scr, dst, multichannel = multichannel_input)
    return ssim

# ----------------------------------------
#              TXT processing
# ----------------------------------------
# read a txt expect EOF
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_root', type = str, \
        default = 'F:\\dataset,~task~related\\VCGAN~dataset\\test', \
            help = 'base_root')
    parser.add_argument('--generated_root', type = str, \
        default = 'F:\\submitted~papers\\SVCNet\\SVCNet~comparison_DAVIS_videvo\\CPNet', \
            help = 'generated_root')
    parser.add_argument('--tag', type = str, \
        default = 'DAVIS', \
            help = 'tag, DAVIS / videvo')
    opt = parser.parse_args()
    print(opt)
    
    opt.base_root = opt.base_root.replace('~', ' ')
    opt.generated_root = opt.generated_root.replace('~', ' ')
    
    # Define the dataset
    imgroot = define_dataset(opt)
    tail = get_files_specific_tail(opt.generated_root)
    tail = '.' + tail[0].split('__')[-1]
    method = opt.generated_root.split('\\')[-1]
    
    # Compute the metrics
    psnrratio = 0
    ssimratio = 0
    count = 0
    # Choose a category of dataset, it is fair for each dataset to be chosen
    for index in range(len(imgroot)):
        N = len(imgroot[index])
        for i in range(N):

            # Read images
            dstpath = os.path.join(opt.base_root, opt.tag, imgroot[index][i].split('/')[0], imgroot[index][i].split('/')[1])
            scrpath = os.path.join(opt.generated_root, opt.tag, imgroot[index][i].split('/')[0], imgroot[index][i].split('/')[1].replace('.jpg', tail))
            scr, dst = read_DAVIS_videvo(scrpath, dstpath)
            
            # Compute metrics for an image
            #mse = MSE(scr, dst)
            #nrmse = NRMSE(scr, dst, mse_type = 'Euclidean')
            psnr = PSNR(scr, dst)
            ssim = SSIM(scr, dst, multichannel_input = True)
            psnrratio = psnrratio + psnr
            ssimratio = ssimratio + ssim
            count = count + 1

    psnrratio = psnrratio / count
    ssimratio = ssimratio / count

    print('%s %s: psnr: %f, ssim: %f' % (opt.tag, method, psnrratio, ssimratio))
