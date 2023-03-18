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
    # read a folder, return the complete path
    for root, dirs, files in os.walk(path):
        for filespath in files:
            tail = filespath.split('.')[-1]
            break
    return tail

# Read images: srcpath is the generated image; dstpath is the ground truth image
def read_image(srcpath, dstpath):
    # read images
    scr = cv2.imread(srcpath)
    dst = cv2.imread(dstpath)
    # convert grayscale to 3 channels
    if len(scr.shape) == 2:
        scr = scr[:, :, np.newaxis]
        scr = np.concatenate((scr, scr, scr), axis = 2)
    # assert
    assert scr.shape == dst.shape
    return scr, dst

def BGR2LAB(scr):
    scr = cv2.cvtColor(scr, cv2.COLOR_BGR2Lab)
    scr = np.concatenate((scr[:, :, [1]], scr[:, :, [2]]), axis = 2)
    return scr

def define_dataset():
    imglist = text_readlines('ctest10k.txt')
    #print('There are %d images in the test set.' % (len(imglist)))
    return imglist

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
'''
def PSNR(scr, dst, pixel_max_cnt = 255):
    mse = np.multiply(scr - dst, scr - dst)
    rmse_avg = np.mean(mse) ** 0.5
    psnr = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return psnr
'''
def PSNR(scr, dst):
    psnr = measure.compare_psnr(scr, dst)
    return psnr

# Compute the mean structural similarity index between two images
def SSIM(scr, dst, multichannel_input = True):
    scr = scr / 255.0
    dst = dst / 255.0
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
        content[i] = content[i][:len(content[i])-1]
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
        default = 'F:\\dataset\\ILSVRC2012_processed\\ILSVRC2012_val_256', \
            help = 'base_root')
    parser.add_argument('--generated_root', type = str, \
        #default = 'F:\\submitted papers\\SVCNet\\SVCNet comparison_ImageNet (40 color scribbles)\\CPNet (without scribble)', \
        default = 'E:\\research accepted papers\\SCGAN, TCSVT 2020\\single image colorization results_ImageNet\\RUIC (Zhang et al.)', \
            help = 'generated_root')
    opt = parser.parse_args()
    #print(opt)

    opt.generated_root = opt.generated_root.replace('~', ' ')
    
    # Define the dataset
    imgroot = define_dataset()
    tail = get_files_specific_tail(opt.generated_root)
    tail = '.' + tail
    method = opt.generated_root.split('\\')[-1]
    
    # Compute the metrics
    psnrratio = 0
    ssimratio = 0
    count = 0
    # Choose a category of dataset, it is fair for each dataset to be chosen
    for index in range(len(imgroot)):
        imgname = imgroot[index]
        # Read images
        dstpath = os.path.join(opt.base_root, imgname)
        scrpath = os.path.join(opt.generated_root, imgname.replace('.JPEG', tail))
        scr, dst = read_image(scrpath, dstpath)
        scr = BGR2LAB(scr)
        dst = BGR2LAB(dst)
        
        # Compute metrics for an image
        #mse = MSE(scr, dst)
        #nrmse = NRMSE(scr, dst, mse_type = 'Euclidean')
        #psnr = PSNR(scr, dst, pixel_max_cnt = 255)
        psnr = PSNR(scr, dst)
        ssim = SSIM(scr, dst, multichannel_input = True)
        psnrratio = psnrratio + psnr
        ssimratio = ssimratio + ssim
        count = count + 1

    psnrratio = psnrratio / count
    ssimratio = ssimratio / count

    print('%s: psnr: %f, ssim: %f' % (method, psnrratio, ssimratio))
