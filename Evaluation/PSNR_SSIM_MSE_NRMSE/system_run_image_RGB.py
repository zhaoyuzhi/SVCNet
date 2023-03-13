import os

base_path = 'F:\\dataset\\ILSVRC2012_processed\\ILSVRC2012_val_256'
processed_method_list = [
    './single_image_colorization~results_ImageNet\\LRAC\\ctest10k_256', \
    './single_image_colorization~results_ImageNet\\CIC', \
    './single_image_colorization~results_ImageNet\\LTBC', \
    './single_image_colorization~results_ImageNet\\Pix2Pix', \
    './single_image_colorization~results_ImageNet\\DeOldify', \
    './single_image_colorization~results_ImageNet\\FAVC\\result2', \
    './single_image_colorization~results_ImageNet\\ChromaGAN', \
    './single_image_colorization~results_ImageNet\\SCGAN', \
    './single_image_colorization~results_ImageNet\\IAC', \
    './single_image_colorization~results_ImageNet\\VCGAN', \
    './SVCNet_comparison_ImageNet_(40_color_scribbles)\\RUIC_(without_scribble)', \
    './SVCNet_comparison_ImageNet_(40_color_scribbles)\\CPNet_(without_scribble)', \
    './SVCNet_comparison_ImageNet_(40_color_scribbles)\\RUIC', \
    './SVCNet_comparison_ImageNet_(40_color_scribbles)\\CPNet'
]

for i in range(len(processed_method_list)):
    method_path = processed_method_list[i]
    order = 'python comp_psnr_ssim_image_RGB.py --base_root %s --generated_root %s' % (base_path, method_path)
    os.system(order)
