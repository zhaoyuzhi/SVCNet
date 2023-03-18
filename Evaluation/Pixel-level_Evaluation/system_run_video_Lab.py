import os

base_path = './SVCNet_2dataset_RGB'
processed_path = './SVCNet_comparison_DAVIS_videvo'
processed_method_list = ['3DVC', 'FAVC', 'VCGAN_second_stage', \
    'ChromaGAN', 'ChromaGAN+BTC', 'ChromaGAN+DEVC', 'ChromaGAN+DVP', \
    'CIC', 'CIC+BTC', 'CIC+DEVC', 'CIC+DVP', \
    'CPNet', 'CPNet+BTC', 'CPNet+DEVC', 'CPNet+DVP', \
    'IAC', 'IAC+BTC', 'IAC+DEVC', 'IAC+DVP', \
    'LTBC', 'LTBC+BTC', 'LTBC+DEVC', 'LTBC+DVP', \
    'RUIC', 'RUIC+BTC', 'RUIC+DEVC', 'RUIC+DVP', \
    'RUIC_without_scribble', 'RUIC_without_scribble+BTC', 'RUIC_without_scribble+DEVC', 'RUIC_without_scribble+DVP', \
    'SCGAN', 'SCGAN+BTC', 'SCGAN+DEVC', 'SCGAN+DVP']
tag_list = ['DAVIS', 'videvo']

for i in range(len(processed_method_list)):
    for j in range(len(tag_list)):
        method_path = os.path.join(processed_path, processed_method_list[i])
        tag = tag_list[j]
        order = 'python comp_psnr_ssim_video_Lab.py --base_root %s --generated_root %s --tag %s' % (base_path, method_path, tag)
        os.system(order)
