import os

base_path = './data/davis/Annotations/480p'
processed_path = '/media/zyz/Elements/submitted papers/SVCNet/SVCNet comparison_segmentations'
processed_method_list = ['3DVC', 'FAVC', 'VCGAN_second_stage', \
    'ChromaGAN', 'ChromaGAN+BTC', 'ChromaGAN+DEVC', 'ChromaGAN+DVP', \
    'CIC', 'CIC+BTC', 'CIC+DEVC', 'CIC+DVP', \
    'CPNet', 'CPNet+BTC', 'CPNet+DEVC', 'CPNet+DVP', \
    'IAC', 'IAC+BTC', 'IAC+DEVC', 'IAC+DVP', \
    'LTBC', 'LTBC+BTC', 'LTBC+DEVC', 'LTBC+DVP', \
    'RUIC', 'RUIC+BTC', 'RUIC+DEVC', 'RUIC+DVP', \
    'RUIC_without_scribble', 'RUIC_without_scribble+BTC', 'RUIC_without_scribble+DEVC', 'RUIC_without_scribble+DVP', \
    'SCGAN', 'SCGAN+BTC', 'SCGAN+DEVC', 'SCGAN+DVP']

for i in range(len(processed_method_list)):
    method_path = os.path.join(processed_path, processed_method_list[i], 'DAVIS')
    order = 'python svcnet_compute_miou.py --gt_path %s --pred_path %s' % (base_path, method_path)
    os.system(order)
