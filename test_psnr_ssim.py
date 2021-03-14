import cv2
from sewar.full_ref import *

ori_path = 'Monarch.tif'
res_path = '6mse.tif'
# test
a1_image = cv2.imread(ori_path)
a2_image = cv2.imread(res_path)

print(psnr(a1_image, a2_image))
print(ssim(a1_image, a2_image))
