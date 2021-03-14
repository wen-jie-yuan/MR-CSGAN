import os
import cv2
import torch
from PIL import Image
from sewar.full_ref import *
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor

ori_img_path = './dataset/test_img/Set14/A/'
res_img_path = './res_img/Set14/'
model_path = './model_best/Perceptual_loss_model/ps_25.pth'
device = torch.device('cuda')
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)
for filename in os.listdir(ori_img_path):
    img = Image.open(ori_img_path + filename)
    img1 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    data = (ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
    data = data.to(device)
    cudnn.benchmark = True
    out = model(data)
    out = out.cpu()
    out_img = out.data[0].numpy()
    out_img *= 255.0
    out_img = out_img.clip(0, 255)
    out_img = Image.fromarray(np.uint8(out_img[0]), mode='L')
    out_img.save(res_img_path + filename)
    a1_image = cv2.imread(ori_img_path + filename)
    a2_image = cv2.imread(res_img_path + filename)
    print(psnr(a1_image, a2_image))
    print(ssim(a1_image, a2_image))
