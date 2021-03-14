# 生成验证图像
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from sewar.full_ref import *
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor, ToPILImage

model_path = './model_best/MSE_model/mse_10.pth'  # MSE_model/Perceptual_loss_model
# model_path = './model_best/MSE_model/mse_4.pth'
# model_path = './model_best/Perceptual_loss_model/ps_4.pth'
ori_path = './barbara.tif'
res_path = './test1.png'
# 参数配置
parser = argparse.ArgumentParser(description='Compressed sensing with NN')
parser.add_argument('--model', type=str, default=model_path, help='model_path')
parser.add_argument('--input', type=str, required=False, default=ori_path, help='test_img')
parser.add_argument('--output', type=str, default=res_path, help='save_path')
args = parser.parse_args()
print(args)

# 单通道
with torch.no_grad():
    GPU_IN_USE = torch.cuda.is_available()
    img = Image.open(ori_path)

    device = torch.device('cuda')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = model.to(device)
    data = (ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
    data = data.to(device)
    if GPU_IN_USE:
        cudnn.benchmark = True
    out = model(data)
    out_img = ToPILImage()(out[0].data.cpu())
    # out_img *= 255.0
    # out_img = out_img.clip(0, 255)
    # out_img = Image.fromarray(np.uint8(out_img[0]), mode='L')
    out_img.save(args.output)
    a1_image = cv2.imread(args.input)
    a2_image = cv2.imread(args.output)
    print(psnr(a1_image, a2_image))
    print(ssim(a1_image, a2_image))

#
# # 三通道图片
# # 输入图片配置
# with torch.no_grad():
#     GPU_IN_USE = torch.cuda.is_available()
#     img = Image.open(args.input).convert('RGB')
#     R, G, B = img.split()
#     if GPU_IN_USE:
#         cudnn.benchmark = True
#     # 模型输入设置
#     device = torch.device('cpu')
#     model = torch.load(args.model, map_location=lambda storage, loc: storage)
#     model = model.to(device)
#
#     data_R = (ToTensor()(R)).view(1, -1, R.size[1], R.size[0])
#     data_R = data_R.to(device)
#     out_R = model(data_R)
#     out_R = out_R.cpu()
#     out_img_R = out_R.data[0].numpy()
#     out_img_R *= 255.0
#     out_img_R = out_img_R.clip(0, 255)
#     out_img_R = Image.fromarray(np.uint8(out_img_R[0]), mode='L')
#
#     data_G = (ToTensor()(G)).view(1, -1, G.size[1], G.size[0])
#     data_G = data_G.to(device)
#     out_G = model(data_G)
#     out_G = out_G.cpu()
#     out_img_G = out_G.data[0].numpy()
#     out_img_G *= 255.0
#     out_img_G = out_img_G.clip(0, 255)
#     out_img_G = Image.fromarray(np.uint8(out_img_G[0]), mode='L')
#
#     data_B = (ToTensor()(B)).view(1, -1, B.size[1], B.size[0])
#     data_B = data_B.to(device)
#     out_B = model(data_B)
#     out_B = out_B.cpu()
#     out_img_B = out_B.data[0].numpy()
#     out_img_B *= 255.0
#     out_img_B = out_img_B.clip(0, 255)
#     out_img_B = Image.fromarray(np.uint8(out_img_B[0]), mode='L')
#
#     # 输出和保存
#     out_img = Image.merge('RGB', [out_img_R, out_img_G, out_img_B]).convert('RGB').convert("RGBA")
#     out_img.save(args.output)
#
#     # test
#     a1_image = cv2.imread(ori_path)
#     a2_image = cv2.imread(res_path)
#     a1 = cv2.cvtColor(a1_image, cv2.COLOR_BGR2GRAY)
#     a2 = cv2.cvtColor(a2_image, cv2.COLOR_BGR2GRAY)
#     print('psnr:' + str(psnr(a1, a2, 255)))
#     print('ssim:' + str(ssim(a1, a2)))
