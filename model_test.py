from math import log10
import torch
import torchvision
from torch.utils.data import DataLoader
from utils import ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './model_best/Perceptual_loss_model/ps_1.pth'    # MSE_model/Perceptual_loss_model
ori_img_path = './dataset/test_img/Set5'
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])
test_dataset = torchvision.datasets.ImageFolder(ori_img_path, transform=transforms_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
sum_psnr = 0
sum_ssim = 0
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)
        mse = torch.nn.MSELoss()(prediction, data)
        psnr1 = 10 * log10(1 / mse.item())
        ssim1 = ssim(prediction, data)
        sum_psnr += psnr1
        sum_ssim += ssim1
        # print('PSNR: %.4f' % (psnr1))
        # print('SSIM: %4f' % (ssim1))


print("Average PSNR: {:.4f} dB".format(sum_psnr / len(test_loader)))
print("Average SSIM: {:.4f} dB".format(sum_ssim / len(test_loader)))

ori_img_path = './dataset/test_img/Set11'
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])
test_dataset = torchvision.datasets.ImageFolder(ori_img_path, transform=transforms_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
sum_psnr = 0
sum_ssim = 0
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)
        mse = torch.nn.MSELoss()(prediction, data)
        psnr1 = 10 * log10(1 / mse.item())
        ssim1 = ssim(prediction, data)
        sum_psnr += psnr1
        sum_ssim += ssim1
        # print('PSNR: %.4f' % (psnr1))
        # print('SSIM: %4f' % (ssim1))

print("Average PSNR: {:.4f} dB".format(sum_psnr / len(test_loader)))
print("Average SSIM: {:.4f} dB".format(sum_ssim / len(test_loader)))

ori_img_path = './dataset/test_img/Set14'
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])
test_dataset = torchvision.datasets.ImageFolder(ori_img_path, transform=transforms_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
sum_psnr = 0
sum_ssim = 0
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)
        mse = torch.nn.MSELoss()(prediction, data)
        psnr1 = 10 * log10(1 / mse.item())
        ssim1 = ssim(prediction, data)
        sum_psnr += psnr1
        sum_ssim += ssim1
        # print('PSNR: %.4f' % (psnr1))
        # print('SSIM: %4f' % (ssim1))

print("Average PSNR: {:.4f} dB".format(sum_psnr / len(test_loader)))
print("Average SSIM: {:.4f} dB".format(sum_ssim / len(test_loader)))

ori_img_path = './dataset/test_img/bsd100'
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])
test_dataset = torchvision.datasets.ImageFolder(ori_img_path, transform=transforms_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
sum_psnr = 0
sum_ssim = 0
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)
        mse = torch.nn.MSELoss()(prediction, data)
        psnr1 = 10 * log10(1 / mse.item())
        ssim1 = ssim(prediction, data)
        sum_psnr += psnr1
        sum_ssim += ssim1
        # print('PSNR: %.4f' % (psnr1))
        # print('SSIM: %4f' % (ssim1))

print("Average PSNR: {:.4f} dB".format(sum_psnr / len(test_loader)))
print("Average SSIM: {:.4f} dB".format(sum_ssim / len(test_loader)))
