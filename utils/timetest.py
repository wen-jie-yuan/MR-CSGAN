import torch
import torchvision
from torch.utils.data import DataLoader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './model_best/Perceptual_loss_model/ps_10.pth'  # MSE_model/Perceptual_loss_model
ori_img_path = './dataset/test_img/bsd100'
model = torch.load(model_path, map_location=lambda storage, loc: storage)
model = model.to(device)

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize(size=(256, 256)),
    torchvision.transforms.ToTensor(),
])
test_dataset = torchvision.datasets.ImageFolder(ori_img_path, transform=transforms_test)
test_loader = DataLoader(test_dataset, batch_size=11, shuffle=False)


strat = time.time()
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)

end = time.time()
print(end - strat)
