import torchvision
from torch.utils.data import DataLoader

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(64),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder('./dataset/train/', transform=transforms)
train_loader = DataLoader(train_dataset, batch_size=32)

for batch_num, (data, target) in enumerate(train_loader):
    print(data.size(0))
    print(data.size(1))
