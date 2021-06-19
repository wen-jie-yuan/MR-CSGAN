import argparse
import torchvision
from torch.utils.data import DataLoader
from CSNET.slover import CSNET_Trainer
from MRCSGAN.slover import CSGAN_Trainer

# set Training parameter
parser = argparse.ArgumentParser(description='Compressed sensing with NN')

# Set super parameters
parser.add_argument('--batchSize', type=int, default=64, help='Small batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='Test batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='Iterations')
parser.add_argument('--imageSize', type=int, default=64, metavar='N')
parser.add_argument('--samplingRate', type=int, default=10, help='sampling_rate = 1/4/10/25')
parser.add_argument('--resBlock', type=int, default=8, help='res blocks')
parser.add_argument('--samplingPoint', type=int, default=102, help='1% - 10 4% - 41 10% - 103 25% - 256')
parser.add_argument('--trainPath', default='./dataset/B400C/')
parser.add_argument('--valPath', default='./dataset/test_images/set11/')
parser.add_argument('--lr', type=float, default=0.0004, help='Learning Rate. Default=0.001')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--cuda', action='store_true', default=True)

args = parser.parse_args()
kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(args.imageSize),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.RandomHorizontalFlip(p=0.3),
    torchvision.transforms.ToTensor(),
])

transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(args.trainPath, transform=transforms)
val_dataset = torchvision.datasets.ImageFolder(args.valPath, transform=transforms_test)
train_loader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.testBatchSize, shuffle=False)

# saved_model select
# model = CSNET_Trainer(args, train_loader, val_loader)
model = CSGAN_Trainer(args, train_loader, val_loader)

model.run()
