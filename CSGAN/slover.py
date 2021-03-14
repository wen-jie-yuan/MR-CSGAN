import torch
import pandas as pd
from CSGAN.CSGAN import *
from sewar.full_ref import *
from utils import progress_bar
import torch.backends.cudnn as cudnn
from torchvision.models.vgg import vgg19


# Feature extraction
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'3': 'conv1_1',
                  '9': 'conv2_2',
                  '18': 'conv3_4',
                  '26': 'poolfinal',
                  '28': 'conv5_4'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        if name == "transion_layer":
            x = layer(x)
        else:
            for name, layer in vgg._modules["resnet_layer"]._modules.items():
                x = layer(x)
                if name in layers:
                    features[layers[name]] = x
    return features


# The output of the last two layers of VGG is regarded as loss
class Net(torch.nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # Take the last two layers of the model
        self.transion_layer = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.resnet_layer = torch.nn.Sequential(*list(model.children())[1:])

    def forward(self, x):
        x = self.transion_layer(x)
        x = self.resnet_layer(x)
        return x


vgg = vgg19(pretrained=True).features
vgg = Net(vgg)
vgg = vgg.cuda()
# 空数组存放最大psnr
a = []
results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}


class CSGAN_Trainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(CSGAN_Trainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.netG = None
        self.netD = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.epoch_pretrain = 30
        self.sampling_rate = config.samplingRate
        self.sampling_point = config.samplingPoint
        self.output_path = './epochs'
        self.criterionG = None
        self.criterionD = None
        self.optimizerG = None
        self.optimizerD = None
        self.feature_extractor = None
        self.scheduler = None
        self.seed = config.seed
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.criterionG = torch.nn.L1Loss()
        self.criterionD = torch.nn.BCELoss()
        self.criterionMSE = torch.nn.MSELoss()

    def build_model(self):
        self.netG = Generator(num_channels=1, base_filter=self.sampling_point).to(
            self.device)
        self.netD = Discriminator(base_filter=64, num_channel=1).to(self.device)
        self.netG.weight_init(mean=0.0, std=0.2)
        self.netD.weight_init(mean=0.0, std=0.2)
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterionG.cuda()
            self.criterionD.cuda()

        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizerD = torch.optim.SGD(self.netD.parameters(), lr=self.lr / 10, momentum=0.9, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizerG,
                                                              milestones=[300, 600, 900, 1200, 1500, 1800, 2100, 2400,
                                                                          2700],
                                                              # [300, 600, 900, 1400, 2000, 2500, 3500]

                                                              gamma=0.5)  # lr decay
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizerD,
                                                              milestones=[300, 600, 900, 1200, 1500, 1800, 2100, 2400,
                                                                          2700],
                                                              # [300, 600, 900, 1400, 2000, 2500, 3500]

                                                              gamma=0.5)  # lr decay

    @staticmethod
    def to_data(x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def pretrain(self):
        self.netG.train()
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.netG.zero_grad()
            loss = self.criterionG(self.netG(data), data)
            loss.backward()
            self.optimizerG.step()

    def train(self):
        # models setup
        self.netG.train()
        self.netD.train()
        g_train_loss = 0
        d_train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            # setup noise
            real_label = torch.ones(data.size(0), data.size(1)).to(self.device)
            fake_label = torch.zeros(data.size(0), data.size(1)).to(self.device)
            data, target = data.to(self.device), target.to(self.device)

            # Train Discriminator
            self.optimizerD.zero_grad()
            d_real = self.netD(data)
            d_real_loss = self.criterionD(d_real, real_label)

            d_fake = self.netD(self.netG(data))
            d_fake_loss = self.criterionD(d_fake, fake_label)
            d_total = d_real_loss + d_fake_loss
            d_train_loss += d_total.item()
            d_total.backward()
            self.optimizerD.step()

            # Train generator
            self.optimizerG.zero_grad()
            g_real = self.netG(data)
            g_fake = self.netD(g_real)
            gan_loss = self.criterionD(g_fake, real_label)
            mse_loss = self.criterionG(g_real, data)
            g_real1 = get_features(g_real, vgg)
            data1 = get_features(data, vgg)
            vgg19_loss = self.criterionMSE(g_real1['conv5_4'], data1['conv5_4'])
            g_total = mse_loss + 0.05 * vgg19_loss + 1e-3 * gan_loss

            g_train_loss += g_total.item()
            g_total.backward()
            self.optimizerG.step()

            progress_bar(batch_num, len(self.training_loader), 'G_Loss: %.4f | D_Loss: %.4f' % (
                g_train_loss / (batch_num + 1), d_train_loss / (batch_num + 1)))

        print("    Average G_Loss: {:.4f}".format(g_train_loss / len(self.training_loader)))
        return format(g_train_loss / (batch_num + 1)), format(d_train_loss / (batch_num + 1))

    def test(self):
        self.netG.eval()
        sum_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.netG(data)
                mse1 = self.criterionMSE(prediction, data)
                psnr1 = 10 * log10(1 / mse1.item())
                sum_psnr += psnr1
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (sum_psnr / (batch_num + 1)))
        avg_psnr = sum_psnr / len(self.testing_loader)
        return avg_psnr

    def run(self):
        self.build_model()
        for epoch in range(1, self.epoch_pretrain + 1):
            self.pretrain()
            print("{}/{} pretrained".format(epoch, self.epoch_pretrain))

        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.train()
            results['d_loss'].append(loss[0])
            results['g_loss'].append(loss[1])
            avg_psnr = self.test()
            results['psnr'].append(avg_psnr)
            self.scheduler.step(epoch)
            model_out_path = self.output_path + "/model_path" + str(epoch) + ".pth"
            torch.save(self.netG, model_out_path)
            a.append(str(avg_psnr))

        max_index = 0
        list_index = 0
        for num in a:
            if num > a[max_index]:
                max_index = list_index
            list_index += 1
        print("max_avg_psnr: Epoch " + str(max_index + 1) + " , " + str(a[max_index]))
        out_path = 'data_results/'
        data_frame = pd.DataFrame(
            data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'PSNR': results['psnr']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'mscsgan_' + 'train_results_' + str(self.sampling_rate) + '.csv',
                          index_label='Epoch')
