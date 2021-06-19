import pandas as pd
from MRCSGAN.CSGAN import *
from sewar.full_ref import *
from utils.utils import progress_bar
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.models.vgg import vgg19, vgg16

vgg = vgg19(pretrained=True)
loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
loss_network.cuda()
for param in loss_network.parameters():
    param.requires_grad = False

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
        self.epoch_pretrain = 0
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
        self.batchsize = config.batchSize
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
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizerD = torch.optim.SGD(self.netD.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, step_size=400, gamma=0.5)
        # self.schedulerD = torch.optim.lr_scheduler.StepLR(self.optimizerD, step_size=400, gamma=0.5)

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
        # print('generator parameters:', sum(param.numel() for param in self.netG.parameters()))
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

            d_fake = self.netD(self.netG(data).detach())
            d_fake_loss = self.criterionD(d_fake, fake_label)
            d_total = d_real_loss + d_fake_loss
            d_train_loss += d_total.item()
            d_total.backward()
            self.optimizerD.step()

            # Train generator
            self.optimizerG.zero_grad()
            g_real = self.netG(data)
            g_fake = self.netD(g_real).mean()
            gan_loss = torch.mean(1 - g_fake)

            mse_loss = self.criterionG(g_real, data)
            g_real = torch.cat([g_real, g_real, g_real], dim=1, out=None)
            data = torch.cat([data, data, data], dim=1, out=None)

            vgg19_loss = self.criterionMSE(loss_network(g_real), loss_network(data))
            g_total = mse_loss + 0.006 * vgg19_loss + 1e-3 * gan_loss

            g_train_loss += g_total.item()

            g_total.backward()
            self.optimizerG.step()
            progress_bar(batch_num, len(self.training_loader),
                         'G_Loss: %.4f | D_Loss: %.4f | vgg19_loss: %4f |gan_loss: %4f' % (
                             g_train_loss / (batch_num + 1), d_train_loss / (batch_num + 1), 0.006 * vgg19_loss,
                             1e-3 * gan_loss))
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
            results['g_loss'].append(loss[0])
            results['d_loss'].append(loss[1])
            avg_psnr = self.test()
            results['psnr'].append(avg_psnr)
            self.schedulerG.step()
            # self.schedulerD.step()
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
            data={'Loss_G': results['g_loss'], 'Loss_D': results['d_loss'], 'PSNR': results['psnr']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'MR-CSGAN_' + 'train_results_' + str(self.sampling_rate) + '.csv',
                          index_label='Epoch')
