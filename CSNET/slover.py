import torch
import pandas as pd
from math import log10
from CSNET.CSNet import CSNET
from utils.utility import progress_bar
import torch.backends.cudnn as cudnn

results = {'loss': [], 'psnr': []}


class CSNET_Trainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(CSNET_Trainer, self).__init__()
        self.criterion = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.output_path = './epochs'
        self.nEpochs = config.nEpochs
        self.sampling_rate = config.samplingRate
        self.sampling_point = config.samplingPoint
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.model = CSNET(num_channels=1, base_filter=self.sampling_point).to(self.device)
        self.model.weight_init(mean=0.0, std=0.02)
        torch.manual_seed(self.seed)
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #                                                       milestones=[1000], gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterionMSE(self.model(data), data)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))
        return format(train_loss / len(self.training_loader))

    def test(self):
        self.model.eval()
        avg_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterionMSE(prediction, data)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
        return format(avg_psnr / len(self.testing_loader))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.train()
            results['loss'].append(loss)
            avg_psnr = self.test()
            results['psnr'].append(avg_psnr)
            self.scheduler.step()
            model_out_path = self.output_path + "/model_path" + str(epoch) + ".pth"
            torch.save(self.model, model_out_path)
        out_path = 'data_results/'
        data_frame = pd.DataFrame(
            data={'Loss': results['loss'], 'PSNR': results['psnr']},
            index=range(1, epoch + 1))
        data_frame.to_csv(out_path + 'MR_CSNet_' + 'train_results_' + str(self.sampling_rate) + '.csv',
                          index_label='Epoch')
