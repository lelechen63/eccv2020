import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
# np.set_printoptions(suppress=True)
from collections import OrderedDict
import argparse

from data.a2l_dataset import  GRID_raw_pca_landmark, GRID_raw_pca_3dlandmark

from models.networks import  A2L

from torch.nn import init
from utils import util
def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict

def initialize_weights( net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

class Trainer():
    def __init__(self, config):
        self.generator = A2L()
        
        self.l1_loss_fn =  nn.L1Loss()
        self.mse_loss_fn = nn.MSELoss()
        self.config = config

        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.generator     = nn.DataParallel(self.generator, device_ids=device_ids).cuda()
            self.mse_loss_fn   = self.mse_loss_fn.cuda(config.cuda1)
            self.l1_loss_fn = self.l1_loss_fn.cuda(config.cuda1)

        initialize_weights(self.generator)
        self.start_epoch = 0
       
        self.opt_g = torch.optim.Adam( self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        if config.threeD:
            self.train_dataset = GRID_raw_pca_3dlandmark( train=config.is_train)

            self.test_dataset = GRID_raw_pca_3dlandmark( train= 'test')
        
        else:
            self.train_dataset = GRID_raw_pca_landmark( train=config.is_train)

            self.test_dataset = GRID_raw_pca_landmark( train= 'test')
        

        self.data_loader = DataLoader(self.train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=config.batch_size,
                                      num_workers= config.num_thread,
                                      shuffle=True, drop_last=True)
    def fit(self):
        config = self.config
    
        num_steps_per_epoch = len(self.data_loader)
        cc = 0
        t0 = time.time()
        xLim=(0.0, 256.0)
        yLim=(0.0, 256.0)
        zLim=(-128, 128.0)
        xLab = 'x'
        yLab = 'y'
        
        if config.threeD:
            mean =  np.load('./basics/mean_grid_front_3d.npy')
            component = np.load('./basics/U_grid_front_3d.npy')
        else:
            mean =  np.load('./basics/mean_grid_front.npy')
            component = np.load('./basics/U_grid_front.npy')
        if config.load_model:
            self.generator.load_state_dict(torch.load(config.model_name))
            print ('load pretrained [{}]'.format(config.model_name))
        self.generator.train()
        for epoch in range(self.start_epoch, config.max_epochs):
            Flage = True
            if (epoch + 1) % 10==0:
                Flage = False
                self.generator.eval()
                with torch.no_grad():
                    for step,  (example_landmark, lmark, audio, path) in enumerate(self.test_loader):
                        lmark  = Variable(lmark.float()).cuda()
                        audio = Variable(audio.float()).cuda()
                        example_landmark = Variable(example_landmark.float()).cuda()
                        fake_lmark, _ = self.generator( example_landmark, audio)
                        loss =  self.mse_loss_fn(fake_lmark , lmark) 
                        print ('===========================')
                        print (fake_lmark[0,2:6])
                        print ('----------------------')
                        print (lmark[0,2:6])
                        print("[{}/{}][{}/{}]   loss1: {:.8f}".format(epoch+1, config.max_epochs, step+1, num_steps_per_epoch, loss))
                        
                        if (epoch + 1) % 50 ==0:
                            lmark = lmark.data.cpu().numpy()
                            fake_lmark = fake_lmark.data.cpu().numpy()
                            lmark = np.dot(lmark,component) + mean
                            fake_lmark = np.dot(fake_lmark,component) + mean
                            if config.threeD:
                                lmark = lmark.reshape(config.batch_size , 68 * 3)
                                fake_lmark = fake_lmark.reshape(config.batch_size , 68 * 3)
                            else:
                                lmark = lmark.reshape(config.batch_size , 68 * 2)
                                fake_lmark = fake_lmark.reshape(config.batch_size , 68 * 2)
                            for indx in range( min (config.batch_size , 64 )):
                                real_name = "{}test_real_{}_{}.png".format(config.sample_dir,cc, indx)
                                fake_name = "{}test_fake_{}_{}.png".format(config.sample_dir,cc, indx)
                                if config.threeD:
                                    util.plot_flmarks3D(lmark[indx], real_name, xLim, yLim, zLim, figsize=(10, 10), sentence = path[indx])
                                    util.plot_flmarks3D(fake_lmark[indx], fake_name, xLim, yLim, zLim, figsize=(10, 10), sentence = path[indx])
                                else:
                                    util.plot_flmarks(lmark[indx], real_name, xLim, yLim, xLab, yLab, figsize=(10, 10), sentence = path[indx])
                                    util.plot_flmarks(fake_lmark[indx], fake_name, xLim, yLim, xLab, yLab, figsize=(10, 10), sentence = path[indx])
                        if step == 3:
                            break
            if Flage == False:
                self.generator.train()
            for step, (example_landmark, lmark, audio , path) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    lmark    = Variable(lmark.float()).cuda()
                    audio = Variable(audio.float()).cuda()
                    example_landmark = Variable(example_landmark.float()).cuda()
                else:
                    lmark    = Variable(lmark.float())
                    audio = Variable(audio.float())
                    example_landmark = Variable(example_landmark.float())

                fake_lmark , _ = self.generator( example_landmark, audio)
               
                loss =  self.mse_loss_fn(fake_lmark , lmark) 
                loss.backward() 
                self.opt_g.step()
                self._reset_gradients()
                if (step+1) % 10 == 0 or (step+1) == num_steps_per_epoch:
                    print("[{}/{}][{}/{}]   loss1: {:.8f},data time: {:.4f},  model time: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss,  t1-t0,  time.time() - t1))
                # if (step) % (int(num_steps_per_epoch  / 2 )) == 0 and step != 0:
                t0 = time.time()
            if epoch  % 20 == 0:
                lmark = lmark.data.cpu().numpy()
                fake_lmark = fake_lmark.data.cpu().numpy()
                lmark = np.dot(lmark,component) + mean
                fake_lmark = np.dot(fake_lmark,component) + mean
                if config.threeD:
                    lmark = lmark.reshape(config.batch_size , 68 * 3)
                    fake_lmark = fake_lmark.reshape(config.batch_size , 68 * 3)
                else:
                    lmark = lmark.reshape(config.batch_size , 68 * 2)
                    fake_lmark = fake_lmark.reshape(config.batch_size , 68 * 2)
                for indx in range( min (config.batch_size , 64 )):
                    real_name = "{}test_real_{}_{}.png".format(config.sample_dir,cc, indx)
                    fake_name = "{}test_fake_{}_{}.png".format(config.sample_dir,cc, indx)
                    if config.threeD:
                        util.plot_flmarks3D(lmark[indx], real_name, xLim, yLim, zLim, figsize=(10, 10), sentence = path[indx])
                        util.plot_flmarks3D(fake_lmark[indx], fake_name, xLim, yLim, zLim, figsize=(10, 10), sentence = path[indx])
                    else:
                        util.plot_flmarks(lmark[indx], real_name, xLim, yLim, xLab, yLab, figsize=(10, 10), sentence = path[indx])
                        util.plot_flmarks(fake_lmark[indx], fake_name, xLim, yLim, xLab, yLab, figsize=(10, 10), sentence = path[indx])
                torch.save(self.generator.state_dict(),
                            "{}/atnet_lstm_{}.pth"
                            .format(config.model_dir,cc))
                        
                cc += 1
            
    def _reset_gradients(self):
        self.generator.zero_grad()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",
                        type=float,
                        default=0.00002)
                        
    parser.add_argument("--beta1",
                        type=float,
                        default=0.5)
    parser.add_argument("--beta2",
                        type=float,
                        default=0.999)
    parser.add_argument("--lambda1",
                        type=int,
                        default=100)
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=4)
    parser.add_argument("--max_epochs",
                        type=int,
                        default=1000000)
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        # default="../dataset/")
                        default="/mnt/ssd0/dat/lchen63/grid/pickle/")
                        # default = '/media/lele/DATA/lrw/data2/pickle')
    parser.add_argument("--model_dir",
                        type=str,
                        default="./checkpoints/atnet_raw_pca/")
                        # default="/mnt/disk1/dat/lchen63/grid/model/model_gan_r")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./sample/atnet_raw_pca/")
                        # default="/mnt/disk1/dat/lchen63/grid/sample/model_gan_r/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='GRID')
    parser.add_argument('--lstm', type=bool, default= True)
    parser.add_argument('--num_thread', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=4e-4)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_name', type=str, default = './checkpoints/atnet_raw_pca_with_exmaple/atnet_lstm_7.pth')
    parser.add_argument('--threeD', action='store_true')

    return parser.parse_args()


def main(config):
    t = trainer.Trainer(config)
    t.fit()

if __name__ == "__main__":

    config = parse_args()
    config.is_train = 'train'

    import raw2lmark_pca_train as trainer
    if config.threeD:
        config.model_dir = config.model_dir + '_3d'
        config.sample_dir = config.sample_dir + '_3d'

    if not os.path.exists(config.model_dir):
        os.mkdir(config.model_dir)
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    config.cuda1 = torch.device('cuda:0')
    main(config)
