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
from collections import OrderedDict
import argparse

from data.dataset import  GRID_1D_lstm_landmark

from models.networks import  AT_net

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


def parse_args():
    parser = argparse.ArgumentParser()
  
    parser.add_argument("--batch_size",
                        type=int,
                        default=1)
    parser.add_argument("--lstm_len",
                        type=int,
                        default=32)
    parser.add_argument("--cuda",
                        default=True)
    
    parser.add_argument("--model_name",
                        type=str,
                        default="./checkpoints/atnet/atnet_lstm_2.pth")
                        # default="/mnt/disk1/dat/lchen63/lrw/model/model_gan_r2/r_generator_38.pth")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./sample/atnet_test/")
                        # default="/mnt/disk1/dat/lchen63/lrw/test_result/model_gan_r2/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='grid')
    parser.add_argument('--lstm', type=bool, default=True)
    parser.add_argument('--num_thread', type=int, default=1)
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
   

    return parser.parse_args()
config = parse_args()


def test():
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.cuda1 = torch.device('cuda:0')
    config.is_train = 'test'
    generator = AT_net()
    device_ids = [int(i) for i in config.device_ids.split(',')]
    generator    = nn.DataParallel(generator, device_ids= device_ids).cuda()
    
    
    xLim=(0.0, 256.0)
    yLim=(0.0,256.0)
    xLab = 'x'
    yLab = 'y'
    # try:
    #     state_dict = multi2single(config.model_name, 1)
    #     generator.load_state_dict(state_dict)
    # except:
    generator.load_state_dict(torch.load(config.model_name))
    print ('load pretrained [{}]'.format(config.model_name))

    
    dataset = GRID_1D_lstm_landmark( train=config.is_train)
    data_loader = DataLoader(dataset,
                    batch_size=config.batch_size,
                    num_workers= config.num_thread,
                    shuffle=False, drop_last=True)
    data_iter = iter(data_loader)
    data_iter.next()
    if not os.path.exists(config.sample_dir):
        os.mkdir(config.sample_dir)
    if not os.path.exists(os.path.join(config.sample_dir, 'fake')):
        os.mkdir(os.path.join(config.sample_dir, 'fake'))
    if not os.path.exists(os.path.join(config.sample_dir, 'real')):
        os.mkdir(os.path.join(config.sample_dir, 'real'))
    # if config.cuda:
    #     generator = generator.cuda()
    generator.eval()
    mse_loss_fn = nn.MSELoss()
    for step,  (example_landmark, lmark, audio, lmark_path) in enumerate(data_loader):
        with torch.no_grad():
            print (step, lmark_path)
            if step == 5:
                break
            if config.cuda:
                lmark    = Variable(lmark.float()).cuda(config.cuda1)
                audio = Variable(audio.float()).cuda(config.cuda1)
                example_landmark = Variable(example_landmark.float()).cuda(config.cuda1) 
                mse_loss_fn   = mse_loss_fn.cuda(config.cuda1)
            fake_lmark= generator(example_landmark, audio)
            loss =  mse_loss_fn(fake_lmark , lmark) 
            print("  loss1: {:.8f}".format( loss))
            fake_lmark = fake_lmark.view(config.batch_size, config.lstm_len, 136)
            fake_lmark = fake_lmark.data.cpu().numpy() 
            lmark = lmark.view(config.batch_size, config.lstm_len, 136)
            lmark = lmark.data.cpu().numpy()

            for indx in range(config.batch_size):
                for jj in range(config.lstm_len):
                    name = "{}/real_{}_{}_{}.png".format(os.path.join(config.sample_dir, 'real'),step, indx,jj)
                    util.plot_flmarks(lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
                    name = "{}/fake_{}_{}_{}.png".format(os.path.join(config.sample_dir, 'fake'),step, indx,jj)
                    util.plot_flmarks(fake_lmark[indx,jj], name, xLim, yLim, xLab, yLab, figsize=(10, 10))
           



test()