



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

from data.a2l_dataset import  GRID_raw_pca_landmark, Demo_raw_pca_landmark#, GRID_raw_pca_3dlandmark , GRID_deepspeech_pca_landmark

from models.networks import  A2L, A2L_deeps

from models.networks import  A2L

from torch.nn import init
from utils import util, face_utils
import librosa

from scipy.io import wavfile

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
                        default=60)
    parser.add_argument("--cuda",
                        default=True)
    
    parser.add_argument("--model_name",
                        type=str,
                        default="./checkpoints/atnet_raw_pca_with_exmaple_select/atnet_lstm_23.pth")
                        # default="/mnt/disk1/dat/lchen63/lrw/model/model_gan_r2/r_generator_38.pth")
                        # default='/media/lele/DATA/lrw/data2/model')
    parser.add_argument("--sample_dir",
                        type=str,
                        default="./sample/grid_test/")
                        # default="/mnt/disk1/dat/lchen63/lrw/test_result/model_gan_r2/")
                        # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--dataset', type=str, default='Grid')
    parser.add_argument('--lstm', type=bool, default=True)
    parser.add_argument('--num_thread', type=int, default=0)
    parser.add_argument('--threeD', action='store_true')
    # parser.add_argument('--flownet_pth', type=str, help='path of flownets model')
    parser.add_argument('--deeps', action='store_true')
   

    return parser.parse_args()
config = parse_args()


def test():
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
    config.cuda1 = torch.device('cuda:0')
    config.is_train = 'demo'
    if config.deeps:
        generator = A2L_deeps()
    else:
        generator = A2L()
    device_ids = [int(i) for i in config.device_ids.split(',')]
    generator    = nn.DataParallel(generator, device_ids= device_ids).cuda()
    
    
    xLim=(0.0, 256.0)
    yLim=(0.0,256.0)
    xLab = 'x'
    yLab = 'y'
    if config.threeD:
            mean =  np.load('./basics/mean_grid_front_3d.npy')
            component = np.load('./basics/U_grid_front_3d.npy')
    else:
        mean =  np.load('./basics/mean_grid_front.npy')
        component = np.load('./basics/U_grid_front.npy')
    # try:
    #     state_dict = multi2single(config.model_name, 1)
    #     generator.load_state_dict(state_dict)
    # except:
    generator.load_state_dict(torch.load(config.model_name))
    print ('load pretrained [{}]'.format(config.model_name))

    
    if config.threeD:
        dataset = GRID_raw_pca_3dlandmark( train= config.is_train)
        
    else:
        # dataset = crema_raw_pca_landmark( train= config.is_train, length= 60)
        
        dataset = Demo_raw_pca_landmark( train= config.is_train, name = config.dataset)
        
    data_loader = DataLoader(dataset,
                    batch_size=config.batch_size,
                    num_workers= config.num_thread,
                    shuffle=True, drop_last=True)
    # data_iter = iter(data_loader)
    # data_iter.next()
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
    for step,  (example_landmark, lmark, audio, lmark_path, diff, audio_path, rt_path) in enumerate(data_loader):
        # if step == 5:
        #     break
        print (step, lmark_path)
        length = lmark.shape[1]
        norm_lmark = np.load('./basics/standard.npy')[:,-1].reshape(1,68,1)
    # print (norm_lmark.shape)
        with torch.no_grad():
            print (step, lmark_path)
            if config.dataset == 'Grid':
                video_p =   lmark_path[0].split('/')[-2] +'__' +lmark_path[0].split('/')[-1][:-4]
            elif config.dataset == 'Vox':
                video_p = lmark_path[0].split('/')[-3] + '__' +  lmark_path[0].split('/')[-2] +'__' +lmark_path[0].split('/')[-1][:-4]
            # if step == 20:
            #     break
            if config.cuda:
                if config.is_train == 'demo':
                    lmark    = Variable(lmark[0].float()).cuda(config.cuda1)
                    audio = Variable(audio[0].float()).cuda(config.cuda1)
                    example_landmark = Variable(example_landmark[0].float()).cuda(config.cuda1) 
                else:
                    lmark    = Variable(lmark.float()).cuda(config.cuda1)
                    audio = Variable(audio.float()).cuda(config.cuda1)
                    example_landmark = Variable(example_landmark.float()).cuda(config.cuda1) 
                # mse_loss_fn   = mse_loss_fn.cuda(config.cuda1)
            fake_lmark, _ = generator(example_landmark, audio)
            ##### for save
            gg =  fake_lmark.cpu().numpy().copy()
            gg = np.dot(gg,component) + mean
            # print  (gg.shape)
            gg = gg.reshape(length, 68, 2)
            gg = gg + diff.cpu().numpy()
            norm_lmark = np.repeat(norm_lmark,length,  axis = 0)
            # print (norm_lmark.shape)
            # print (gg.shape, norm_lmark.shape)
            gg = np.concatenate((gg,norm_lmark),2 )
            rotated_gg = np.zeros((gg.shape[0], 68 , 3))
            rt = np.load(rt_path[0])
            for i in range(gg.shape[0]):
                rotated_gg[i] = util.reverse_rt(gg[i], rt[i])
            np.save( os.path.join(config.sample_dir , video_p + '_diff.npy'),gg )

            np.save( os.path.join(config.sample_dir , video_p + '_diff_rotated.npy'),rotated_gg )
            print ('+++++++++++++++++++++++++++',  os.path.join(config.sample_dir , video_p + '_diff_rotated.npy'))
            # # for visualization
            # lmark = example_landmark.data.cpu().numpy()
            # fake_lmark = fake_lmark.data.cpu().numpy()
            # lmark = np.dot(lmark,component) + mean
            # fake_lmark = np.dot(fake_lmark,component) + mean 
            # if config.threeD:
            #     lmark = lmark.reshape(config.batch_size, length , 68 ,3)
            #     fake_lmark = fake_lmark.reshape(config.batch_size, length , 68 ,3)
            # else:
            #     print (lmark.shape)
            #     lmark = lmark.reshape(config.batch_size,length, 68 ,2)
            #     fake_lmark = fake_lmark.reshape(config.batch_size,length , 68 , 2) #+ diff.cpu().numpy()
            

            # lmark = lmark[:,:,:,:2].reshape(config.batch_size, length, 68 * 2)
            # fake_lmark = fake_lmark[:,:,:,:2].reshape(config.batch_size, length, 68 * 2)
            # face_utils.write_video_wpts_wsound(fake_lmark[0], audio_path[0], config.sample_dir, 'fake__'+ video_p , [0.0,256.0], [0.0,256.0])

            # face_utils.write_video_wpts_wsound(lmark[0],audio_path[0], config.sample_dir, 'real__'+ video_p , [0.0,256.0], [0.0,256.0])
            

test()