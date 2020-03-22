import os
from datetime import datetime
import pickle as pkl
import random
import scipy.ndimage.morphology

import PIL
import cv2
import matplotlib
# matplotlib.use('pdf')
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import mmcv
from io import BytesIO
from PIL import Image
import sys
# sys.path.insert(1, '../utils')
# from .. import utils
from utils import util
from utils import face_utils
from torch.utils.data import DataLoader
from scipy.io import wavfile
def mounth_open2close(lmark): # if the open rate is too large, we need to manually make the mounth to be closed.
    # the input lamrk need to be (68,2 ) or (68,3)
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    upper_part = [49,50,51,52,53]
    lower_part = [59,58,57,56,55]
    diffs = []

    for k in range(3):
        mean = (lmark[open_pair[k][0],:2] + lmark[open_pair[k][1],:2] )/ 2
        print (mean)
        tmp = lmark[open_pair[k][0],:2]
        diffs.append((mean - lmark[open_pair[k][0],:2]).copy())
        lmark[open_pair[k][0],:2] = mean - (mean - lmark[open_pair[k][0],:2]) * 0.3
        lmark[open_pair[k][1],:2] = mean + (mean - lmark[open_pair[k][0],:2]) * 0.3
    diffs.insert(0, 0.6 * diffs[2])
    diffs.append( 0.6 * diffs[2])
    print (diffs)
    diffs = np.asarray(diffs)
    lmark[49:54,:2] +=  diffs
    lmark[55:60,:2] -=  diffs 
    return lmark
class GRID_raw_pca_landmark(Dataset):
    def __init__(self,
                 train='train'):
        self.train = train
        self.root_path = '/home/cxu-serve/p1/common/grid'
        if self.train=='train':
            _file = open(os.path.join(self.root_path,  'pickle','train_audio2lmark_grid.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
            self.datalist0 = pkl.load(_file)
            _file.close()
            self.datalist= []
            for gg in range(len(self.datalist0)):
                if self.datalist0[gg][0] in set(['s14', 's15']):
                    self.datalist.append(self.datalist0[gg])
        print (len(self.datalist))
        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
        if self.train == 'train':
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_%05d_diff.npy'%self.datalist[index][4])
            lmark = np.load(lmark_path)[:,:,:2]
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][4])
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'.wav' )
            rnd_dB = np.random.randint(0, high=len(self.augList), size=[1, ])[0]
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            lmark = np.dot(lmark - self.mean, self.component.T)

            lmark = torch.FloatTensor(lmark)
            fs, mfcc = wavfile.read( audio_path)
            chunck_size = int(fs * 0.04 )
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            if self.datalist[index][2] == True :
                if  self.datalist[index][3] == True:
                    r =random.choice( [x for x in range(0, 10)] + [x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(0, 10)] )
            else:
                if  self.datalist[index][3] == True:
                    r =random.choice([x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(10, 65)] )
            # if lmark.shape[0] != 75:
            #     print  (lmark.shape[0])
            # r = 74
            t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
            t_mfcc = t_mfcc*np.power(10.0, self.augList[rnd_dB]/20.0)
            t_mfcc = torch.FloatTensor(t_mfcc)

            landmark  =lmark[r]
            # example_landmark = example_landmark.contiguous().view(-1)
            # landmark = landmark.contiguous().view( self.num_frames, -1 )

            return example_landmark, landmark, t_mfcc ,  lmark_path +'___' +  str(r)
        elif self.train=='test':

            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'.wav' )
            lmark = np.load(lmark_path)[:,:,:-1]
            # if len(self.datalist[index]) != 5:
            #     print (len(self.datalist[index]) , self.datalist[index])
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_%05d_diff.npy'%self.datalist[index][4])
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][4])
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            # print (lmark.shape, self.mean.shape, self.component.T.shape)
            lmark = np.dot(lmark - self.mean, self.component.T)
            lmark = torch.FloatTensor(lmark)
            
            fs, mfcc = wavfile.read( audio_path)
            chunck_size =int(fs * 0.04 ) 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            if self.datalist[index][2] == True :
                if  self.datalist[index][3] == True:
                    r =random.choice( [x for x in range(0, 10)] + [x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(0, 10)] )
            else:
                if  self.datalist[index][3] == True:
                    r =random.choice([x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(10, 65)] )
            
            t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
            t_mfcc = torch.FloatTensor(t_mfcc)
            landmark  =lmark[r]
            

            return example_landmark, landmark, t_mfcc,  lmark_path +'___' +  str(r)

        elif self.train=='demo':
            

            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'.wav' )
            lmark = np.load(lmark_path)[:,:,:-1]
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_%05d_diff.npy'%self.datalist[index][4])
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][4])
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            # print (lmark.shape, self.mean.shape, self.component.T.shape)
            lmark = np.dot(lmark - self.mean, self.component.T)
            lmark = torch.FloatTensor(lmark)#.view(75,20)

            fs, mfcc = wavfile.read( audio_path)
            chunck_size =int(fs * 0.04 ) 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 

            example_landmark = example_landmark.repeat(75,1)
            mfccs = []
            for r in range(75):
                t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, 0)
            # example_landmark = example_landmark.contiguous().view(-1)
            # lmark = lmark.contiguous().view(self.num_frames, -1 )
            print ('++++')

            print (example_landmark.shape, lmark.shape, mfccs.shape, lmark_path)

            return example_landmark, lmark, mfccs,  lmark_path , diff
       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)


class crema_raw_pca_landmark(Dataset):
    def __init__(self,
                 train='train', length = 50):
        self.train = train
        self.length = length
        self.root_path = '/home/cxu-serve/p1/common/CREMA'
        if self.train=='train':
            _file = open(os.path.join(self.root_path,  'pickle','train_lmark2img.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(self.root_path,  'pickle','train_lmark2img.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(self.root_path,  'pickle','train_lmark2img.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        print (len(self.datalist))
        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
        if self.train == 'train':
            print (self.datalist[index])
            lmark_path = os.path.join(self.root_path,  'VideoFlash', self.datalist[index][0][:-10] +'_front.npy'  )
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][0][:-10] + '_%05d_diff.npy'%self.datalist[index][-1])
            lmark = np.load(lmark_path)[:,:,:2]
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][-1])
            audio_path = os.path.join(self.root_path, 'AudioWAV' , self.datalist[index][0][:-11] +'.wav' )
            rnd_dB = np.random.randint(0, high=len(self.augList), size=[1, ])[0]
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            lmark = np.dot(lmark - self.mean, self.component.T)

            lmark = torch.FloatTensor(lmark)
            fs, mfcc = wavfile.read( audio_path)
            chunck_size = int(fs * 0.04 )
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            if self.datalist[index][2] == True :
                if  self.datalist[index][3] == True:
                    r =random.choice( [x for x in range(0, 10)] + [x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(0, 10)] )
            else:
                if  self.datalist[index][3] == True:
                    r =random.choice([x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(10, 65)] )
            # if lmark.shape[0] != 75:
            #     print  (lmark.shape[0])
            # r = 74
            t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
            t_mfcc = t_mfcc*np.power(10.0, self.augList[rnd_dB]/20.0)
            t_mfcc = torch.FloatTensor(t_mfcc)

            landmark  =lmark[r]
            # example_landmark = example_landmark.contiguous().view(-1)
            # landmark = landmark.contiguous().view( self.num_frames, -1 )

            return example_landmark, landmark, t_mfcc ,  lmark_path +'___' +  str(r)
        elif self.train=='test':

            lmark_path = os.path.join(self.root_path,  'VideoFlash', self.datalist[index][0][:-10] +'_front.npy'  )
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][0][:-10] + '_%05d_diff.npy'%self.datalist[index][-1])
            lmark = np.load(lmark_path)[:,:,:2]
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][-1])
            audio_path = os.path.join(self.root_path, 'AudioWAV' , self.datalist[index][0][:-11] +'.wav' )
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            # print (lmark.shape, self.mean.shape, self.component.T.shape)
            lmark = np.dot(lmark - self.mean, self.component.T)
            lmark = torch.FloatTensor(lmark)
            
            fs, mfcc = wavfile.read( audio_path)
            chunck_size =int(fs * 0.04 ) 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            if self.datalist[index][2] == True :
                if  self.datalist[index][3] == True:
                    r =random.choice( [x for x in range(0, 10)] + [x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(0, 10)] )
            else:
                if  self.datalist[index][3] == True:
                    r =random.choice([x for x in range(65, 74)])
                else:
                    r =random.choice( [x for x in range(10, 65)] )
            
            t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
            t_mfcc = torch.FloatTensor(t_mfcc)
            landmark  =lmark[r]
            

            return example_landmark, landmark, t_mfcc,  lmark_path +'___' +  str(r)

        elif self.train=='demo':
            
            lmark_path = os.path.join(self.root_path,  'VideoFlash', self.datalist[index][0][:-10] +'_front.npy'  )
            diff_path =  os.path.join(self.root_path ,  'VideoFlash' ,  self.datalist[index][0][:-10] + '_%05d_diff.npy'%self.datalist[index][-1])
            orig_lmark = np.load(lmark_path)[:,:,:2]
            diff = np.load(diff_path)
            orig_lmark = np.insert( orig_lmark, -1, orig_lmark[0] ,axis=  0)
            print (orig_lmark.shape)
            lmark = np.zeros((int(orig_lmark.shape[0] * 25 / 30) , 68,2))
            print (lmark.shape, orig_lmark.shape, )
            print ('-----')
            for i in range(lmark.shape[0]):
                print (i, orig_lmark[int(1.2* i)+1].shape)
                lmark[i] = orig_lmark[int(1.2* i)+1]    

            self.length = lmark.shape[0]
            
            reference_id = int(self.datalist[index][-1])
            audio_path = os.path.join(self.root_path, 'AudioWAV' , self.datalist[index][0][:-11] +'.wav' )
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            # print (lmark.shape, self.mean.shape, self.component.T.shape)
            lmark = np.dot(lmark - self.mean, self.component.T)
            lmark = torch.FloatTensor(lmark)#.view(75,20)
            command = 'ffmpeg -i ' + audio_path +' -ar 50000 -y ./tmp.wav'
            os.system(command)
            fs, mfcc = wavfile.read( './tmp.wav')
            print (mfcc.shape)
            print ('==========================')
            chunck_size =int(fs * 0.04 ) 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-7 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 

            example_landmark = example_landmark.repeat(self.length,1)
            mfccs = []
            # print (s)
            for r in range(self.length):
                t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
                t_mfcc = torch.FloatTensor(t_mfcc)
                print (t_mfcc.shape)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, 0)
            # example_landmark = example_landmark.contiguous().view(-1)
            # lmark = lmark.contiguous().view(self.num_frames, -1 )
            print ('++++')

            print (example_landmark.shape, lmark.shape, mfccs.shape, lmark_path)

            return example_landmark, lmark, mfccs,  lmark_path, audio_path, diff
       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)


class LRW_raw_pca_landmark(Dataset):
    def __init__(self,
                 train='train'):
        self.train = train
        self.root_path = '/home/cxu-serve/p1/common/lrw'
        if self.train=='train':
            _file = open(os.path.join(self.root_path,  'pickle','train_audio2lmark_grid.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(self.root_path,  'pickle','test3_lmark2img.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
            
        print (len(self.datalist))
        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
    def __getitem__(self, index):
        if self.train=='demo':
            print (self.datalist[index])
            lmark_path = self.datalist[index][0] + '_front.npy'
            ref_lmark = np.load('/home/cxu-serve/p1/common/grid/align/s23/bbad1s_front.npy')[:,:,:2]
            audio_path = self.datalist[index][0].replace('video', 'audio') +'.wav'
            lmark = np.load(lmark_path)
            lmark = np.load(lmark_path)[:,:,:2]
            diff_path = '/home/cxu-serve/p1/common/grid/align/s23/bbad1s_00067_diff.npy'  #lmark_path[:-10] +'_%05d_diff.npy'%(self.datalist[index][-1]) 
            diff = np.load(diff_path)
            tmpwavfile = './tmp.wav'
            
            commandlline = 'ffmpeg  -y -i %s  -ar %d %s '%( audio_path, 50000 , tmpwavfile)
            os.system(commandlline)
            fs, mfcc = wavfile.read( tmpwavfile)
            chunck_size =int(fs * 0.04 ) 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            # print (lmark.shape, self.mean.shape, self.component.T.shape)
            lmark = np.dot(lmark - self.mean, self.component.T)
            lmark = torch.FloatTensor(lmark)

            example_landmark =ref_lmark[67] - diff  # since the lips in all 0 frames are closed 
            example_landmark = example_landmark.reshape(1, 136)
            print (example_landmark.shape)

            example_landmark = np.dot(example_landmark - self.mean, self.component.T)
            example_landmark = example_landmark.repeat(29,0)
            mfccs = []
            for r in range(29):
                t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, 0)
            print (example_landmark.shape, lmark.shape, mfccs.shape, lmark_path, diff.shape, audio_path)
            return example_landmark, lmark, mfccs,  lmark_path , diff, audio_path
       
    
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)



class Demo_raw_pca_landmark(Dataset):
    def __init__(self,
                 train='train', name = 'Vox' ):
        self.train = train
        self.name = name

        
        if self.name == 'Vox':
            self.root_path = '/home/cxu-serve/p1/common/voxceleb2'
            if self.train=='train':
                _file = open(os.path.join(self.root_path,  'pickle','train_audio2lmark_grid.pkl'), "rb")
                self.datalist = pkl.load(_file)
                _file.close()
            elif self.train =='test':
                _file = open(os.path.join(self.root_path,  'pickle','test_lmark2img.pkl'), "rb")
                self.datalist = pkl.load(_file)
                _file.close()
            elif self.train =='demo' :
                _file = open(os.path.join(self.root_path,  'pickle','test_lmark2img.pkl'), "rb")
                self.datalist = pkl.load(_file)
                _file.close()
                
            self.get_files()
            print (len(self.datalist))
            
        elif self.name== 'Grid':
            self.root_path = '/home/cxu-serve/p1/common/grid'
            if self.train =='demo' :
                _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
                self.datalist0 = pkl.load(_file)
                _file.close()
                
            self.datalist= []
            for gg in range(len(self.datalist0)):
                if self.datalist0[gg][0] in set([ 's15']):
                    self.datalist.append(self.datalist0[gg])
        print (len(self.datalist))
        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
    def get_files(self):
        root_file = "/home/cxu-serve/p1/common/vox_good"
        video_files = os.listdir(root_file)
        self.datalist = [data for data in self.datalist if '{}_{}_{}_aligned'.format(data[0], data[1], data[2]) in video_files]



    def __getitem__(self, index):
        if self.train=='demo':
            print (self.datalist[index])
            ref_lmark = np.load('./basics/s1_pgbk6n_01.npy')[:,:2]
            if self.name == 'Vox':
                lmark_path = os.path.join( self.root_path, 'unzip', 'test_video'  , self.datalist[index][0] , self.datalist[index][1] , self.datalist[index][2] + '_aligned_front.npy')  
                rt_path = os.path.join( self.root_path, 'unzip', 'test_video'  , self.datalist[index][0] , self.datalist[index][1] , self.datalist[index][2] + '_aligned_rt.npy')
                audio_path = os.path.join( self.root_path, 'unzip', 'test_audio'  , self.datalist[index][0] , self.datalist[index][1] , self.datalist[index][2] + '.wav')  
            elif self.name == 'Grid':
                lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
                lmark_path2 = os.path.join(self.root_path ,  'align' ,'s14' , 'brbl5p_front.npy') 
                rt_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_rt.npy') 
                
                audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'.wav' )
            lmark = np.load(lmark_path)
            self.length = lmark.shape[0]
            lmark = np.load(lmark_path)[:,:,:2]
            current_tempolate =mounth_open2close(np.load(lmark_path2)[:,:,:2][0].copy())
            diff = current_tempolate -  ref_lmark
            tmpwavfile = './tmp.wav'
            
            commandlline = 'ffmpeg  -y -i %s  -ar %d %s '%( audio_path, 50000 , tmpwavfile)
            os.system(commandlline)
            fs, mfcc = wavfile.read( tmpwavfile)
            chunck_size =int(fs * 0.04 ) 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            # lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 136)
            # print (lmark.shape, self.mean.shape, self.component.T.shape)
            lmark = np.dot(lmark - self.mean, self.component.T)
            lmark = torch.FloatTensor(lmark)

            example_landmark =   ref_lmark #[67] #- diff  # since the lips in all 0 frames are closed 
            example_landmark = example_landmark.reshape(1, 136)
            print (example_landmark.shape)

            example_landmark = np.dot(example_landmark - self.mean, self.component.T)
            example_landmark = example_landmark.repeat(self.length,0)
            mfccs = []
            for r in range(self.length):
                t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, 0)
            print (example_landmark.shape, lmark.shape, mfccs.shape, lmark_path, diff.shape, audio_path, rt_path)
            return example_landmark, lmark, mfccs,  lmark_path , diff, audio_path, rt_path
       
    
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)
