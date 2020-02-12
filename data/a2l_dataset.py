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

class GRID_raw_lstm_pca_landmark(Dataset):
    def __init__(self,
                 train='train'):
        self.train = train
        self.num_frames = 32
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
            _file = open(os.path.join(self.root_path, "img_demo.pkl"), "rb")
            self.demo_data = pkl.load(_file)
            _file.close()

        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
# data_original = np.dot(data_reduced,component) + mean
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
        if self.train == 'train':
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            # mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
            lmark = np.load(lmark_path)[:,:,:2]
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'.wav' )
            rnd_dB = np.random.randint(0, high=len(self.augList), size=[1, ])[0]
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark.reshape(lmark.shape[0], 136)
            lmark = np.dot(lmark - self.mean, self.component.T)

            lmark = torch.FloatTensor(lmark)
            fs, mfcc = wavfile.read( audio_path)
            chunck_size = int(fs * 0.04 )
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[0,:]  # since the lips in all 0 frames are closed 
            r =random.choice(
                [x for x in range(0,41)])
            mfccs = []
            for ind in range(self.num_frames):
                t_mfcc =mfcc[(r + ind )* chunck_size : (r + ind + 7)* chunck_size]
                t_mfcc = t_mfcc*np.power(10.0, self.augList[rnd_dB]/20.0)
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r : r + self.num_frames,:]
            # example_landmark = example_landmark.contiguous().view(-1)
            # landmark = landmark.contiguous().view( self.num_frames, -1 )

            return example_landmark, landmark, mfccs
        else:

            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'.wav' )
            lmark = np.load(lmark_path)[:,:,:-1]
            
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark.reshape(lmark.shape[0], 136)
            # print (lmark.shape, self.mean.shape, self.component.T.shape)
            lmark = np.dot(lmark - self.mean, self.component.T)
            lmark = torch.FloatTensor(lmark)
            
            fs, mfcc = wavfile.read( audio_path)
            chunck_size =int(fs * 0.04 ) 
            example_landmark =lmark[0,:]  # since the lips in all 0 frames are closed 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[0,:]  # since the lips in all 0 frames are closed 
            r =random.choice(
                [x for x in range(0,41)])
            mfccs = []
            for ind in range(self.num_frames):
                t_mfcc =mfcc[(r + ind )*chunck_size: (r + ind + 7)*chunck_size]
                
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            lmark  =lmark[r : r + self.num_frames,:]
            # example_landmark = example_landmark.contiguous().view(-1)
            # lmark = lmark.contiguous().view(self.num_frames, -1 )

            return example_landmark, lmark, mfccs, lmark_path

       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            print ('8888888888888')


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
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][2]) 
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][2].split('_')[1])
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

            return example_landmark, lmark, mfccs,  lmark_path 
       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)


class GRID_deepspeech_pca_landmark(Dataset):
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
            self.datalist = pkl.load(_file)
            _file.close()
        print (len(self.datalist))
        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
        if self.train == 'train':
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_%05d_diff.npy'%self.datalist[index][4])
            lmark = np.load(lmark_path)[:,:,:2]
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][4])
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'_dp.npy' )
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
            dp_feature = np.load(audio_path)
            left_append = dp_feature[:3]
            right_append = dp_feature[-4 :]
            dp_feature = np.insert( dp_feature, 0, left_append ,axis=  0)
            dp_feature = np.insert( dp_feature, -1, right_append ,axis=  0)
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
            t_dp =dp_feature[r  : (r + 7)].reshape(1, -1)
            t_dp = torch.FloatTensor(t_dp)

            landmark  =lmark[r]
            # example_landmark = example_landmark.contiguous().view(-1)
            # landmark = landmark.contiguous().view( self.num_frames, -1 )

            return example_landmark, landmark, t_dp ,  lmark_path +'___' +  str(r)
        elif self.train=='test':

            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'_dp.npy' )
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
            
            dp_feature = np.load(audio_path)
            left_append = dp_feature[:3]
            right_append = dp_feature[-4 :]
            dp_feature = np.insert( dp_feature, 0, left_append ,axis=  0)
            dp_feature = np.insert( dp_feature, -1, right_append ,axis=  0)

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
            
            t_dp =dp_feature[r  : (r + 7)].reshape(1, -1)
            t_dp = torch.FloatTensor(t_dp)

            landmark  =lmark[r]
            

            return example_landmark, landmark, t_dp,  lmark_path +'___' +  str(r)

        elif self.train=='demo':
            
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'_dp.npy' )
            lmark = np.load(lmark_path)[:,:,:-1]
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][2]) 
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][2].split('_')[1])
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
           
            dp_feature = np.load(audio_path)
            left_append = dp_feature[:3]
            right_append = dp_feature[-4 :]
            dp_feature = np.insert( dp_feature, 0, left_append ,axis=  0)
            dp_feature = np.insert( dp_feature, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            
            example_landmark = example_landmark.repeat(75,1)
            dps = []
            for r in range(75):
                t_dp =dp_feature[r  : (r + 7)].reshape(1, -1)
                t_dp = torch.FloatTensor(t_dp)
                dps.append(t_mfcc)
            dps = torch.stack(dps, 0)
            # example_landmark = example_landmark.contiguous().view(-1)
            # lmark = lmark.contiguous().view(self.num_frames, -1 )

            return example_landmark, lmark, dps,  lmark_path 
       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)

class GRID_raw_pca_3dlandmark(Dataset):
    def __init__(self,
                 train='train'):
        self.train = train
        self.num_frames = 32
        self.root_path = '/home/cxu-serve/p1/common/grid'
        if self.train=='train':
            _file = open(os.path.join(self.root_path,  'pickle','train_audio2lmark_grid_3d.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid_3d.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid_3d.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()
        print (len(self.datalist))
        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front_3d.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front_3d.npy')
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
        if self.train == 'train':
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_%05d_diff_3d.npy'%self.datalist[index][4])
            lmark = np.load(lmark_path)
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
            lmark = lmark.reshape(lmark.shape[0], 204)
            lmark = np.dot(lmark - self.mean, self.component.T)
            length = lmark.shape[0]
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
                    r =random.choice( [x for x in range(0, 10)] + [x for x in range(65, min(75, length))])
                else:
                    r =random.choice( [x for x in range(0, 10)] )
            else:
                if  self.datalist[index][3] == True:
                    r =random.choice([x for x in range(65,  min(75, length))])
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
            lmark = np.load(lmark_path)
            # if len(self.datalist[index]) != 5:
            #     print (len(self.datalist[index]) , self.datalist[index])
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_%05d_diff_3d.npy'%self.datalist[index][4])
            diff = np.load(diff_path)
            length = lmark.shape[0]
            reference_id = int(self.datalist[index][4])
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 204)
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
                    r =random.choice( [x for x in range(0, 10)] + [x for x in range(65,  min(75, length))])
                else:
                    r =random.choice( [x for x in range(0, 10)] )
            else:
                if  self.datalist[index][3] == True:
                    r =random.choice([x for x in range(65,  min(75, length))])
                else:
                    r =random.choice( [x for x in range(10, 65)] )
            
            t_mfcc =mfcc[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
            t_mfcc = torch.FloatTensor(t_mfcc)
            landmark  =lmark[r]
            

            return example_landmark, landmark, t_mfcc,  lmark_path +'___' +  str(r)

        elif self.train=='demo':
            
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            audio_path = os.path.join('/home/cxu-serve/p1/common/grid/audio' ,self.datalist[index][0],  self.datalist[index][1] +'.wav' )
            lmark = np.load(lmark_path)
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][2]) 
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][2].split('_')[1])
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            lmark = lmark - diff
            lmark = lmark.reshape(lmark.shape[0], 204)
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

            return example_landmark, lmark, mfccs,  lmark_path 
       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)
# dataset = GRID_raw_pca_landmark( train='train')
# data_loader = DataLoader(dataset,
#                             batch_size=2,
#                             num_workers=1,
#                             shuffle=False, drop_last=True)
# for i in range (10):
#     for step, (example_landmark, lmark, audio) in enumerate(data_loader):

#         print (example_landmark.shape)
