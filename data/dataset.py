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




class LRSLmark2rgbDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """

    def __init__(self,opt):
        """
        Instantiates the Dataset.
        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        self.output_shape   = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.num_frames
        self.opt = opt
        self.root  = opt.dataroot
        if opt.isTrain:
            _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
        else :
            _file = open(os.path.join(self.root, 'pickle','test_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
        # elif opt.demo:

        #     _file = open(os.path.join(self.root, 'txt', "demo.pkl"), "rb")
            
        #     self.data = pkl.load(_file)
        #     _file.close()

        # else:
        #     _file = open(os.path.join(self.root, 'txt', "front_rt2.pkl"), "rb")
        #     self.data = pkl.load(_file)
        #     _file.close()
        print (len(self.data))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.data) 

    
    def name(self):
        return 'LRSLmark2rgbDataset'

    def __getitem__(self, index):
        # try:
            # mis_vid = self.data[random.randint(0, self.__len__() - 1)]

            v_id = self.data[index]

            video_path = os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] + '_crop.mp4'  )
            # mis_video_path = os.path.join(self.root, 'pretrain', mis_vid[0] , mis_vid[1][:5] + '_crop.mp4'  )

            lmark_path = os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] +'_original.npy'  )
            print (lmark_path)
            lmark = np.load(lmark_path)
            lmark = lmark[:,:,:-1]
            v_length = lmark.shape[0]
            # real_video  = mmcv.VideoReader(video_path)

            # sample frames for embedding network
            if self.opt.use_ft:
                if self.num_frames  ==1 :
                    input_indexs = [0]
                    target_id = 0
                elif self.num_frames == 8:
                    input_indexs = [0,7,15,23,31,39,47,55]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs) 

                elif self.num_frames == 32:
                    input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs)                    
            else:
                input_indexs  = set(random.sample(range(0,64), self.num_frames))
                # we randomly choose a target frame 
                target_id =  random.randint( 64, v_length - 1)
                   
            if type(target_id) == list:
                target_id = target_id[0]
            cap = cv2.VideoCapture(video_path)
            real_video = []
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    real_video.append(frame)
                    
                else:
                    break
            reference_frames = []
            for t in input_indexs:
                rgb_t =   cv2.cvtColor(real_video[t],cv2.COLOR_BGR2RGB )
                lmark_t = lmark[t]
                lmark_rgb = util.plot_landmarks( lmark_t)
                # resize  to 256
                rgb_t  = cv2.resize(rgb_t, self.output_shape)
                lmark_rgb  = cv2.resize(lmark_rgb, self.output_shape)
                # to tensor
                rgb_t = self.transform(rgb_t)
                lmark_rgb = self.transform(lmark_rgb)
                reference_frames.append(torch.cat([rgb_t, lmark_rgb],0))  # (6, 256, 256)   
            ############################################################################
            
            target_rgb = real_video[target_id]
            target_lmark = lmark[target_id]
            # mis_rgb = mmcv.VideoReader(mis_video_path)[random.randint(0, 64)]
            target_rgb = mmcv.bgr2rgb(target_rgb)
            target_rgb = cv2.resize(target_rgb, self.output_shape)
            target_rgb = self.transform(target_rgb)

            # dif_rgb = real_video[random.randint(0, v_length - 1)]
            # dif_rgb = mmcv.bgr2rgb(dif_rgb)
            # dif_rgb = cv2.resize(dif_rgb, self.output_shape)
            # dif_rgb = self.transform(dif_rgb)

            # mis_rgb = mmcv.bgr2rgb(mis_rgb)
            # mis_rgb = cv2.resize(mis_rgb, self.output_shape)
            # mis_rgb = self.transform(mis_rgb)

        
            target_lmark = util.plot_landmarks(target_lmark)
            target_lmark  = cv2.resize(target_lmark, self.output_shape)
            target_lmark = self.transform(target_lmark)

            reference_frames = torch.cat(reference_frames, dim = 0)
            target_img_path  = os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] , '%05d.png'%target_id  )
            input_dic = {'v_id' : target_img_path, 'target_lmark': target_lmark, 'reference_frames': reference_frames, \
            'target_rgb': target_rgb,  'target_id': target_id }#,  'dif_img': dif_rgb , 'mis_img' :mis_rgb}
            return input_dic
        # except:
        #     return None





class FaceForensicsLmark2rgbDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """
    def __init__(self,opt):
        """
        Instantiates the Dataset.
        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        self.output_shape   = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.num_frames
        self.opt = opt
        self.root  = opt.dataroot
        if opt.isTrain:
            _file = open(os.path.join(self.root, 'pickle','train_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
        else :
            _file = open(os.path.join(self.root, 'pickle','test_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
       
        print (len(self.data))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.data) 

    
    def name(self):
        return 'FaceForensicsLmark2rgbDataset'

    def __getitem__(self, index):
        # try:
            # mis_vid = self.data[random.randint(0, self.__len__() - 1)]
            
            # v_id = self.data[index]

            video_path = self.data[index][1] #os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] + '_crop.mp4'  )
            # mis_video_path = os.path.join(self.root, 'pretrain', mis_vid[0] , mis_vid[1][:5] + '_crop.mp4'  )

            lmark_path  = self.data[index][0]  #= os.path.join(self.root, 'pretrain', v_id[0] , v_id[1]  )

            lmark = np.load(lmark_path)#[:,:,:-1]
            v_length = lmark.shape[0]
            # real_video  = mmcv.VideoReader(video_path)

            # sample frames for embedding network
            if self.opt.use_ft:
                if self.num_frames  ==1 :
                    input_indexs = [0]
                    target_id = 0
                elif self.num_frames == 8:
                    input_indexs = [0,7,15,23,31,39,47,55]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs) 

                elif self.num_frames == 32:
                    input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs)                    
            else:
                input_indexs  = set(random.sample(range(0,64), self.num_frames))
                # we randomly choose a target frame 
                target_id =  random.randint( 64, v_length - 2)
                   
            if type(target_id) == list:
                target_id = target_id[0]
            cap = cv2.VideoCapture(video_path)
            real_video = []
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    real_video.append(frame)
                    
                else:
                    break
            reference_frames = []
            for t in input_indexs:
                rgb_t =   cv2.cvtColor(real_video[t],cv2.COLOR_BGR2RGB )
                lmark_t = lmark[t]
                lmark_rgb = util.plot_landmarks( lmark_t)
                # resize  to 256
                rgb_t  = cv2.resize(rgb_t, self.output_shape)
                lmark_rgb  = cv2.resize(lmark_rgb, self.output_shape)
                # to tensor
                rgb_t = self.transform(rgb_t)
                lmark_rgb = self.transform(lmark_rgb)
                reference_frames.append(torch.cat([rgb_t, lmark_rgb],0))  # (6, 256, 256)   
            ############################################################################
            # print (len(real_video) , len(lmark))
            target_rgb = real_video[target_id]
            target_lmark = lmark[target_id]
            # mis_rgb = mmcv.VideoReader(mis_video_path)[random.randint(0, 64)]
            target_rgb = mmcv.bgr2rgb(target_rgb)
            target_rgb = cv2.resize(target_rgb, self.output_shape)
            target_rgb = self.transform(target_rgb)

            # dif_rgb = real_video[random.randint(0, v_length - 1)]
            # dif_rgb = mmcv.bgr2rgb(dif_rgb)
            # dif_rgb = cv2.resize(dif_rgb, self.output_shape)
            # dif_rgb = self.transform(dif_rgb)

            # mis_rgb = mmcv.bgr2rgb(mis_rgb)
            # mis_rgb = cv2.resize(mis_rgb, self.output_shape)
            # mis_rgb = self.transform(mis_rgb)

        
            target_lmark = util.plot_landmarks(target_lmark)
            target_lmark  = cv2.resize(target_lmark, self.output_shape)
            target_lmark = self.transform(target_lmark)

            reference_frames = torch.cat(reference_frames, dim = 0)
            target_img_path  = os.path.join(video_path[:-4] , '%05d.png'%target_id  )
            input_dic = {'v_id' : target_img_path, 'target_lmark': target_lmark, 'reference_frames': reference_frames, \
            'target_rgb': target_rgb,  'target_id': target_id}# ,  'dif_img': dif_rgb , 'mis_img' :mis_rgb}
            return input_dic
        # except:
        #     return None




class GridLmark2rgbDataset(Dataset):
    """ Dataset object used to access the pre-processed VoxCelebDataset """
    def __init__(self,opt):
        """
        Instantiates the Dataset.
        :param root: Path to the folder where the pre-processed dataset is stored.
        :param extension: File extension of the pre-processed video files.
        :param shuffle: If True, the video files will be shuffled.
        :param transform: Transformations to be done to all frames of the video files.
        :param shuffle_frames: If True, each time a video is accessed, its frames will be shuffled.
        """
        self.output_shape   = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.num_frames
        self.opt = opt
        self.root  = opt.dataroot
        if opt.isTrain:
            _file = open(os.path.join(self.root, 'pickle','train_audio2lmark_grid.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
        else :
            _file = open(os.path.join(self.root, 'pickle','test_audio2lmark_grid.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
       
        print (len(self.data))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.data) 

    
    def name(self):
        return 'GridLmark2rgbDataset'

    def __getitem__(self, index):
            lmark_path = os.path.join(self.root ,  'align' , self.data[index][0] , self.data[index][1] + '_original.npy') 
            video_path = os.path.join(self.root ,  'align' , self.data[index][0] , self.data[index][1] + '_crop.mp4') 
            lmark = np.load(lmark_path)[:,:,:2]
            v_length = lmark.shape[0]

            # sample frames for embedding network
            if self.opt.use_ft:
                if self.num_frames  ==1 :
                    input_indexs = [0]
                    target_id = 0
                elif self.num_frames == 8:
                    input_indexs = [0,7,15,23,31,39,47,55]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs) 

                elif self.num_frames == 32:
                    input_indexs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
                    target_id =  random.sample(input_indexs, 1)
                    input_indexs = set(input_indexs ) - set(target_id)
                    input_indexs =list(input_indexs)                    
            else:
                input_indexs  = set(random.sample(range(0,64), self.num_frames))
                # we randomly choose a target frame 
                target_id =  random.randint( 64, v_length - 2)
                   
            if type(target_id) == list:
                target_id = target_id[0]
            cap = cv2.VideoCapture(video_path)
            real_video = []
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                    real_video.append(frame)
                    
                else:
                    break
            reference_frames = []
            for t in input_indexs:
                rgb_t =   cv2.cvtColor(real_video[t],cv2.COLOR_BGR2RGB )
                lmark_t = lmark[t]
                lmark_rgb = util.plot_landmarks( lmark_t)
                # resize  to 256
                rgb_t  = cv2.resize(rgb_t, self.output_shape)
                lmark_rgb  = cv2.resize(lmark_rgb, self.output_shape)
                # to tensor
                rgb_t = self.transform(rgb_t)
                lmark_rgb = self.transform(lmark_rgb)
                reference_frames.append(torch.cat([rgb_t, lmark_rgb],0))  # (6, 256, 256)   
            ############################################################################
            target_rgb = real_video[target_id]
            target_lmark = lmark[target_id]
            target_rgb = mmcv.bgr2rgb(target_rgb)
            target_rgb = cv2.resize(target_rgb, self.output_shape)
            target_rgb = self.transform(target_rgb)

        
            target_lmark = util.plot_landmarks(target_lmark)
            target_lmark  = cv2.resize(target_lmark, self.output_shape)
            target_lmark = self.transform(target_lmark)

            reference_frames = torch.cat(reference_frames, dim = 0)
            target_img_path  = os.path.join(video_path[:-4] , '%05d.png'%target_id  )
            input_dic = {'v_id' : target_img_path, 'target_lmark': target_lmark, 'reference_frames': reference_frames, \
            'target_rgb': target_rgb,  'target_id': target_id}# ,  'dif_img': dif_rgb , 'mis_img' :mis_rgb}
            return input_dic
        # except:
        #     return None


class GRID_1D_lstm_landmark(Dataset):
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
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()

    
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
        if self.train == 'train':
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
            lmark = np.load(lmark_path)[:,:,:2]
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
            lmark = torch.FloatTensor(lmark)
            mfcc = np.load(mfcc_path)
            left_append = mfcc[:12]
            right_append = mfcc[-16:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            r =random.choice(
                [x for x in range(0,41)])
            mfccs = []
            for ind in range(self.num_frames):
                t_mfcc =mfcc[(r + ind )*4: (r + ind + 7)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r : r + self.num_frames,:]

            example_landmark = example_landmark.contiguous().view(-1)
            landmark = landmark.contiguous().view( self.num_frames, -1 )

            return example_landmark, landmark, mfccs
        elif self.train =='test':

            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
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
            lmark = torch.FloatTensor(lmark - diff)
            mfcc = np.load(mfcc_path)
            example_landmark =lmark[0,:]  # since the lips in all 0 frames are closed 
           
            left_append = mfcc[:12]
            right_append = mfcc[-16:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            r =random.choice(
                [x for x in range(0,41)])
            mfccs = []
            for ind in range(self.num_frames):
                t_mfcc =mfcc[(r + ind )*4: (r + ind + 7)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            lmark  =lmark[r : r + self.num_frames,:]
            example_landmark = example_landmark.contiguous().view(-1)
            lmark = lmark.contiguous().view(self.num_frames, -1 )

            return example_landmark, lmark, mfccs, lmark_path
        else:
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
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
            lmark = torch.FloatTensor(lmark - diff)
            mfcc = np.load(mfcc_path)
            example_landmark =lmark[0,:]  # since the lips in all 0 frames are closed 
           
            left_append = mfcc[:12]
            right_append = mfcc[-16:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            # r =random.choice(
            #     [x for x in range(0,41)])
            mfccs = []
            for ind in range(75):
                t_mfcc =mfcc[( ind )*4: ( ind + 7)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            # lmark  =lmark[r : r + self.num_frames,:]
            example_landmark = example_landmark.contiguous().view(-1)
            lmark = lmark.contiguous().view(75,  -1 )

            return example_landmark, lmark, mfccs, lmark_path

       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)
class GRID_1D_lstm_pca_landmark(Dataset):
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
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
            self.datalist = pkl.load(_file)
            _file.close()

        self.mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
        self.component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')

# data_original = np.dot(data_reduced,component) + mean
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
        if self.train == 'train':
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
            lmark = np.load(lmark_path)[:,:,:2]
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
            lmark = np.dot(lmark - self.mean, self.component.T)

            lmark = torch.FloatTensor(lmark)
            mfcc = np.load(mfcc_path)
            left_append = mfcc[:12]
            right_append = mfcc[-16:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            r =random.choice(
                [x for x in range(0,41)])
            mfccs = []
            for ind in range(self.num_frames):
                t_mfcc =mfcc[(r + ind )*4: (r + ind + 7)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            landmark  =lmark[r : r + self.num_frames,:]

            # example_landmark = example_landmark.contiguous().view(-1)
            # landmark = landmark.contiguous().view( self.num_frames, -1 )

            return example_landmark, landmark, mfccs
        elif self.train == 'test':
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
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
            lmark = torch.FloatTensor(lmark)
            
            mfcc = np.load(mfcc_path)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
           
            left_append = mfcc[:12]
            right_append = mfcc[-16:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            r =random.choice(
                [x for x in range(0,41)])
            mfccs = []
            for ind in range(self.num_frames):
                t_mfcc =mfcc[(r + ind )*4: (r + ind + 7)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            lmark  =lmark[r : r + self.num_frames,:]
            # example_landmark = example_landmark.contiguous().view(-1)
            # lmark = lmark.contiguous().view(self.num_frames, -1 )

            return example_landmark, lmark, mfccs, lmark_path
        else:
            lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_front.npy') 
            mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
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
            lmark = torch.FloatTensor(lmark)
            
            mfcc = np.load(mfcc_path)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
           
            left_append = mfcc[:12]
            right_append = mfcc[-16:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            
            mfccs = []
            for ind in range(75):
                t_mfcc =mfcc[( ind )*4: ( ind + 7)*4, 1:]
                t_mfcc = torch.FloatTensor(t_mfcc)
                mfccs.append(t_mfcc)
            mfccs = torch.stack(mfccs, dim = 0)
            # example_landmark = example_landmark.contiguous().view(-1)
            # lmark = lmark.contiguous().view(self.num_frames, -1 )

            return example_landmark, lmark, mfccs, lmark_path
       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            return len(self.datalist)

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
            diff_path =  os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][2]) 
            lmark = np.load(lmark_path)[:,:,:2]
            diff = np.load(diff_path)
            reference_id = int(self.datalist[index][2].split('_')[1])
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
            if self.datalist[index][2] == True:
                ss = 0
            else:
                ss = 8
            if self.datalist[index][3] == True:
                ee = 75
            else:
                ee = 68
            r =random.choice(
                [x for x in range(ss, ee)])
            
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
            lmark = torch.FloatTensor(lmark)
            
            fs, mfcc = wavfile.read( audio_path)
            chunck_size =int(fs * 0.04 ) 
           
            left_append = mfcc[: 3 * chunck_size]
            right_append = mfcc[-4 * chunck_size:]
            mfcc = np.insert( mfcc, 0, left_append ,axis=  0)
            mfcc = np.insert( mfcc, -1, right_append ,axis=  0)
            example_landmark =lmark[reference_id,:]  # since the lips in all 0 frames are closed 
            if self.datalist[index][2] == True:
                ss = 0
            else:
                ss = 8
            if self.datalist[index][3] == True:
                ee = 75
            else:
                ee = 68
            r =random.choice(
                [x for x in range(ss, ee)])
            
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
# dataset = GRID_raw_pca_landmark( train='train')
# data_loader = DataLoader(dataset,
#                             batch_size=2,
#                             num_workers=1,
#                             shuffle=False, drop_last=True)
# for i in range (10):
#     for step, (example_landmark, lmark, audio) in enumerate(data_loader):

#         print (example_landmark.shape)
