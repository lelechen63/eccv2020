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

            lmark_path = os.path.join(self.root, 'pretrain', v_id[0] , v_id[1]  )

            lmark = np.load(lmark_path)[:,:,:-1]
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
            print (len(real_video) , len(lmark))
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


class GRID_1D_lstm_landmark(Dataset):
    def __init__(self,
                 train='train'):
        self.train = train
        self.num_frames = 32
        self.root_path = '/home/cxu-serve/p1/common/grid'
        
        if self.train=='train':
            _file = open(os.path.join(self.root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
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

    
    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
            # try:
                lmark_path = os.path.join(self.root_path ,  'align' , self.datalist[index][0] , self.datalist[index][1] + '_original.npy') 
                mfcc_path = os.path.join(self.root_path, 'mfcc' , self.datalist[index][0],  self.datalist[index][1] +'_mfcc.npy') 
                lmark = np.load(lmark_path)[:,:,:-1]
                
                for i in range(lmark.shape[1]):
                    x = lmark[: , i,0]
                    x = face_utils.smooth(x, window_len=5)
                    lmark[: ,i,0 ] = x[2:-2]
                    y = lmark[:, i, 1]
                    y = face_utils.smooth(y, window_len=5)
                    lmark[: ,i,1  ] = y[2:-2] 
                lmark = torch.FloatTensor(lmark)
                mfcc = np.load(mfcc_path)
                example_landmark =lmark[0,:]  # since the lips in all 0 frames are closed 
                # r = random.choice(
                #     [x for x in range(3,40)])
                r = 35
                mfccs = []
                for ind in range(self.num_frames):
                    t_mfcc =mfcc[(r + ind - 3)*4: (r + ind + 4)*4, 1:]
                    t_mfcc = torch.FloatTensor(t_mfcc)
                    mfccs.append(t_mfcc)
                mfccs = torch.stack(mfccs, dim = 0)
                landmark  =lmark[r : r + self.num_frames,:]

                example_landmark = example_landmark.contiguous().view(-1)
                landmark = landmark.contiguous().view( self.num_frames, -1 )

                return example_landmark, landmark, mfccs
            # except:
            #     self.__getitem__(index + 1)
       
       
    def __len__(self):
        if self.train=='train':
            return len(self.datalist)
        elif self.train=='test':
            return len(self.datalist)
        else:
            pass

# dataset = GRID_1D_lstm_landmark( train='train')
# data_loader = DataLoader(dataset,
#                             batch_size=2,
#                             num_workers=1,
#                             shuffle=False, drop_last=True)
# for i in range (1000):
#     for step, (example_landmark, lmark, audio) in enumerate(data_loader):

#         print (example_landmark.shape)