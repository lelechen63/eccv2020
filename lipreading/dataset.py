import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
# from cvtransforms import *
import torchvision.transforms as transforms

import torch
import glob
import re
import copy
import json
import random
import editdistance
import torchtext
from torchtext.data.utils import get_tokenizer
import pickle
def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1

def prepare_data():
    path ='/mnt/ssd0/dat/lchen63/grid'
    trainset = []
    testset  =[]
    align_path = os.path.join( path , 'align')
    for i in os.listdir(align_path):
        
        for vid in os.listdir( os.path.join(align_path, i ) ):
            if os.path.exists(os.path.join( path , 'data' ,vid[:-6]) ) :
                if  i == 's1' or i == 's2' or i == 's20' or i == 's22':
                    testset.append( [i , vid] )
                else:
                    trainset.append( [i , vid] )
        # break
    print (len(trainset))
    print (len(testset))
    with open(os.path.join(path, 'pickle','train_lipreading_grid.pkl'), 'wb') as handle:
        pickle.dump(trainset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'pickle','test_lipreading_grid.pkl'), 'wb') as handle:
        pickle.dump(testset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

class GRID_character(Dataset):
    def __init__(self, data_root = '', txt_pad = 100 , phase = 'test'):
        self.character_dic = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        for i in range(10):
            self.character_dic.append(str(i))
        self.character_dic.append('\'')
        #need [PAD], [EOS]
        self.character_dic = listToString(self.character_dic)
        self.n_letters = len(self.character_dic)
        self.data_root = data_root
        self.phase = phase
        self.txt_pad = txt_pad
        self.video_path = os.path.join(self.data_root , 'data')
        if self.phase=='train':
            _file = open(os.path.join(self.data_root, 'pickle', "train_lipreading_grid.pkl"), "rb")
            self.datalist = pickle.load(_file)
            _file.close()
        elif self.phase =='test':
            _file = open(os.path.join(self.data_root, 'pickle', "test_lipreading_grid.pkl"), "rb")
            self.datalist = pickle.load(_file)
            _file.close()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    def __getitem__(self, idx):
        video_sample = self.datalist[idx]

        text_sample_path =  os.path.join(self.data_root , 'align', video_sample[0] , video_sample[1]  ) 
        print (text_sample_path)
        ########## fetch align, with padding to pad size
        align = self._load_align(text_sample_path)
        print (align)
        align_tensor = self.sentenceToTensor(align)
        print (align_tensor.shape) 
        align_tensor = self._padding(align_tensor, self.txt_pad)
        print (align_tensor.shape) 
        ##### fetch video frames
        video_sample_path = os.path.join(self.data_root , 'data', video_sample[1][:-6], video_sample[1][:-6]  ) 
        print (video_sample_path)
        video_vec = self._load_vid( video_sample_path )
        print (video_vec.shape)

        sample = {'video': video_vec, 'align' : align_tensor , 'align_path': text_sample_path}

        return sample

    def __len__(self):
        return 1
    def _load_vid(self, p): 
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  for im in array]
        array = [cv2.resize(im, (128, 128), interpolation=cv2.INTER_LANCZOS4) for im in array]

        array = [self.transform(im) for im in array]
        array = torch.stack(array)
        return array
    
    def _load_align(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return ' '.join(txt).upper()

    def sentenceToTensor(self, line):
        tensor = torch.zeros(len(line), self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][self.letterToIndex(letter)] = 1
        return tensor

    def letterToIndex(self, letter):
        return self.character_dic.find(letter)

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    def letterToTensor(self, letter):
        tensor = torch.zeros( self.n_letters)
        tensor[self.letterToIndex(letter)] = 1
        return tensor

 


class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self,data_root = '', txt_pad = 100 , phase ='train'):
        self.data_root = data_root
        self.vid_pad = 75
        self.txt_pad = txt_pad
        self.phase = phase
        self.video_path = os.path.join(self.data_root , 'data')
        if self.phase=='train':
            _file = open(os.path.join(self.data_root, 'pickle', "train_lipreading_grid.pkl"), "rb")
            self.datalist = pickle.load(_file)
            _file.close()
        elif self.phase =='test':
            _file = open(os.path.join(self.data_root, 'pickle', "test_lipreading_grid.pkl"), "rb")
            self.datalist = pickle.load(_file)
            _file.close()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        
        
                
    def __getitem__(self, idx):
        video_sample = self.datalist[idx]
        text_sample_path =  os.path.join(self.data_root , 'align', video_sample[0] , video_sample[1]  ) 
        # print (text_sample_path)
        anno = self._load_anno(text_sample_path)
        
        video_sample_path = os.path.join(self.data_root , 'data', video_sample[1][:-6], video_sample[1][:-6]  ) 

        vid = self._load_vid(video_sample_path)

        # if(self.phase == 'train'):
        #     vid = HorizontalFlip(vid)
          
        # vid = ColorNormalize(vid)                   
        
        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        vid = self.video_padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
        return {'vid': vid.permute( 1, 0,  2 ,3 ), 
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}
            
    def __len__(self):
        return len(self.datalist)
        
    def _load_vid(self, p): 
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  for im in array]
        array = [cv2.resize(im, (128, 128), interpolation=cv2.INTER_LANCZOS4) for im in array]

        array = [self.transform(im) for im in array]
        array = torch.stack(array)
        return array
    
    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = [line[2] for line in lines]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        return MyDataset.txt2arr(' '.join(txt).upper(), 1)
    
    def video_padding(self, tensor, length):
        if tensor.shape[0] == length:
            return tensor
        new_tensor = torch.zeros(length , tensor.shape[1], tensor.shape[2] , tensor.shape[3] )
        new_tensor[:tensor.shape[0]] = tensor
        return new_tensor

    def _padding(self, array, length):

        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
        
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()
            
    @staticmethod
    def wer(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer


# prepare_data()
# dataset = GRID_character(data_root= '/mnt/ssd0/dat/lchen63/grid')
# sample = dataset[0]
# def dataset2dataloader(dataset, num_workers=1, shuffle=True):
#     return DataLoader(dataset,
#         batch_size = 1, 
#         shuffle = shuffle,
#         num_workers = num_workers,
#         drop_last = True)

