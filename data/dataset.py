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

from data.base_dataset import BaseDataset, get_transform
from data.keypoint2img import interpPoints, drawEdge
from PIL import Image



class VoxLmark2rgbDataset(BaseDataset):

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
        self.output_shape = tuple([opt.loadSize, opt.loadSize])
        self.num_frames = opt.num_frames
        self.n_frames_total = opt.n_frames_G
        self.opt = opt
        self.root  = opt.dataroot
        self.fix_crop_pos = True

        # mapping from keypoints to face part 
        self.add_upper_face = not opt.no_upper_face
        self.part_list = [[list(range(0, 17)) + ((list(range(68, 83)) + [0]) if self.add_upper_face else [])], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48], range(60, 65), [64,65,66,67,60]], # mouth and tongue
                    ]
       
        if opt.isTrain:
            _file = open(os.path.join(self.root, 'pickle','dev_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()
        else :
            _file = open(os.path.join(self.root, 'pickle','test_lmark2img.pkl'), "rb")
            self.data = pkl.load(_file)
            _file.close()

        if opt.isTrain:
            self.video_bag = 'unzip/dev_video'
        else:
            self.video_bag = 'unzip/test_video'
        print (len(self.data))
        
        # get transform for image and landmark
        img_params = self.get_img_params(self.output_shape)
        if opt.isTrain:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BICUBIC)),
                transforms.Lambda(lambda img: self.__color_aug(img, img_params['color_aug'])),
                transforms.Lambda(lambda img: self.__flip(img, img_params['flip'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BICUBIC)),
                transforms.Lambda(lambda img: self.__flip(img, img_params['flip'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

        self.transform_L = transforms.Compose([
            transforms.Lambda(lambda img: self.__scale_image(img, img_params['new_size'], Image.BILINEAR)),
            transforms.Lambda(lambda img: self.__flip(img, img_params['flip'])),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.data) 

    def __color_aug(self, img, params):
        h, s, v = img.convert('HSV').split()    
        h = h.point(lambda i: (i + params[0]) % 256)
        s = s.point(lambda i: min(255, max(0, i * params[1] + params[2])))
        v = v.point(lambda i: min(255, max(0, i * params[3] + params[4])))
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        return img

    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
        
    def __scale_image(self, img, size, method=Image.BICUBIC):
        w, h = size    
        return img.resize((w, h), method)

    def name(self):
        return 'VoxLmark2rgbDataset'

    def __getitem__(self, index):
        if self.opt.dataset_name == 'face':
            video_path = self.data[index][1] #os.path.join(self.root, 'pretrain', v_id[0] , v_id[1][:5] + '_crop.mp4'  )
            lmark_path = self.data[index][0]  #= os.path.join(self.root, 'pretrain', v_id[0] , v_id[1]  )
        elif self.opt.dataset_name == 'vox':
            paths = self.data[index]
            video_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned.mp4")
            lmark_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned.npy")
            ani_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned_ani.mp4")
            rt_path = os.path.join(self.root, self.video_bag, paths[0], paths[1], paths[2]+"_aligned_rt.npy")
        # read in data
        lmarks = np.load(lmark_path)#[:,:,:-1]
        real_video = self.read_videos(video_path)
        v_length = len(real_video)
        ani_video = self.read_videos(ani_path)
        # print ('+++++++++', len(ani_video))
        rt = np.load(rt_path)[:,:3]
        # sample index of frames for embedding network
        input_indexs, target_id = self.get_image_index(self.n_frames_total, v_length)

        # define scale
        self.define_scale()

        # get reference
        ref_images, ref_lmarks = self.prepare_datas(real_video, lmarks, input_indexs)
        

        # get target
        [tgt_images , ani_images], [tgt_lmarks,_] = self.prepare_datas([real_video, ani_video], lmarks, target_id)

        # get animation
        # ani_images, _ = self.prepare_datas(ani_video, lmarks, target_id)
        
        # get warping reference
        reference_rt_diffs = []
        target_rt = rt[target_id]
        for t in input_indexs:
            reference_rt_diffs.append( rt[t] - target_rt )
        reference_rt_diffs = np.mean(np.absolute(reference_rt_diffs), axis =1)
        similar_id  = np.argmin(reference_rt_diffs)

        warping_refs, warping_ref_lmarks = self.prepare_datas(real_video, lmarks, [similar_id])

        target_img_path  = [os.path.join(video_path[:-4] , '%05d.png'%t_id) for t_id in target_id]

        ref_images = torch.cat([ref_img.unsqueeze(0) for ref_img in ref_images], 0)
        ref_lmarks = torch.cat([ref_lmark.unsqueeze(0) for ref_lmark in ref_lmarks], 0)
        tgt_images = torch.cat([tgt_img.unsqueeze(0) for tgt_img in tgt_images], 0)
        tgt_lmarks = torch.cat([tgt_lmark.unsqueeze(0) for tgt_lmark in tgt_lmarks], 0)
        warping_refs = torch.cat([warping_ref.unsqueeze(0) for warping_ref in warping_refs], 0)
        warping_ref_lmarks = torch.cat([warping_ref_lmark.unsqueeze(0) for warping_ref_lmark in warping_ref_lmarks], 0)
        ani_images = torch.cat([ani_image.unsqueeze(0) for ani_image in ani_images], 0)


        # print (tgt_lmarks.shape)   # 1, 1 , 256,256
        # print (ref_images.shape) # 8, 3 , 256,256
        # print (tgt_images.shape) ## 1, 3 , 256,256
        # print (ref_lmarks.shape)  # 8, 1 , 256,256
        # print (warping_refs.shape) # 1, 3 , 256,256
        # print (warping_ref_lmarks.shape)   # 1, 1 , 256,256
        # print (ani_images.shape)  # 1, 3, 256,256
        input_dic = {'v_id' : target_img_path, 'tgt_label': tgt_lmarks, 'ref_image':ref_images , 'ref_label': ref_lmarks, \
        'tgt_image': tgt_images,  'target_id': target_id , 'warping_ref' : warping_refs , 'warping_ref_lmark' : warping_ref_lmarks , 'ani_image' : ani_images }

        return input_dic


    # get index for target and reference
    def get_image_index(self, n_frames_total, cur_seq_len, max_t_step=4):            
        n_frames_total = min(cur_seq_len, n_frames_total)             # total number of frames to load
        max_t_step = min(max_t_step, (cur_seq_len-1) // max(1, (n_frames_total-1)))        
        t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames                
        
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible frame index for the first frame

        start_idx = np.random.randint(offset_max)                 # offset for the first frame to load    

        # indices for target
        target_ids = [start_idx + step * t_step for step in range(self.n_frames_total)]

        # indices for reference frames
        if self.opt.isTrain:
            max_range, min_range = 300, 14                            # range for possible reference frames
            ref_range = list(range(max(0, start_idx - max_range), max(1, start_idx - min_range))) \
                    + list(range(min(start_idx + min_range, cur_seq_len - 1), min(start_idx + max_range, cur_seq_len)))
            ref_indices = np.random.choice(ref_range, size=self.num_frames)   
        else:
            ref_indices = self.opt.ref_img_id.split(',')
            ref_indices = [int(i) for i in ref_indices]

        return ref_indices, target_ids

    # load in all frames from video
    def read_videos(self, video_path):
        cap = cv2.VideoCapture(video_path)
        real_video = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                real_video.append(frame)
            else:
                break

        return real_video

    # plot landmarks
    def get_face_image(self, keypoints, transform_L, size, bw):   
        w, h = size
        edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
        # edge map for face region from keypoints
        im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
        for edge_list in self.part_list:
            for edge in edge_list:
                for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i+edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]
                                    
                    curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape                    
                    drawEdge(im_edges, curve_x, curve_y, bw=1)        
        input_tensor = transform_L(Image.fromarray(im_edges))
        return input_tensor

    # preprocess for landmarks
    def get_keypoints(self, keypoints, transform_L, size, crop_coords, bw):
        # crop landmarks
        keypoints[:, 0] -= crop_coords[2]
        keypoints[:, 1] -= crop_coords[0]

        # add upper half face by symmetry
        if self.add_upper_face:
            pts = keypoints[:17, :].astype(np.int32)
            baseline_y = (pts[0,1] + pts[-1,1]) / 2
            upper_pts = pts[1:-1,:].copy()
            upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
            keypoints = np.vstack((keypoints, upper_pts[::-1,:])) 

        # get image from landmarks
        lmark_image = self.get_face_image(keypoints, transform_L, size, bw)

        return lmark_image

    # preprocess for image
    def get_image(self, image, transform_I, size, crop_coords):
        # crop
        img = mmcv.bgr2rgb(image)
        img = self.crop(Image.fromarray(img), crop_coords)
        crop_size = img.size

        # transform
        img = transform_I(img)

        return img, crop_size

    # get scale for random crop
    def define_scale(self, scale_max = 0.2):
        self.scale = [np.random.uniform(1 - scale_max, 1 + scale_max), 
                        np.random.uniform(1 - scale_max, 1 + scale_max)]    

    # get image and landmarks
    def prepare_datas(self, imagess, lmarks, choice_ids):
        # get cropped coordinates
        crop_lmark = lmarks[choice_ids[0]]
        crop_coords = self.get_crop_coords(crop_lmark)
        bw = max(1, (crop_coords[1]-crop_coords[0]) // 256)

        # get images and landmarks
        # print (type(imagess[0]))
        if type(imagess[0]) == list:
            tmp_lmarks = []
            tmp_images = []
            for images in imagess:
                result_lmarks = []
                result_images = []
                for choice in choice_ids:
                    image, crop_size = self.get_image(images[choice], self.transform, self.output_shape, crop_coords)
                    lmark = self.get_keypoints(lmarks[choice], self.transform_L, crop_size, crop_coords, bw)

                    result_lmarks.append(lmark)
                    result_images.append(image)
                tmp_images.append(result_images)
                tmp_lmarks.append(result_lmarks)
            return tmp_images, tmp_lmarks
        else:
            images = imagess
            result_lmarks = []
            result_images = []
            for choice in choice_ids:
                image, crop_size = self.get_image(images[choice], self.transform, self.output_shape, crop_coords)
                lmark = self.get_keypoints(lmarks[choice], self.transform_L, crop_size, crop_coords, bw)

                result_lmarks.append(lmark)
                result_images.append(image)
            return result_images, result_lmarks
    # get crop standard from one landmark
    def get_crop_coords(self, keypoints, crop_size=None):           
        min_y, max_y = int(keypoints[:,1].min()), int(keypoints[:,1].max())
        min_x, max_x = int(keypoints[:,0].min()), int(keypoints[:,0].max())
        x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2                
        w = h = (max_x - min_x)
        if crop_size is not None:
            h, w = crop_size[0] / 2, crop_size[1] / 2
        if self.opt.isTrain and self.fix_crop_pos:
            offset_max = 0.2
            offset = [np.random.uniform(-offset_max, offset_max), 
                      np.random.uniform(-offset_max, offset_max)]             
            w *= self.scale[0]
            h *= self.scale[1]
            x_cen += int(offset[0]*w)
            y_cen += int(offset[1]*h)
                        
        min_x = x_cen - w
        min_y = y_cen - h*1.25
        max_x = min_x + w*2        
        max_y = min_y + h*2

        return int(min_y), int(max_y), int(min_x), int(max_x)

    def get_img_params(self, size):
        w, h = size
        
        # for color augmentation
        h_b = random.uniform(-30, 30)
        s_a = random.uniform(0.8, 1.2)
        s_b = random.uniform(-10, 10)
        v_a = random.uniform(0.8, 1.2)
        v_b = random.uniform(-10, 10)    
        
        flip = random.random() > 0.5
        return {'new_size': (w, h), 'flip': flip, 
                'color_aug': (h_b, s_a, s_b, v_a, v_b)}

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

