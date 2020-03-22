import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
from face_tracker import _crop_video ,crop_image
from utils import face_utils, util
from scipy.spatial.transform import Rotation 
from scipy.io import wavfile
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R


# from .dp2model import load_model
# from .dp2dataloader import SpectrogramParser

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_id",
                     type=int,
                     default=1)
    
    return parser.parse_args()
config = parse_args()




def read_videos( video_path):
    cap = cv2.VideoCapture(video_path)
    real_video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            real_video.append(frame)
        else:
            break

    return real_video

def landmark_extractor(img_path):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')
    fa2 = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:0')
    x_list, y_list, dis_list, videos, multi_face_times = crop_image(img_path)
    dis = np.mean(dis_list)
    print (dis)
    top_left_x = x_list - (80 * dis / 90)
    top_left_y = y_list - (100* dis / 90)
    side_length = int((205 * dis / 90))
    print (x_list, y_list, dis_list)
    print  (top_left_x, top_left_y, side_length)
    for i in tqdm(range(x_list.shape[0])):
        if top_left_x[i] < 0 or top_left_y[i] < 0:
            
            img_size = videos[i].shape
            tempolate = np.ones((img_size[0] * 2, img_size[1]* 2 , 3), np.uint8) * 255
            tempolate_middle  = [int(tempolate.shape[0]/2), int(tempolate.shape[1]/2)]
            middle = [int(img_size[0]/2), int(img_size[1]/2)]
            tempolate[tempolate_middle[0]  -middle[0]:tempolate_middle[0]  -middle[0] +img_size[0] , tempolate_middle[1]-middle[1]:tempolate_middle[1]-middle[1]+img_size[1], :] = videos[i]
            top_left_x[i] = top_left_x[i] + tempolate_middle[0]  -middle[0]
            top_left_y[i] = top_left_y[i] + tempolate_middle[1]  -middle[1]
            roi = tempolate[int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
            roi =cv2.resize(roi,(256,256))
            cv2.imwrite(img_path[:-4] +'_crop.png', roi)
        else:
            roi = videos[i][int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
            roi =cv2.resize(roi,(256,256))
            cv2.imwrite(img_path[:-4] +'_crop.png', roi)

    frame = cv2.imread(img_path[:-4] +'_crop.png')
   
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )

    preds = fa.get_landmarks(frame)[0]
          
    np.save(img_path[:-4] +'_original.npy', preds)


    preds = fa2.get_landmarks(frame)[0]
          
    np.save(img_path[:-4] +'_original_2d.npy', preds)


            
        

def RT_compute_single(id ='mulan2'):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]

    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)

    lmark_path = '/home/cxu-serve/p1/common/demo/' + id + '_original.npy'   
    rt_path =   '/home/cxu-serve/p1/common/demo/' + id +  '_original_rt.npy'

    flip_lmark_path = '/home/cxu-serve/p1/common/demo/' + id + '_original_flip.npy'   
    flip_rt_path =   '/home/cxu-serve/p1/common/demo/' + id +  '_original_rt_flip.npy'


    front_path =  '/home/cxu-serve/p1/common/demo/'+ id + '_original_front.npy'
    lmark = np.load(lmark_path)
    lmark = lmark.reshape(1,68,3)
    ############################################## smooth the landmark
    length = lmark.shape[0] 
    lmark_part = np.zeros((length,len(consider_key),3))
    RTs =  np.zeros((length,6))
    frontlized =  np.zeros((length,68,3))
    for j in range(length ):
        for m in range(len(consider_key)):
            lmark_part[:,m] = lmark[:,consider_key[m]] 

        target = np.mat(lmark_part[j])
        ret_R, ret_t = face_utils.rigid_transform_3D( target, source)

        source_lmark  = np.mat(lmark[j])

        A2 = ret_R*source_lmark.T
        A2+= np.tile(ret_t, (1, 68))
        A2 = A2.T
        frontlized[j] = A2
        r = Rotation.from_dcm(ret_R)
        vec = r.as_rotvec()             
        RTs[j,:3] = vec
        RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
    np.save(rt_path, RTs)
    np.save(front_path, frontlized)
    print (front_path)

    ##############################################  flipped version
    flipped  = lmark
    flipped[:,:,0] = 256 - lmark[:,:,0] 
    length = lmark.shape[0] 
    lmark_part = np.zeros((length,len(consider_key),3))
    RTs =  np.zeros((length,6))
    frontlized =  np.zeros((length,68,3))
    for j in range(length ):
        for m in range(len(consider_key)):
            lmark_part[:,m] = flipped[:,consider_key[m]] 
        target = np.mat(lmark_part[j])
        ret_R, ret_t = face_utils.rigid_transform_3D( target, source)

        source_lmark  = np.mat(flipped[j])

        A2 = ret_R*source_lmark.T
        A2+= np.tile(ret_t, (1, 68))
        A2 = A2.T
        frontlized[j] = A2
        r = Rotation.from_dcm(ret_R)
        vec = r.as_rotvec()             
        RTs[j,:3] = vec
        RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
    np.save(flip_rt_path, RTs)
    np.save(flip_lmark_path, flipped)
    print (front_path)
        # break
    # break

def RT_compute():
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    root_path = '/home/cxu-serve/p1/common/lrw'
    train_list = sorted(os.listdir( os.path.join(root_path,  'video' ) ))
    batch_length = int( len(train_list))
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    for i in tqdm(range(batch_length)):
        p_id = train_list[i]
        for jj in range(1,51):
            lmark_path = os.path.join(root_path,  'video',  p_id, 'test', p_id +'_%05d_original.npy'%jj)   
            rt_path = os.path.join(root_path,  'video',  p_id, 'test', p_id +'_%05d_rt.npy'%jj)  
            front_path =  os.path.join(root_path,  'video',  p_id, 'test', p_id +'_%05d_front.npy'%jj)

            if os.path.exists(front_path):
                continue
            if not os.path.exists(lmark_path):
                continue
            lmark = np.load(lmark_path)
            ############################################## smooth the landmark
        
            length = lmark.shape[0] 
            lmark_part = np.zeros((length,len(consider_key),3))
            RTs =  np.zeros((length,6))
            frontlized =  np.zeros((length,68,3))
            for j in range(length ):
                for m in range(len(consider_key)):
                    lmark_part[:,m] = lmark[:,consider_key[m]] 

                target = np.mat(lmark_part[j])
                ret_R, ret_t = face_utils.rigid_transform_3D( target, source)

                source_lmark  = np.mat(lmark[j])

                A2 = ret_R*source_lmark.T
                A2+= np.tile(ret_t, (1, 68))
                A2 = A2.T
                frontlized[j] = A2
                r = Rotation.from_dcm(ret_R)
                vec = r.as_rotvec()             
                RTs[j,:3] = vec
                RTs[j,3:] =  np.squeeze(np.asarray(ret_t))            
            np.save(rt_path, RTs)
            np.save(front_path, frontlized)
        print (front_path)
                # break
            # break
import torch
import random
from sklearn.decomposition import PCA
from utils import face_utils

def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append(np.absolute(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2]))
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean()
import pickle as pkl


def interpolate_features(features, input_rate, output_rate, output_len=None):
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps,
                                             input_timestamps,
                                             features[:, feat])
    return output_features

def parse_audio(audio, audio_parser, model, device):
    audio_spect = audio_parser.parse_audio(audio).contiguous()
    audio_spect = audio_spect.view(1, 1, audio_spect.size(0), audio_spect.size(1))
    audio_spect = audio_spect.to(device)
    input_sizes = torch.IntTensor([audio_spect.size(3)]).int()
    parsed_audio, output_sizes = model(audio_spect, input_sizes)

    # audio (124667, ), audio_spect (1, 1, 161, 780), parsed_audio (1, 390, 29)
    return parsed_audio, output_sizes



def get_front():
    root = '/home/cxu-serve/p1/common/lrw'
    _file = open(os.path.join(root, 'pickle','test_lmark2img.pkl'), "rb")
    data = pkl.load(_file)
    _file.close()
    new_data = []
    for index in tqdm(range(len(data))):
        try:
            
            video_path = data[index][0] + '_crop.mp4'
            v_frames = read_videos(video_path)
            lmark_path =  data[index][0] + '_rt.npy' 
            rt = np.load(lmark_path)
            lmark_length = rt.shape[0]
            find_rt = []
            for t in range(0, lmark_length):
                find_rt.append(sum(np.absolute(rt[t,:3])))
            find_rt = np.asarray(find_rt)

            min_index = np.argmin(find_rt)
            
            img_path =  data[index][0] + '_%05d.png'%min_index  
            cv2.imwrite(img_path, v_frames[min_index])
            data[index].append(min_index)
            new_data.append(data[index])
        except:
            print (video_path)
            continue
    # data = data[:index]
    print (len(new_data))
    with open(os.path.join( root, 'pickle','test2_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(new_data, handle, protocol=pkl.HIGHEST_PROTOCOL)

def diff():
    root_path  = '/home/cxu-serve/p1/common/CREMA'
    _file = open(os.path.join(root_path, 'pickle','train_lmark2img.pkl'), "rb")
    datalist = pkl.load(_file)
    _file.close()
    batch_length = int( len(datalist))
    landmarks = []
    k = 20
    norm_lmark = np.load('../basics/s1_pgbk6n_01.npy')[:,:2]
   
    for index in tqdm(range(batch_length)):
        lmark_path = os.path.join(root_path,  'VideoFlash', datalist[index][0][:-10] +'_front.npy'  )
        lmark = np.load(lmark_path)[:,:,:2]


        openrates = []
        for  i in range(lmark.shape[0]):
            openrates.append(openrate(lmark[i]))
        openrates = np.asarray(openrates)
        min_index = np.argmin(openrates)
        diff =  lmark[min_index] - norm_lmark
        np.save(lmark_path[:-10] +'_%05d_diff.npy'%(min_index) , diff)
        datalist[index].append(min_index) 
    
    with open(os.path.join(root_path, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(datalist, handle, protocol=pkl.HIGHEST_PROTOCOL)
# get_front()
# pca_lmark_grid()
# deepspeech_grid()
# pca_3dlmark_grid()
# data = np.load('/home/cxu-serve/p1/common/grid/align/s1/lwae8n_front.npy')[:,:,:2]
# data = data.reshape(data.shape[0], 136)
# print (data.shape)
# mean = np.load('../basics/mean_grid_front.npy')
# component = np.load('../basics/U_grid_front.npy')
# data_reduced = np.dot(data - mean, component.T)
# RT_compute_single()
# data_original = np.dot(data_reduced,component) + mean
# np.save( 'gg.npy', data_original )
# print (data - data_original)
# landmark_extractor()
# generatert()
# RT_compute()
# diff()
img_path = '/home/cxu-serve/p1/common/demo/self2.jpg'
landmark_extractor(img_path)
RT_compute_single('self2')

