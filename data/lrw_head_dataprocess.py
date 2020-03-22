import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
from face_tracker import _crop_video
from utils import face_utils, util
from scipy.spatial.transform import Rotation 
from scipy.io import wavfile

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

def landmark_extractor():
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    root_path = '/home/cxu-serve/p1/common/lrw'
    train_list = sorted(os.listdir( os.path.join(root_path,  'video' ) ))
    batch_length = int( 0.1 * len(train_list))
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')
    root_path = '/home/cxu-serve/p1/common/lrw'
    # train_list = sorted(os.listdir(os.path.join(root_path, 'video')))
    # batch_length =  int(1 * len(train_list))
    for i in tqdm(range(batch_length * (config.batch_id -1), batch_length * (config.batch_id ))):
        p_id = train_list[i]
        
        for jj in range(1,21):
            original_video_path = os.path.join(root_path,  'video',  p_id, 'test', p_id +'_%05d.mp4'%jj)    
            lmark_path = os.path.join(root_path,  'video',  p_id, 'test', p_id +'_%05d_original.npy'%jj)   
            cropped_video_path =  os.path.join(root_path,  'video',  p_id, 'test', p_id +'_%05d_crop.mp4'%jj)  
            # print (fffs)
            if not os.path.exists(lmark_path):
                try:
                    _crop_video(original_video_path, config.batch_id)
                    
                    command = 'ffmpeg -framerate 25  -i ./temp%05d'%config.batch_id + '/%05d.png  -vcodec libx264  -vf format=yuv420p -y ' +  cropped_video_path
                    os.system(command)
                    cap = cv2.VideoCapture(cropped_video_path)
                    lmark = []
                    while(cap.isOpened()):
                        
                        ret, frame = cap.read()
                        if ret == True:
                            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )

                            preds = fa.get_landmarks(frame)[0]
                            lmark.append(preds)
                        else:
                            break
                            
                    lmark = np.asarray(lmark)
                    np.save(lmark_path, lmark)
                except:
                    print (cropped_video_path)

                    continue
            if os.path.exists(lmark_path):
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


def check():
    root = '/home/cxu-serve/p1/common/lrw'
    _file = open(os.path.join(root, 'pickle','test2_lmark2img.pkl'), "rb")
    data = pkl.load(_file)
    _file.close()
    new_data = []
    for index in tqdm(range(len(data))):
            
        if os.path.exists(data[index][0] + '_ani.mp4') :
           
            new_data.append(data[index])
        
    # data = data[:index]
    print (len(new_data))
    with open(os.path.join( root, 'pickle','test3_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(new_data, handle, protocol=pkl.HIGHEST_PROTOCOL)

def diff():
    root_path  = '/home/cxu-serve/p1/common/lrw'
    _file = open(os.path.join(root_path, 'pickle','test3_lmark2img.pkl'), "rb")
    datalist = pkl.load(_file)
    _file.close()
    batch_length = int( len(datalist))
    landmarks = []
    k = 20
    norm_lmark = np.load('../basics/s1_pgbk6n_01.npy')[:,:2]
    for index in tqdm(range(batch_length)):
        lmark_path = datalist[index][0] + '_front.npy'
        lmark = np.load(lmark_path)[:,:,:2]


        openrates = []
        for  i in range(lmark.shape[0]):
            openrates.append(openrate(lmark[i]))
        openrates = np.asarray(openrates)
        min_index = np.argmin(openrates)
        diff =  lmark[min_index] - norm_lmark
        np.save(lmark_path[:-10] +'_%05d_diff.npy'%(min_index) , diff)
        datalist[index].append(min_index) 
    
    with open(os.path.join(root_path, 'pickle','test3_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(datalist, handle, protocol=pkl.HIGHEST_PROTOCOL)
# get_front()
# check()
# pca_lmark_grid()
# deepspeech_grid()
# pca_3dlmark_grid()
# data = np.load('/home/cxu-serve/p1/common/grid/align/s1/lwae8n_front.npy')[:,:,:2]
# data = data.reshape(data.shape[0], 136)
# print (data.shape)
# mean = np.load('../basics/mean_grid_front.npy')
# component = np.load('../basics/U_grid_front.npy')
# data_reduced = np.dot(data - mean, component.T)

# data_original = np.dot(data_reduced,component) + mean
# np.save( 'gg.npy', data_original )
# print (data - data_original)
# landmark_extractor()
# 
# RT_compute()
diff()
# landmark_extractor()
