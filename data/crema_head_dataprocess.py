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
import torch

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
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')
    root_path = '/home/cxu-serve/p1/common/CREMA'
    train_list = sorted(os.listdir(os.path.join(root_path, 'VideoFlash')))
    batch_length =  int(0.2 * len(train_list))
    for i in tqdm(range(batch_length * (config.batch_id -1), batch_length * (config.batch_id ))):
        p_id = train_list[i]
        original_video_path = os.path.join(root_path,  'VideoFlash',  p_id)
        lmark_path = os.path.join(root_path,  'VideoFlash',  p_id[:-4] + '__original.npy')            
        print (original_video_path)
        cropped_video_path = os.path.join(root_path,  'VideoFlash',   p_id[:-4] + '__crop.mp4')
        
        if os.path.exists(lmark_path):
            continue
        try:
            _crop_video(original_video_path, config.batch_id)
            
            command = 'ffmpeg -framerate 25  -i ./temp%05d'%config.batch_id + '/%05d.png  -vcodec libx264  -vf format=yuv420p -y ' +  cropped_video_path
            os.system(command)
            cap = cv2.VideoCapture(cropped_video_path)
            lmark = []
            while(cap.isOpened()):
                # counter += 1 
                # if counter == 5:
                #     break
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

def RT_compute():
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    root_path = '/home/cxu-serve/p1/common/CREMA'
    train_list = sorted(os.listdir( os.path.join(root_path,  'VideoFlash' ) ))
    batch_length = int( len(train_list))
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    for i in tqdm(range(batch_length)):
        p_id = train_list[i]
        if p_id[-3:] !=  'flv':
            continue
        lmark_path = os.path.join(root_path,  'VideoFlash',  p_id[:-4] + '__original.npy')  
        
        rt_path = os.path.join( root_path,"VideoFlash" , p_id[:-4] +'__rt.npy')
        front_path = os.path.join(  root_path, "VideoFlash" , p_id[:-4] +'__front.npy')
        # normed_path  = os.path.join( person_path,vid[:-12] +'normed.npy')
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
def pca_lmark_grid():
    root_path  ='/home/cxu-serve/p1/common/grid'
    _file = open(os.path.join(root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
    datalist = pkl.load(_file)
    _file.close()
    batch_length = int( len(datalist))
    landmarks = []
    k = 20
    norm_lmark = np.load('../basics/s1_pgbk6n_01.npy')
   
    for index in tqdm(range(batch_length)):
        # if index == 10:
        #     break
        lmark_path = os.path.join(root_path ,  'align' , datalist[index][0] , datalist[index][1] + '_front.npy') 
        lmark = np.load(lmark_path)[:,:,:2]
        # if lmark.shape[0]< 74:
        #     continue

        openrates = []
        for  i in range(lmark.shape[0]):
            openrates.append(openrate(lmark[i]))
        openrates = np.asarray(openrates)
        min_index = np.argmin(openrates)
        diff =  lmark[min_index] - norm_lmark
        np.save(lmark_path[:-10] +'_%05d_diff.npy'%(min_index) , diff)
        datalist[index].append(min_index) 
    #     lmark = lmark - diff
    #     if datalist[index][2] == True: 
    #         indexs = random.sample(range(0,10), 6)
    #         for i in indexs:
    #             landmarks.append(lmark[i])
    #     if datalist[index][3] == True: 
    #         indexs = random.sample(range(65,74), 6)
    #         for i in indexs:
    #             landmarks.append(lmark[i])

    #     indexs = random.sample(range(11,65), 10)
    #     for i in indexs:
    #         landmarks.append(lmark[i])
       
    # landmarks = np.stack(landmarks)
    # print (landmarks.shape)
    # landmarks = landmarks.reshape(landmarks.shape[0], 136)
    # pca = PCA(n_components=20)
    # pca.fit(landmarks)
    
    # np.save('../basics/mean_grid_front.npy', pca.mean_)
    # np.save('../basics/U_grid_front.npy',  pca.components_)
    with open(os.path.join(root_path, 'pickle','test_audio2lmark_grid.pkl'), 'wb') as handle:
        pkl.dump(datalist, handle, protocol=pkl.HIGHEST_PROTOCOL)


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


def deepspeech_grid():
    device = 'cuda:0'
    model = load_model(device, '/u/lchen63/voca/deepspeech_pytorch/models/deepspeech.pth', False)
    model.eval()
    audio_parser = SpectrogramParser(model.audio_conf, normalize=True)
    root_path  ='/home/cxu-serve/p1/common/grid'
    _file = open(os.path.join(root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
    datalist = pkl.load(_file)
    _file.close()
    batch_length = int( len(datalist))
   
    for index in tqdm(range(batch_length)):
        audio_path = os.path.join( root_path , 'audio' ,datalist[index][0],  datalist[index][1] +'.wav' )
        sample_rate, audio_sample = wavfile.read( audio_path)
        audio_parser = SpectrogramParser(model.audio_conf, normalize=True)
        parsed_audio, output_sizes = parse_audio(audio_sample, audio_parser, model, device)
        audio_len_s = float(audio_sample.shape[0]) / sample_rate
        num_frames = int(round(audio_len_s * 25))
        network_output = interpolate_features(parsed_audio.data[0].cpu().numpy(), 25, 25,
                                                    output_len=num_frames)
        # print (network_output.shape)
        # print  (network_output)
        np.save(audio_path[:-4] +'_dp.npy' , network_output )
        # break




def get_front():
    root = '/home/cxu-serve/p1/common/CREMA'
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
    data = pkl.load(_file)
    _file.close()
    for index in tqdm(range(len(data))):
        v_id = data[index]
        video_path = os.path.join(root, 'VideoFlash', v_id[0][:-10] + '_crop.mp4'  )
            # mis_video_path = os.path.join(self.root, 'pretrain', mis_vid[0] , mis_vid[1][:5] + '_crop.mp4'  )
        v_frames = read_videos(video_path)
        lmark_path = os.path.join(root,  'VideoFlash', v_id[0][:-10] +'_rt.npy'  )
        rt = np.load(lmark_path)
        lmark_length = rt.shape[0]
        find_rt = []
        for t in range(0, lmark_length):
            find_rt.append(sum(np.absolute(rt[t,:3])))
        find_rt = np.asarray(find_rt)

        min_index = np.argmin(find_rt)
        
        img_path =  os.path.join(root,  'VideoFlash', v_id[0][:-10] + '_%05d.png'%min_index  )
        cv2.imwrite(img_path, v_frames[min_index])
        data[index].append(min_index)
    with open(os.path.join( root, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

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

# data_original = np.dot(data_reduced,component) + mean
# np.save( 'gg.npy', data_original )
# print (data - data_original)
# landmark_extractor()
# 
# RT_compute()
diff()
# landmark_extractor()
