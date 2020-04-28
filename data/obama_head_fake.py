import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
from face_tracker import _crop_video
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

def landmark_extractor(method = 'baseline'):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')
    # root_path = '/home/cxu-serve/p1/common/Obama'
    # train_list = sorted(os.listdir(os.path.join(root_path, 'video')))
    # batch_length =  int(0.2 * len(train_list))
    # for i in tqdm(range(batch_length * (config.batch_id -1), batch_length * (config.batch_id ))):
    #     p_id = train_list[i]
    #     if 'crop' in p_id or p_id[-3:] == 'npy':
    #         continue
    
    if not os.path.exists(os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  method , 'tmp')):
        os.mkdir(os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  method , 'tmp'))

    original_video_path = os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  method,  'test.mp4')

    cropped_video_path = original_video_path[:-4] +'_crop.mp4'


    lmark_path = cropped_video_path[:-3] + 'npy'           
    print (original_video_path)
    
    # frames = read_videos(original_video_path)
    # print (len(frames))
    # for i in range(len(frames)):
    #     fake_frame = frames[i][ :, 256 *2 :256 *3 ]
    #     cv2.imwrite(os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  method , 'tmp', '%05d.jpg'%i),fake_frame)


        
    command1 = 'ffmpeg -framerate 25  -i '    + os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  method , 'tmp', '%05d.jpg' )+'  -vcodec libx264  -vf format=yuv420p -y ' +  cropped_video_path
    # command2 = 'ffmpeg -framerate 25  -i '    + os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  'gt' , 'tmp', '%05d.jpg' )+'  -vcodec libx264  -vf format=yuv420p -y ' +   os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  'gt',  'test_crop.mp4')
    print (command1)
    # os.system(command1)
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

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    t = -R*centroid_A.T + centroid_B.T

    return R, t
def RT_compute(method = 'baseline'):
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    lmark_path = os.path.join('/home/cxu-serve/p1/common/other/obama_fake',  method,  'test_crop.npy')  
    
    rt_path = lmark_path[:-4] +'_rt.npy'
    front_path = lmark_path[:-4] +'_front.npy'
   
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
        ret_R, ret_t = rigid_transform_3D( target, source)

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




def get_front():
    root = '/home/cxu-serve/p1/common/Obama'
    _file = open(os.path.join(root, 'pickle','train_lmark2img.pkl'), "rb")
    data = pkl.load(_file)
    _file.close()
    for index in tqdm(range(len(data))):
        v_id = data[index]
        video_path = os.path.join(root, 'video', v_id[0][:-11] + '_crop2.mp4'  )
            # mis_video_path = os.path.join(self.root, 'pretrain', mis_vid[0] , mis_vid[1][:5] + '_crop.mp4'  )
        v_frames = read_videos(video_path)
        lmark_path = os.path.join(root,  'video', v_id[0][:-11] +'_rt2.npy'  )
        rt = np.load(lmark_path)
        lmark_length = rt.shape[0]
        find_rt = []
        for t in range(0, lmark_length):
            find_rt.append(sum(np.absolute(rt[t,:3])))
        find_rt = np.asarray(find_rt)

        min_index = np.argmin(find_rt)
        
        img_path =  os.path.join(root,  'video', v_id[0][:-11] + '_%05d_2.png'%min_index  )
        cv2.imwrite(img_path, v_frames[min_index])
        data[index].append(min_index)
    with open(os.path.join( root, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)


    #     find_rt = []
    #     for t in range(0, lmark_length):
    #         find_rt.append(sum(np.absolute(rt[t,:3])))
    #     find_rt = np.asarray(find_rt)

    #     min_index = np.argmin(find_rt)
        
    #     img_path =  os.path.join(root,  'video', v_id[0][:-11] + '_%05d_2.png'%min_index  )
    #     cv2.imwrite(img_path, v_frames[min_index])
    #     data[index].append(min_index)
    # with open(os.path.join( root, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
    #     pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

# swith_identity('turing1')
# swith_identity_obama()
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
method = 'baseline'
# landmark_extractor(method )
# 
RT_compute(method)
# diff()
# landmark_extractor()
