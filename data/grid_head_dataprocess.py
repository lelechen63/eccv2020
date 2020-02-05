import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
from face_tracker import _crop_video
from utils import face_utils
from scipy.spatial.transform import Rotation 
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_id",
                     type=int,
                     default=1)
    
    return parser.parse_args()
config = parse_args()



def landmark_extractor():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')

    train_list = sorted(os.listdir('/home/cxu-serve/p1/common/grid/align'))
    batch_length = int( len(train_list))
    for i in tqdm(range(batch_length * (config.batch_id -1001), batch_length * (config.batch_id - 1000))):
        p_id = train_list[i]
        person_path = os.path.join('/home/cxu-serve/p1/common/grid/align', p_id)
        videos = sorted(os.listdir(person_path))
        for vid in videos:
            if vid[-5:] !=  'align':
                continue
            original_video_path =  os.path.join('/home/cxu-serve/p1/common/grid/video2', p_id , vid[:-5] + 'mpg')
            print (original_video_path)
            cropped_video_path = os.path.join( person_path, vid[:-6] + '_crop.mp4')
            lmark_path = cropped_video_path[:-9] +'_original.npy'
            
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
        #     break
        # break


def RT_compute():
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    train_list = sorted(os.listdir('/home/cxu-serve/p1/common/grid/align'))
    batch_length = int( len(train_list))
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    for i in tqdm(range(batch_length)):
        p_id = train_list[i]
        person_path = os.path.join('/home/cxu-serve/p1/common/grid/align', p_id)
        videos = sorted(os.listdir(person_path))
        for vid in videos:
            if vid[-12:] !=  'original.npy':
                continue
            lmark_path = os.path.join( person_path,vid)
            rt_path = os.path.join( person_path,vid[:-12] +'rt.npy')
            front_path = os.path.join( person_path,vid[:-12] +'front.npy')
            # normed_path  = os.path.join( person_path,vid[:-12] +'normed.npy')
            # if os.path.exists(front_path):
            #     continue
            lmark = np.load(lmark_path)
            ############################################## smooth the landmark
            
            # for i in range(lmark.shape[1]):
            #     x = lmark[: , i,0]
            #     x = face_utils.smooth(x, window_len=5)
            #     lmark[: ,i,0 ] = x[2:-2]
            #     y = lmark[:, i, 1]
            #     y = face_utils.smooth(y, window_len=5)
            #     lmark[: ,i,1  ] = y[2:-2]
            #     z = lmark[:, i, 2]
            #     z = face_utils.smooth(z, window_len=5)
            #     lmark[: ,i, 2  ] = z[2:-2]
            # np.save(normed_path, lmark)
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


def pca_3dlmark_grid():  ## this time we will use standard as tempolate to be consistent with voxceleb
    root_path  ='/home/cxu-serve/p1/common/grid'
    _file = open(os.path.join(root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
    datalist = pkl.load(_file)
    _file.close()
    batch_length = int( len(datalist))
    landmarks = []
    k = 20
    norm_lmark = np.load('../basics/standard.npy')
    datalist3d = datalist.copy()
    for index in tqdm(range(batch_length)):
        lmark_path = os.path.join(root_path ,  'align' , datalist[index][0] , datalist[index][1] + '_front.npy') 
        lmark = np.load(lmark_path)

        openrates = []
        for  i in range(lmark.shape[0]):
            openrates.append(openrate(lmark[i]))
        openrates = np.asarray(openrates)
        min_index = np.argmin(openrates)
        diff =  lmark[min_index] - norm_lmark
        np.save(lmark_path[:-10] +'_%05d_diff_3d.npy'%(min_index) , diff)
        datalist3d[index][-1] = min_index 
        lmark = lmark - diff
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
    # landmarks = landmarks.reshape(landmarks.shape[0], 204)
    # pca = PCA(n_components=k)
    # pca.fit(landmarks)
    
    # np.save('../basics/mean_grid_front_3d.npy', pca.mean_)
    # np.save('../basics/U_grid_front_3d.npy',  pca.components_)
    with open(os.path.join(root_path, 'pickle','test_audio2lmark_grid_3d.pkl'), 'wb') as handle:
        pkl.dump(datalist, handle, protocol=pkl.HIGHEST_PROTOCOL)

# pca_lmark_grid()
pca_3dlmark_grid()
# data = np.load('/home/cxu-serve/p1/common/grid/align/s1/lwae8n_front.npy')[:,:,:2]
# data = data.reshape(data.shape[0], 136)
# print (data.shape)
# mean = np.load('../basics/mean_grid_front.npy')
# component = np.load('../basics/U_grid_front.npy')
# data_reduced = np.dot(data - mean, component.T)

# data_original = np.dot(data_reduced,component) + mean
# np.save( 'gg.npy', data_original )
# print (data - data_original)
# RT_compute()
# landmark_extractor()
