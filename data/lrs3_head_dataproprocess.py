import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
from scipy.spatial.transform import Rotation 
from utils import face_utils
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_id",
                     type=int,
                     default=1)
    
    return parser.parse_args()
config = parse_args()



def _extract_audio(lists):
    for line in lists:
        command = 'ffmpeg -i ' + line + ' -ar 44100  -ac 2 -y  ' + line.replace('mp4','wav')
        try:
            # pass
            os.system(command)
        except BaseException:
            print ('++++++++++++++++++++++++' , line)


def extract_audio():
    root_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'
    total = []
    train_list = sorted(os.listdir(root_path))
    print (len(train_list))
    batch_length = int(0.2 * len(train_list))
    for i in range(batch_length ):
        pid = train_list[i]        
        for ff in os.listdir( os.path.join( root_path, pid) ):
            if ff.endswith("_crop.mp4"):
                total.append(os.path.join(root_path, pid, ff.split('_')[0] + '.mp4' ))
    batch = 1
    datas = []
    batch_size = len(total) / batch
    temp = []
    for i, d in enumerate(total):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []
    _extract_audio(datas[0])

def landmark_extractor():
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')
    rootpath ='/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/trainval' 
    train_list = sorted(os.listdir(rootpath))
    batch_length = int( 0.1 *  len(train_list))
    for i in tqdm(range(batch_length * (config.batch_id -1), batch_length * (config.batch_id))):
        p_id = train_list[i]
        person_path = os.path.join(rootpath, p_id)
        chunk_txt = sorted(os.listdir(person_path))
        for txt in chunk_txt:
            if txt[-8:] !=  'crop.mp4':
                continue
            cropped_video_path = os.path.join( person_path, txt)
            lmark_path = cropped_video_path[:-9] +'_original.npy'
            if os.path.exists(lmark_path):
                continue
            
            cap = cv2.VideoCapture(cropped_video_path)
            lmark = []
            try:
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
    root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'
    train_list = sorted(os.listdir(root))
    batch_length = int( len(train_list))
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    for i in tqdm(range(batch_length)):
        p_id = train_list[i]
        person_path = os.path.join(root, p_id)
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
        open_rate1.append(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2])
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean()
def pca_lmark_lrs():
    root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'
    train_list = sorted(os.listdir(root))
    batch_length = int( len(train_list))
    landmarks = []
    k = 20
    norm_lmark = np.load('../basics/s1_pgbk6n_01.npy')
    for i in tqdm(range(batch_length)):
        p_id = train_list[i]
        person_path = os.path.join(root, p_id)
        videos = sorted(os.listdir(person_path))
        for vid in videos:
            
            if vid[-9:] !=  'front.npy':
                continue
            lmark_path = os.path.join( person_path,vid)
            lmark = np.load(lmark_path)[:,:,:2]
            # if lmark.shape[0]< 70:
            #     continue
            for i in range(lmark.shape[1]):
                x = lmark[: , i,0]
                x = face_utils.smooth(x, window_len=5)
                lmark[: ,i,0 ] = x[2:-2]
                y = lmark[:, i, 1]
                y = face_utils.smooth(y, window_len=5)
                lmark[: ,i,1  ] = y[2:-2] 
            openrates = []
            for  i in range(lmark.shape[0]):
                openrates.append(openrate(lmark[i]))
            openrates = np.asarray(openrates)
            min_index = np.argmin(np.absolute(openrates))
            diff =  lmark[min_index] - norm_lmark
            # print (lmark_path[:-10] +'_diff.npy')
            np.save(lmark_path[:-10] +'_%05d_diff.npy'%(min_index) , diff)
            lmark = lmark - diff
            indexs = random.sample(range(0,70), 20)
            for i in indexs:
                landmarks.append(lmark[i])
       
    landmarks = np.stack(landmarks)
    print (landmarks.shape)
    landmarks = landmarks.reshape(landmarks.shape[0], 136)
    pca = PCA(n_components=20)
    pca.fit(landmarks)
    
    np.save('../basics/mean_grid_front.npy', pca.mean_)
    np.save('../basics/U_grid_front.npy',  pca.components_)

RT_compute()
# landmark_extractor()
# extract_audio()