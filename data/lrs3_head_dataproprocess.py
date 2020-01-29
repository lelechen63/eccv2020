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

    train_list = sorted(os.listdir('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'))
    batch_length = int(0.01 * len(train_list))
    for i in tqdm(range(batch_length * (config.batch_id -1), batch_length * (config.batch_id))):
        p_id = train_list[i]
        person_path = os.path.join('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain', p_id)
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
    train_list = sorted(os.listdir('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'))
    batch_length = int( len(train_list))
    source = np.zeros((len(consider_key),3))
    ff = np.load('../basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    for i in tqdm(range(batch_length)):
        p_id = train_list[i]
        person_path = os.path.join('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain', p_id)
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
# RT_compute()
landmark_extractor()
# extract_audio()