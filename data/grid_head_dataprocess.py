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
    

# RT_compute()
# landmark_extractor()
