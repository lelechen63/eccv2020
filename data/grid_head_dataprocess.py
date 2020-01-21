import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
from face_tracker import _crop_video

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
    batch_length = int(0.1 * len(train_list))
    for i in tqdm(range(batch_length * (config.batch_id -1001), batch_length * (config.batch_id - 1000))):
        p_id = train_list[i]
        person_path = os.path.join('/home/cxu-serve/p1/common/grid/align', p_id)
        videos = sorted(os.listdir(person_path))
        for vid in videos:
            if vid[-5:] !=  'align':
                continue
            original_video_path =  os.path.join('/home/cxu-serve/p1/common/grid/video', p_id , vid[:-5] + 'mpg')
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
            
landmark_extractor()
