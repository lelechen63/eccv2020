import os
import argparse
import shutil
from tqdm import tqdm
import glob, os
import face_alignment
import numpy as np
import cv2
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
            
landmark_extractor()
# extract_audio()