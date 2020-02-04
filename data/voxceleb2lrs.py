import numpy as np 
from collections import OrderedDict
import tempfile
import shutil
import numpy as np 
import mmcv
import scipy.ndimage.morphology
import cv2 
import time 
import dlib
import face_alignment
from scipy.spatial.transform import Rotation 
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from tqdm import tqdm
import time
import  utils.util as util
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R
res = 224
import  utils.visualizer as Visualizer
import pickle
import argparse

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)#,  device='cpu' )
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_id",
                     type=int,
                     default=1)
    
    return parser.parse_args()
config = parse_args()
def crop_image( lmark,  x_list = [], y_list = [], dis_list = [], lStart=36, lEnd=41, rStart=42, rEnd=47):

	new_shape = lmark
	leftEyePts = new_shape[lStart:lEnd]
	rightEyePts = new_shape[rStart:rEnd]

	leftEyeCenter = leftEyePts.mean(axis=0)
	rightEyeCenter = rightEyePts.mean(axis=0)
	max_v = np.amax(new_shape, axis=0)
	min_v = np.amin(new_shape, axis=0)

	max_x, max_y = max_v[0], max_v[1]
	min_x, min_y = min_v[0], min_v[1]
	dis = max(max_y - min_y, max_x - min_x)

	two_eye_center = (leftEyeCenter + rightEyeCenter)/2
	center_y, center_x = two_eye_center[0], two_eye_center[1]
	x_list =np.append( x_list, center_x )
	y_list = np.append(y_list, center_y)
	dis_list = np.append(dis_list, dis)
	return  x_list, y_list, dis_list


def _crop_video(video, ani_video, lmark_path):
    lmark = np.load(lmark_path)
    count = 0
    x_list =  np.array([])
    y_list = np.array([])
    dis_list = np.array([])
    videos = []
    cap  =  cv2.VideoCapture(video)
    cap2 = cv2.VideoCapture(ani_video)
    t = time.time()
    ani_videos = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        if ret == True:
            x_list, y_list, dis_list = crop_image(lmark[count], x_list, y_list, dis_list )
            videos.append(frame)
            ani_videos.append(frame2)
            count += 1
        else:
            break
    dis = np.mean(dis_list)
    top_left_x = x_list - (80 * dis / 90)
    top_left_y = y_list - (100* dis / 90)
    top_left_x = util.oned_smooth(top_left_x )
    top_left_y = util.oned_smooth(top_left_y)	
    side_length = int(205 * dis / 90)
    out = cv2.VideoWriter(video[:-4] + '_aligned.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 25, (256,256))
    out2 = cv2.VideoWriter(video[:-4] + '_aligned_ani.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 25, (256,256))
    for i in tqdm(range(x_list.shape[0])):
        tempolate = np.zeros((side_length, side_length , 3), np.uint8) 
        tempolate2 = np.zeros((side_length, side_length , 3), np.uint8)
        lmark[i,:, 0] = lmark[i,:, 0] - top_left_y[i]  
        lmark[i,: ,1] =  lmark[i,: ,1]  -top_left_x[i]  
        tempolate[int(- top_left_x[i]  )  : int(- top_left_x[i] )   + 224 ,int(- top_left_y[i])   :int(- top_left_y[i] ) + 224 , :] = videos[i]
        tempolate2[int(- top_left_x[i]  )  : int(- top_left_x[i] )    + 224 ,int(- top_left_y[i])    :int(- top_left_y[i] )  + 224 , :] = ani_videos[i]
        lmark[i, :, :2 ] = lmark[i, :, :2 ] * (256 / side_length)
        tempolate2 = cv2.resize(tempolate2,(256,256)) 
        tempolate =cv2.resize(tempolate,(256,256))
        # for jj in range(68):
        #     x=int(lmark[i][jj][1])
        #     y =int(lmark[i][jj][0])
        #     cv2.circle(tempolate, (y, x), 1, (0, 0, 255), -1)
        #     cv2.circle(tempolate2 ,  (y, x), 1, ( 255, 0 , 0), -1)
        out.write(tempolate)
        out2.write(tempolate2)
    np.save(lmark_path[:-4] + "_aligned.npy" , lmark)
    out.release()
    out2.release()

def align_videos(config):
    root_path = '/mnt/Data/lchen63/unzip/test_video'
    identities = sorted(os.listdir( root_path))
    total = len(identities)
    batch_size = int(0.1 * total)
    for index  in range(total):
    # for index  in range(batch_size * (config.batch_id -1) , batch_size * (config.batch_id)):
        video_ids  = os.listdir( os.path.join( root_path , identities[index]))
        for v_id in video_ids:
            all_files  =  os.listdir( os.path.join( root_path , identities[index], v_id))
            for ff in all_files:
                if ff[-3:] == 'npy' and len(ff) == 9:
                    try:
                        ori_video_path =  os.path.join( root_path , identities[index], v_id, ff[:-4] + '.mp4')
                        ani_video_path = os.path.join( root_path , identities[index], v_id, ff[:-4] + '_ani.mp4')
                        lmark_path = os.path.join( root_path , identities[index], v_id, ff)
                        
                        if os.path.exists(ori_video_path[:-4] + '_aligned.mp4'):
                            continue
                        if os.path.exists(ori_video_path) and os.path.exists(ani_video_path) and os.path.exists(lmark_path) :
                            _crop_video(ori_video_path, ani_video_path, lmark_path )
                    except:
                        print  (ori_video_path)
                        continue
        #             break
        #     break
        # break
        
align_videos(config)


def prepare_standard1():  # get cropped image by input the reference image
    img_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/tmp/tmp/00001_00030.png'
    frame = cv2.imread(img_path)
    x_list =  np.array([])
    y_list = np.array([])
    dis_list = np.array([])
    videos = []
    x_list, y_list, dis_list, videos, _ = face_tracker.crop_image(frame, count = 0)

    dis = np.mean(dis_list)
    print (dis)
    top_left_x = x_list - (80 * dis / 90)
    top_left_y = y_list - (100* dis / 90)
    side_length = int((205 * dis / 90))

    for i in tqdm(range(x_list.shape[0])):
        if top_left_x[i] < 0 or top_left_y[i] < 0:
            img_size = videos[i].shape
            tempolate = np.ones((img_size[0] * 2, img_size[1]* 2 , 3), np.uint8) * 255
            tempolate_middle  = [int(tempolate.shape[0]/2), int(tempolate.shape[1]/2)]
            middle = [int(img_size[0]/2), int(img_size[1]/2)]
            tempolate[tempolate_middle[0]  - middle[0] : tempolate_middle[0]+middle[0], tempolate_middle[1]-middle[1]:tempolate_middle[1]+middle[1], :] = videos[i]
            top_left_x[i] = top_left_x[i] + tempolate_middle[0]  - middle[0]
            top_left_y[i] = top_left_y[i] + tempolate_middle[1]  -middle[1]
            roi = tempolate[int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
            roi =cv2.resize(roi,(256,256))
            cv2.imwrite('./gg/%05d.png'%( i), roi)
        else:
            roi = videos[i][int(top_left_x[i]):int(top_left_x[i]) + side_length ,int(top_left_y[i]):int(top_left_y[i]) + side_length]
            roi =cv2.resize(roi,(256,256))
            cv2.imwrite('./gg/%05d.png'%( i), roi)

def prepare_standard2():
    standard_img = cv2.imread('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/00000.png')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')
    frame = cv2.cvtColor(standard_img,cv2.COLOR_BGR2RGB )

    preds = fa.get_landmarks(frame)[0]
    lmark_path = '/u/lchen63/Project/face_tracking_detection/eccv2020/basics/standard.npy'    
    np.save(lmark_path, preds)


