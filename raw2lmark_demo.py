import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.modules.module import _addindent
import numpy as np
from collections import OrderedDict
import argparse


from models.networks import  SPCH2FLM2

from torch.nn import init
from utils import util, face_utils
import librosa

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size",
                     type=int,
                     default=1)
    parser.add_argument("--cuda",
                     default=True)
    parser.add_argument("--lstm",
                     default=True)

    parser.add_argument("--model_name",
                        type=str,
                        default="./checkpoints/atnet_raw_pca_with_exmaple/atnet_lstm_42.pth")
    parser.add_argument( "--sample_dir",
                    type=str,
                    default="./results")
                    # default='/media/lele/DATA/lrw/data2/sample/lstm_gan')test
    parser.add_argument('-i','--in_file', type=str, default='./audio/test.wav')
    parser.add_argument('-p','--person', type=str, default='./image/musk1.jpg')
    parser.add_argument('--device_ids', type=str, default='2')
    parser.add_argument('--num_thread', type=int, default=1)   
    return parser.parse_args()
config = parse_args()

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('./basics/shape_predictor_68_face_landmarks.dat')

def multi2single(model_path, id):
    checkpoint = torch.load(model_path)
    state_dict = checkpoint
    if id ==1:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2])
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean()

def mounth_open2close(lmark): # if the open rate is too large, we need to manually make the mounth to be closed.
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    
    upper_part = [49,50,51,52,53]

    lower_part = [59,58,57,56,55]

    diffs = []

    for k in range(3):
        mean = (lmark[open_pair[k][0],:2] + lmark[open_pair[k][1],:2] )/ 2
        print (mean)
        tmp = lmark[open_pair[k][0],:2]
        diffs.append((mean - lmark[open_pair[k][0],:2]).copy())
        lmark[open_pair[k][0],:2] = mean - (mean - lmark[open_pair[k][0],:2]) * 0.3
        lmark[open_pair[k][1],:2] = mean + (mean - lmark[open_pair[k][0],:2]) * 0.3

    diffs.insert(0, 0.6 * diffs[2])
    diffs.append( 0.6 * diffs[2])
    print (diffs)
    diffs = np.asarray(diffs)
    lmark[49:54,:2] +=  diffs
    lmark[55:60,:2] -=  diffs 
    return lmark




def preprocess_img(img_path):  # get cropped image by input the reference image
    img_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/tmp/tmp/00001_00030.png'
    frame = cv2.imread(img_path)
    x_list =  np.array([])
    y_list = np.array([])
    dis_list = np.array([])
    videos = []
    x_list, y_list, dis_list, videos, _ = face_tracker.crop_image(frame, count = 0)
    dis = np.mean(dis_list)
    top_left_x = x_list - (80 * dis / 90)
    top_left_y = y_list - (100* dis / 90)
    side_length = int((205 * dis / 90))
    if top_left_x[0] < 0 or top_left_y[0] < 0:
        img_size = videos[0].shape
        tempolate = np.ones((img_size[0] * 2, img_size[1]* 2 , 3), np.uint8) * 255
        tempolate_middle  = [int(tempolate.shape[0]/2), int(tempolate.shape[1]/2)]
        middle = [int(img_size[0]/2), int(img_size[1]/2)]
        tempolate[tempolate_middle[0] -middle[0]:tempolate_middle[0]+middle[0], tempolate_middle[1]-middle[1]:tempolate_middle[1]+middle[1], :] = videos[0]
        top_left_x[0] = top_left_x[0] + tempolate_middle[0]  -middle[0]
        top_left_y[0] = top_left_y[0] + tempolate_middle[1]  -middle[1]
        roi = tempolate[int(top_left_x[0]):int(top_left_x[0]) + side_length ,int(top_left_y[0]):int(top_left_y[0]) + side_length]
        roi =cv2.resize(roi,(256,256))
    else:
        roi = videos[0][int(top_left_x[0]):int(top_left_x[0]) + side_length ,int(top_left_y[0]):int(top_left_y[0]) + side_length]
        roi =cv2.resize(roi,(256,256))
    cv2.imwrite(img_path[:-4] +'_croped.png', roi)

    ###### get the 3D landamrk from the cropped image:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda:0')
    frame = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB )
    preds = fa.get_landmarks(frame)[0]
    
    ##########make the mounth closed
    if openrate(preds) > 1:
        preds = mounth_open2close(preds)

    return roi, preds
 
def get_demo_batch(audio_path , lmark):  #lmark size should be 136
    mean =  np.load('./basics/mean_grid_front.npy')
    component = np.load('./basics/U_grid_front.npy')
    norm_lmark = np.load('./basics/s1_pgbk6n_01.npy')
    norm_lmark = norm_lmark.reshape(-1)
    # print (lmark.shape , norm_lmark.shape)
    diff =  lmark - norm_lmark

    speech, fs = librosa.load(audio_path, sr=50000)
    print (speech.shape)
    chunck_size =int(fs * 0.04 )
    length = int(speech.shape[0] / chunck_size)
    left_append = speech[: 3 * chunck_size]
    right_append = speech[-4 * chunck_size:]
    speech = np.insert( speech, 0, left_append ,axis=  0)
    speech = np.insert( speech, -1, right_append ,axis=  0)
    norm_lmark = norm_lmark.reshape(1, 136)
    norm_lmark = np.dot(norm_lmark - mean, component.T)
    norm_lmark = torch.FloatTensor(norm_lmark)
    example_landmark = norm_lmark.repeat((length,1))

    chunks = []
    for r in range(length):
        t_chunk =speech[r * chunck_size : (r + 7)* chunck_size].reshape(1, -1)
        t_chunk = torch.FloatTensor(t_chunk)
        chunks.append(t_chunk)
    chunks = torch.stack(chunks, 0)
    print (chunks.shape, example_landmark.shape)
    return example_landmark,  chunks
 
a,b = get_demo_batch('/home/cxu-serve/p1/common/grid/audio/s1/bbaf3s.wav' , np.load('./basics/mean_grid_front.npy'))


def test():
    mean =  np.load('./basics/mean_grid_front.npy')
    component = np.load('./basics/U_grid_front.npy')
    config.cuda1 = torch.device('cuda:0')

    _ , lmark = preprocess_img(config.person)
    example_landmark,  chunks = get_demo_batch(config.in_file, lmark)

    generator = SPCH2FLM2()
    device_ids = [int(i) for i in config.device_ids.split(',')]
    generator    = nn.DataParallel(generator, device_ids= device_ids).cuda()
    generator.load_state_dict(torch.load(config.model_name))
    print ('load pretrained [{}]'.format(config.model_name))


    if os.path.exists('./temp'):
        shutil.rmtree('./temp')
    os.mkdir('./temp')
        
    
    sound, _ = librosa.load(config.in_file, sr=44100)

        

# test()