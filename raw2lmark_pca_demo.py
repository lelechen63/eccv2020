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
import cv2
from models.networks import  A2L , A2L_deeps
from scipy.io import wavfile
import scipy.signal
from torch.nn import init
from utils import util, face_utils
from data import face_tracker
import librosa
import face_alignment
from data.grid_head_dataprocess import *
from data.dp2model import load_model
from data.dp2dataloader import SpectrogramParser
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
                        default="./checkpoints/atnet_raw_pca_with_exmaple_select/atnet_lstm_23.pth")
    parser.add_argument( "--sample_dir",
                    type=str,
                    default="./results")
    parser.add_argument('-i','--in_file', type=str, default='./audio/f_f.wav')
    parser.add_argument('-p','--person', type=str, default='./image/musk1.jpg')
    parser.add_argument('--device_ids', type=str, default='0')
    parser.add_argument('--num_thread', type=int, default=1)   
    parser.add_argument('--threeD', action='store_true')
    parser.add_argument('--deeps', action='store_true')

    return parser.parse_args()
config = parse_args()


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
        open_rate1.append(np.abs( lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2]))
        
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
    # img_path = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/tmp/tmp/00001_00030.png'
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
    

    ## remove the RT
    lmark = preds
    consider_key = [1,2,3,4,5,11,12,13,14,15,27,28,29,30,31,32,33,34,35,39,42,36,45,17,21,22,26]
    source = np.zeros((len(consider_key),3))
    ff = np.load('./basics/standard.npy')
    for m in range(len(consider_key)):
        source[m] = ff[consider_key[m]]  
    source = np.mat(source)
    lmark_part = np.zeros((len(consider_key),3))
    for m in range(len(consider_key)):
        lmark_part[m] = lmark[consider_key[m]] 

    target = np.mat(lmark_part)
    ret_R, ret_t = face_utils.rigid_transform_3D( target, source)
    source_lmark  = np.mat(lmark)

    A2 = ret_R*source_lmark.T
    A2+= np.tile(ret_t, (1, 68))
    A2 = A2.T
    preds = A2
    return roi, preds
 
def get_demo_batch(audio_path , lmark):  #lmark size should be 136
    if config.threeD:
        if len(lmark.shape) > 1:
            lmark = lmark.reshape(-1)
        mean =  np.load('./basics/mean_grid_front_3d.npy')
        component = np.load('./basics/U_grid_front_3d.npy')
    else:
        if len(lmark.shape) > 1:
            lmark = lmark[:,:2].reshape(-1)
        mean =  np.load('./basics/mean_grid_front.npy')
        component = np.load('./basics/U_grid_front.npy')
    norm_lmark = np.load('./basics/s1_pgbk6n_01.npy')
    if not config.threeD:
        norm_lmark = norm_lmark[:,:2]
    print  (norm_lmark.shape)
    ##########make the mounth closed
    if openrate(norm_lmark) > 1:
        print ('==============enforce mouth to be closed')
        norm_lmark = mounth_open2close(norm_lmark)
    else:
        print ( '+++', openrate(norm_lmark))

    norm_lmark = norm_lmark.reshape(-1)

    diff =  lmark - norm_lmark

    command = 'ffmpeg -i ' + audio_path +' -ar 50000 -y ./tmp.wav'
    os.system(command)
    if config.deeps:
        dp += wavfile.read( './tmp.wav')
        # speech = scipy.signal.resample(speech, 50000)
        # fs = 50000
        print  (fs)
        # speech, fs = librosa.load(audio_path, sr=50000)
        chunck_size =int(fs * 0.04 )
        length = int(speech.shape[0] / chunck_size)
        left_append = speech[: 3 * chunck_size]
        right_append = speech[-4 * chunck_size:]
        speech = np.insert( speech, 0, left_append ,axis=  0)
        speech = np.insert( speech, -1, right_append ,axis=  0)
        
        norm_lmark = norm_lmark.reshape(1, -1)

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
    else:

        fs, speech = wavfile.read( './tmp.wav')
        # speech = scipy.signal.resample(speech, 50000)
        # fs = 50000
        print  ('============================================================')
        # speech, fs = librosa.load(audio_path, sr=50000)
        chunck_size =int(fs * 0.04 )
        length = int(speech.shape[0] / chunck_size)
        left_append = speech[: 3 * chunck_size]
        right_append = speech[-4 * chunck_size:]
        speech = np.insert( speech, 0, left_append ,axis=  0)
        speech = np.insert( speech, -1, right_append ,axis=  0)
        
        norm_lmark = norm_lmark.reshape(1, -1)

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
    return example_landmark,  chunks, diff
 
# a,b = get_demo_batch('/home/cxu-serve/p1/common/grid/audio/s1/bbaf3s.wav' , np.load('./basics/mean_grid_front.npy'))


def test():
    if config.threeD:
            mean =  np.load('./basics/mean_grid_front_3d.npy')
            component = np.load('./basics/U_grid_front_3d.npy')
    else:
        mean =  np.load('./basics/mean_grid_front.npy')
        component = np.load('./basics/U_grid_front.npy')
    config.cuda1 = torch.device('cuda:0')

    _ , lmark = preprocess_img(config.person)
    # config.in_file = '/home/cxu-serve/p1/common/grid/audio/s22/bbic9p.wav'
    example_landmark,  chunks, diff= get_demo_batch(config.in_file, lmark)
    print  (example_landmark.shape , chunks.shape)

    if config.deeps:
        generator = A2L_deeps()
    else:
        generator = A2L()
    device_ids = [int(i) for i in config.device_ids.split(',')]
    generator    = nn.DataParallel(generator, device_ids= device_ids).cuda()
    generator.load_state_dict(torch.load(config.model_name))
    print ('load pretrained [{}]'.format(config.model_name))
    generator.eval() 

    # if os.path.exists('./temp'):
    #     shutil.rmtree('./temp')
    # os.mkdir('./temp')
    if config.cuda:
        chunks = Variable(chunks.float()).cuda(config.cuda1)
        example_landmark = Variable(example_landmark.float()).cuda(config.cuda1) 
    fake_lmark, _ = generator(example_landmark, chunks)
    fake_lmark = fake_lmark.data.cpu().numpy()
    fake_lmark = np.dot(fake_lmark,component) + mean

    example_landmark = example_landmark.data.cpu().numpy()
    example_landmark = np.dot(example_landmark,component) + mean
    print( fake_lmark.shape)
    if config.threeD:
        fake_lmark = fake_lmark.reshape(fake_lmark.shape[0]  , 68 ,3)
    else:
        fake_lmark = fake_lmark.reshape(fake_lmark.shape[0], 68 , 2)

        example_landmark = example_landmark.reshape(example_landmark.shape[0], 68 , 2)
            

    fake_lmark = fake_lmark[:,:,:2].reshape(fake_lmark.shape[0],   68 * 2)
    sound, _ = librosa.load(config.in_file, sr=44100)
    face_utils.write_video_wpts_wsound(fake_lmark, sound, 44100, config.sample_dir, 'fake_demo', [0.0,256.0], [0.0,256.0])


    example_landmark = example_landmark[:,:,:2].reshape(example_landmark.shape[0],   68 * 2)
    sound, _ = librosa.load(config.in_file, sr=44100)
    face_utils.write_video_wpts_wsound(example_landmark, sound, 44100, config.sample_dir, 'ex_demo', [0.0,256.0], [0.0,256.0])


        

test()