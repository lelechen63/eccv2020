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

def get_demo_batch(audio_path , lmark):
    mean =  np.load('./basics/mean_grid_front.npy')
    component = np.load('./basics/U_grid_front.npy')
    norm_lmark = np.load('./basics/s1_pgbk6n_01.npy')
    diff =  lmark - norm_lmark

    speech, fs = librosa.load(audio_path, sr=50000)
    chunck_size =int(fs * 0.04 )
    length = (speech / chunck_size)
    left_append = speech[: 3 * chunck_size]
    right_append = speech[-4 * chunck_size:]
    speech = np.insert( speech, 0, left_append ,axis=  0)
    speech = np.insert( speech, -1, right_append ,axis=  0)
    norm_lmark = norm_lmark.reshape(1, 136)
    norm_lmark = np.dot(norm_lmark - mean, component.T)
    norm_lmark = torch.FloatTensor(norm_lmark)

    example_landmark = norm_lmark.repeat(length,1)

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
    data_batch = get_demo_batch(config.in_file, lmark)

    generator = SPCH2FLM2()
    device_ids = [int(i) for i in config.device_ids.split(',')]
    generator    = nn.DataParallel(generator, device_ids= device_ids).cuda()
    generator.load_state_dict(torch.load(config.model_name))
    print ('load pretrained [{}]'.format(config.model_name))


    if os.path.exists('./temp'):
        shutil.rmtree('./temp')
    os.mkdir('./temp')
    os.mkdir('./temp/img')
    os.mkdir('./temp/motion')
    os.mkdir('./temp/attention')
    
    test_file = config.in_file

    
    example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
    example_image = transform(example_image)

    example_landmark =  example_landmark.reshape((1,example_landmark.shape[0]* example_landmark.shape[1]))

    if config.cuda:
        example_image = Variable(example_image.view(1,3,128,128)).cuda()
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)) ).cuda()
    else:
        example_image = Variable(example_image.view(1,3,128,128))
        example_landmark = Variable(torch.FloatTensor(example_landmark.astype(float)))
    # Load speech and extract features
    example_landmark = example_landmark * 5.0
    example_landmark  = example_landmark - mean.expand_as(example_landmark)
    example_landmark = torch.mm(example_landmark,  pca)
    speech, sr = librosa.load(test_file, sr=16000)
    mfcc = python_speech_features.mfcc(speech ,16000,winstep=0.01)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech,16000,winstep=0.01)

    sound, _ = librosa.load(test_file, sr=44100)

    print ('=======================================')
    print ('Start to generate images')
    t =time.time()
    ind = 3
    with torch.no_grad(): 
        fake_lmark = []
        input_mfcc = []
        while ind <= int(mfcc.shape[0]/4) - 4:
            t_mfcc =mfcc[( ind - 3)*4: (ind + 4)*4, 1:]
            t_mfcc = torch.FloatTensor(t_mfcc).cuda()
            input_mfcc.append(t_mfcc)
            ind += 1
        input_mfcc = torch.stack(input_mfcc,dim = 0)
        input_mfcc = input_mfcc.unsqueeze(0)
        fake_lmark = encoder(example_landmark, input_mfcc)
        fake_lmark = fake_lmark.view(fake_lmark.size(0) *fake_lmark.size(1) , 6)
        example_landmark  = torch.mm( example_landmark, pca.t() ) 
        example_landmark = example_landmark + mean.expand_as(example_landmark)
        fake_lmark[:, 1:6] *= 2*torch.FloatTensor(np.array([1.1, 1.2, 1.3, 1.4, 1.5])).cuda() 
        fake_lmark = torch.mm( fake_lmark, pca.t() )
        fake_lmark = fake_lmark + mean.expand_as(fake_lmark)
    

        fake_lmark = fake_lmark.unsqueeze(0) 

        fake_ims, atts ,ms ,_ = decoder(example_image, fake_lmark, example_landmark )

        for indx in range(fake_ims.size(1)):
            fake_im = fake_ims[:,indx]
            fake_store = fake_im.permute(0,2,3,1).data.cpu().numpy()[0]
            scipy.misc.imsave("{}/{:05d}.png".format(os.path.join('../', 'temp', 'img') ,indx ), fake_store)
            m = ms[:,indx]
            att = atts[:,indx]
            m = m.permute(0,2,3,1).data.cpu().numpy()[0]
            att = att.data.cpu().numpy()[0,0]

            scipy.misc.imsave("{}/{:05d}.png".format(os.path.join('../', 'temp', 'motion' ) ,indx ), m)
            scipy.misc.imsave("{}/{:05d}.png".format(os.path.join('../', 'temp', 'attention') ,indx ), att)

        print ( 'In total, generate {:d} images, cost time: {:03f} seconds'.format(fake_ims.size(1), time.time() - t) )
            
        fake_lmark = fake_lmark.data.cpu().numpy()
        np.save( os.path.join( config.sample_dir,  'obama_fake.npy'), fake_lmark)
        fake_lmark = np.reshape(fake_lmark, (fake_lmark.shape[1], 68, 2))
        utils.write_video_wpts_wsound(fake_lmark, sound, 44100, config.sample_dir, 'fake', [-1.0, 1.0], [-1.0, 1.0])
        video_name = os.path.join(config.sample_dir , 'results.mp4')
        utils.image_to_video(os.path.join('../', 'temp', 'img'), video_name )
        utils.add_audio(video_name, config.in_file)
        print ('The generated video is: {}'.format(os.path.join(config.sample_dir , 'results.mov')))
        

# test()