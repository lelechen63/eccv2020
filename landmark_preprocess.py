import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import argparse
import cv2
import librosa
from utils import face_utils, util
import numpy

def smooth(x,window_len=11,window='hanning'):
   
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

ms = np.load('./basics/grid_mean.npy') 
import time 
def main():
    fs = 44100
    output_path = './sample'
    shape1 = np.load('/home/cxu-serve/p1/common/grid/align/s1/bbas3a_original.npy')[:,:,:-1]
    audio1 = '/home/cxu-serve/p1/common/grid/audio/s1/bbas3a.wav'
    shape2 = np.load('/home/cxu-serve/p1/common/grid/align/s1/bbir8p_original.npy')
    t = time.time()
    for i in range(shape1.shape[1]):
        x = shape1[: , i,0]
        x = smooth(x, window_len=5)
        shape1[: ,i,0 ] = x[2:-2]
        y = shape1[:, i, 1]
        y = smooth(y, window_len=5)
        shape1[: ,i,1  ] = y[2:-2]
        
    # fnorm = face_utils.faceNormalizer()

    # aligned_frames = fnorm.alignEyePoints(shape1)
    # transferredFrames = fnorm.transferExpression(shape1, ms[:,:-1])        
    # frames = fnorm.unitNorm(transferredFrames)
    # sound, sr = librosa.load('/home/cxu-serve/p1/common/grid/audio/s1/bbas3a.wav', sr=fs)
    print (time.time() - t)
    np.save('/home/cxu-serve/p1/common/grid/align/s1/bbas3a_norm.npy', shape1)
    # face_utils.plot_lmark_as_video(shape1, './ggggg.mp4' , audio1,  xLim=(0.0, 256.0), yLim=(0.0, 256.0) ) #, sound, fs, output_path, 'PD_pts', [0, 1], [0, 1])


# util.image_to_video('/tmp/tmptu5rqvhg','./ggggg.mp4' )
# util.add_audio('./ggggg.mp4','/home/cxu-serve/p1/common/grid/audio/s1/bbas3a.wav' )
main()