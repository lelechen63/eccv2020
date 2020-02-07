import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os
import sys
# from utils import face_utils
import librosa
# from utils import util
def smooth(x,window_len=11,window='hanning'):
   
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

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
def vis():
    # v_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_crop.mp4'
    # lmark_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_original.npy'
    lmark_path = '/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/utfjXffHDgg/00198_aligned.npy'
    norm_lmark = np.load('./basics/standard.npy')
    print (norm_lmark.shape)
    xLim=(0.0, 256.0)
    yLim=(0.0, 256.0)
    zLim=(-128, 128)
    # util.plot_flmarks3D(norm_lmark, './gg.png',xLim, yLim, zLim)
    # norm_lmark = mounth_open2close(norm_lmark)
    # np.save('./basics/s1_pgbk6n_01.npy', norm_lmark)
    lmark = np.load(lmark_path)#[:,:,:2]
    # audio_path = lmark_path.replace('align', 'audio').replace('_front.npy', '.wav')
    # sound, _ = librosa.load(audio_path, sr=44100)
    # face_utils.write_video_wpts_wsound(lmark, sound, 44100, './', 'front', [0.0,256.0], [0.0,256.0])
    # for i in range(lmark.shape[1]):
    #     x = lmark[: , i,0]
    #     x = smooth(x, window_len=5)
    #     lmark[: ,i,0 ] = x[2:-2]
    #     y = lmark[:, i, 1]
    #     y = smooth(y, window_len=5)
    #     lmark[: ,i,1  ] = y[2:-2] 
    # face_utils.write_video_wpts_wsound(lmark, sound, 44100, './', 'norm', [0.0,256.0], [0.0,256.0])

    # openrates = []
    # motions = []
    # for  i in range(lmark.shape[0]):
    #     openrates.append(openrate(lmark[i]))
    # openrates = np.asarray(openrates)
    # min_index = np.argmin(np.absolute(openrates))
        
    # diff = lmark[min_index] - norm_lmark

    # lmark = lmark - diff
    # face_utils.write_video_wpts_wsound(lmark, sound, 44100, './', 'dif', [0.0,256.0], [0.0,256.0])

    # mean =  np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/mean_grid_front.npy')
    # component = np.load('/u/lchen63/Project/face_tracking_detection/eccv2020/basics/U_grid_front.npy')
    # data = np.dot(lmark.reshape(lmark.shape[0], -1) - mean, component.T)
    # fake_lmark = np.dot(data,component) + mean
    # fake_lmark = fake_lmark.reshape(75, 68, 2)
    # face_utils.write_video_wpts_wsound(fake_lmark, sound, 44100, './', 'fake', [0.0,256.0], [0.0,256.0])

    # lmark2 = np.load(gg_path)
    # lmark2 = lmark2.reshape( 68 , 2)
    # lmark_path ='/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/0MMSpsvqiG8/00004_original.npy'
    v_path = '/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/utfjXffHDgg/00198_aligned.mp4'
    # v_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/0MMSpsvqiG8/00004_crop.mp4'
    cap  =  cv2.VideoCapture(v_path)
    
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        print ('+++++++++++++', ret)
        # if count == 20:
        #     break
        if ret == True:
            print (count)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )
            preds =  lmark[count]
            
            fig = plt.figure(figsize=plt.figaspect(.5))
            # ax = fig.add_subplot(1, 3, 1)
            # ax.imshow(frame)
            # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
            # preds = norm_lmark
            # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.axis('off') 
            
            # preds = mounth_open2close(preds)
            # ax = fig.add_subplot(1, 3, 2)
            # ax.imshow(frame)
            # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='g',lw=1) 
            # ax.axis('off')

            # ax = fig.add_subplot(1, 3, 3)
            # ax.imshow(frame)
            # preds = norm_lmark
            # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            # # preds = norm_lmark  + motions[count] + diff
            # # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='g',lw=1) 
            # ax.axis('off') 

            # lmark_rgb = util.plot_landmarks( preds)
            # ax = fig.add_subplot(1, 3, 2)
            # ax.imshow(lmark_rgb)
            # ax.axis('off')

            ax = fig.add_subplot(1, 1, 1, projection='3d')
            preds = lmark[count+ 40] 
            surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
            ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
            ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
            ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
            ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
            ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
            ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
            ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
            ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

      

            count += 1
            ax.view_init(elev=90., azim=90.)
            ax.set_xlim(ax.get_xlim()[::-1])
            # import matplotlib as mpl

            # mpl.use('tkAgg')


            plt.show()
            # plt.savefig('./gg.png')

            print ('=======')
        else:
            break

vis()
