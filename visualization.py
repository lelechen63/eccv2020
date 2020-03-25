
# import matplotlib.pyplot as plt
# import numpy as np
# # name_list = ['Monday','Tuesday','Friday','Sunday']
# num_list = np.array([204,147,123,105, 89, 66, 27] )/240.0 

# num_list1 =  np.array([198,132,107,77,59,55,23]) /240.0
# x =list(range(len(num_list)))
# total_width, n = 0.8, 2
# width = total_width / n
 
# plt.bar(x, num_list, width=width, label='Lip-sync',fc = 'y')
# for i in range(len(x)):
#     x[i] = x[i] + width
# plt.bar(x, num_list1, width=width, label='Realistic',fc = 'r')
# plt.legend()

# plt.show()
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os
import sys
from utils import util
import librosa
from PIL import Image  
from data.keypoint2img import *
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


def get_face_image(keypoints):
    part_list = [[list(range(0, 17))  ], # face
                     [range(17, 22)],                                  # right eyebrow
                     [range(22, 27)],                                  # left eyebrow
                     [[28, 31], range(31, 36), [35, 28]],              # nose
                     [[36,37,38,39], [39,40,41,36]],                   # right eye
                     [[42,43,44,45], [45,46,47,42]],                   # left eye
                     [range(48, 55), [54,55,56,57,58,59,48], range(60, 65), [64,65,66,67,60]], # mouth and tongue
                    ]    
    w, h = 256 , 256
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
    for edge_list in part_list:
        for edge in edge_list:
            im_edge = np.zeros((h, w), np.uint8) # edge map for the current edge
            for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i+edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]
                curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape
                drawEdge(im_edges, curve_x, curve_y, bw= 1)
    im_edges =Image.fromarray(im_edges)
    im_edges.save( './gg.png')

def read_videos( video_path):
    cap = cv2.VideoCapture(video_path)
    real_video = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            real_video.append(frame)
        else:
            break

    return real_video
def vis():
    # v_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_crop.mp4'
    # lmark_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_original.npy'
    # lmark_path = '/home/cxu-serve/p1/common/grid/align/s1/pgbk6n_front.npy'
    # norm_lmark = np.load('./basics/s1_pgbk6n_01.npy')
    # norm_lmark = np.load(lmark_path)[1]
    # print (norm_lmark.shape)
    rt  =  np.load('/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/OLguY5ofUrY/00044_aligned_rt.npy')
    # print (rt.shape)
    # lmark_length = rt.shape[0]
    # src_lmark = np.load(src_lmark_path)[:,:,:2]
    
    # tar_lmark = np.load(tar_lmark_path)[:,:,:2]
    # lmark2 = np.load(tar_lmark_path)[:,:2]


    # find_rt = []
    # for t in range(0, lmark_length):
    #     find_rt.append(sum(np.absolute(rt[t,:3])))
    # find_rt = np.asarray(find_rt)

    # min_indexs =  np.argsort(find_rt)[:50]

    # for indx in min_indexs:
    #     print 


    # openrates = []
    # for  i in range(src_lmark.shape[0]):
    #     openrates.append(openrate(src_lmark[i]))
    # openrates = np.asarray(openrates)
    # min_index = np.argmin(openrates)

    v_path ='/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/OLguY5ofUrY/00044_aligned.mp4'
    # v_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/0MMSpsvqiG8/00004_crop.mp4'
    
    count = 0
    frames = read_videos(v_path)

        # text 
      
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
      
    # org 
    org = (00, 185) 
      
    # fontScale 
    fontScale = 1
       
    # Red color in BGR 
    color = (0, 0, 255) 
      
    # Line thickness of 2 px 
    thickness = 1

    for count in range(0,len(frames),10):
        print (count)
        print (rt[count])
        lists = util.rt_to_degree(rt[count])
        lists.append(count)
        lists.append('----')
        lists.append(rt[count])

        text = str(lists)
        print ('text')

        frame = frames[count]

        image = cv2.putText(frame, text, org, font, fontScale,  
                 color, thickness, cv2.LINE_AA, False) 
        cv2.imwrite('./data/temp00001/%05d.png'%count, image)
        # frame = cv2.imread('/home/cxu-serve/p1/common/demo/picasso1_crop.png')
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )
        # preds =  tar_lmark[count]
        # get_face_image(preds)
        
        # fig = plt.figure(figsize=plt.figaspect(.5))
        # ax = fig.add_subplot(1, 1, 1)
        
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
        # count += 1
        
        # # preds = norm_lmark
        # ax = fig.add_subplot(1, 3, 2)
        # ax.imshow(frame)
        # preds =  tar_lmark[count]
        # preds = preds.reshape(68,2)
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
        # preds  = src_lmark[count]
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

        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # preds = lmark[count+ 40] 
        # surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
        # ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
        # ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
        # ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
        # ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
        # ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
        # ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
        # ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
        # ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

    

        
        # ax.view_init(elev=90., azim=90.)
        # ax.set_xlim(ax.get_xlim()[::-1])
        # # import matplotlib as mpl

        # mpl.use('tkAgg')


        # plt.show()
        # plt.savefig('./gg.png')
# 
        # print ('=======')
    

vis()
