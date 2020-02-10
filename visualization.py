import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os
import sys
# from utils import face_utils
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

def vis():
    # v_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_crop.mp4'
    # lmark_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_original.npy'
    lmark_path = '/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/lZf1RB6l5Gs/00152_aligned.npy'
    # norm_lmark = np.load('./basics/standard.npy')
    # print (norm_lmark.shape)
    xLim=(0.0, 256.0)
    yLim=(0.0, 256.0)
    zLim=(-128, 128)
    gg = [[[ 49.89145375, 95.16354821, -45.12697709],
    [ 50.61250406, 107.8694257 , -45.35716938],
    [ 52.96611338, 121.36293542, -45.35252306],
    [ 55.29324537, 133.97776676, -44.09361103],
    [ 57.77832993, 147.46482494, -37.74046008],
    [ 64.20417286, 159.44887738, -25.72713449],
    [ 72.00925374, 168.19269703, -11.422772 ],
    [ 81.35554053, 176.70955558,  2.23570083],
    [ 97.9759953 , 181.9978821 ,  7.81057447],
    [112.33002442, 178.22050602,  1.8373623 ],
    [123.87593008, 171.4343643 , -12.07628049],
    [131.68101096, 162.58733497, -26.76076814],
    [138.90668174, 150.67782897, -39.08595681],
    [143.78547523, 137.13820354, -45.6822884 ],
    [146.10242201, 125.37743149, -47.13346008],
    [148.45628303, 113.39286483, -47.41445474],
    [150.72152135, 98.36450273, -47.42904915],
    [ 60.32862631, 81.70637678, 11.46762078],
    [ 65.86836881, 75.50567628, 20.1459345 ],
    [ 74.26576261, 73.28177351, 26.05277429],
    [ 80.5635882 , 74.37819425, 29.48455884],
    [ 86.91724959, 75.84538422, 30.45473958],
    [113.64523795, 75.84538422, 30.04272782],
    [119.99490319, 74.37819425, 28.70823705],
    [127.8234432 , 74.85699941, 24.98571641],
    [134.64062914, 77.0686822 , 18.67294776],
    [140.21474632, 83.27277129,  9.58812069],
    [ 99.49386889, 91.8902038 , 27.06148911],
    [ 99.50777176, 101.25750619, 32.40631397],
    [ 99.50383012, 109.26669181, 39.73589642],
    [ 99.48793763, 116.95974816, 40.40494473],
    [ 91.22612641, 123.94113802, 25.93041295],
    [ 93.57501241, 124.37432671, 28.1597412 ],
    [ 99.45728666, 126.57790998, 29.5032749 ],
    [103.06447911, 126.96236308, 28.3585525 ],
    [107.70868531, 126.18875969, 26.21486078],
    [ 69.8038982 , 93.18968379, 17.22488767],
    [ 73.93944333, 91.53347246, 23.70911653],
    [ 80.99980229, 91.23051283, 23.7266601 ],
    [ 85.88471707, 94.0055796 , 20.67281375],
    [ 80.78949459, 95.56815588, 22.90337122],
    [ 74.22000068, 95.30854016, 21.74041007],
    [113.08322777, 94.0055796 , 20.4987045 ],
    [119.57333442, 91.23051283, 23.23354844],
    [125.06059553, 91.53347246, 22.71826111],
    [130.74709273, 94.79345206, 15.88981016],
    [124.75585361, 96.88778556, 20.66225959],
    [118.17769325, 95.56815588, 22.35665615],
    [ 81.21106314, 143.25187636, 15.33051545],
    [ 86.70010186, 140.2420454 , 24.45751571],
    [ 94.02435501, 137.92870955, 30.16513592],
    [ 97.94259191, 138.89060862, 31.5157296 ],
    [101.12856949, 137.92870955, 30.79592879],
    [109.18508196, 141.97165925, 26.79355926],
    [115.44306962, 147.42651129, 18.21382614],
    [108.73491469, 151.39817461, 24.04136408],
    [102.48947153, 153.62747853, 26.39907442],
    [ 97.94259191, 154.44602356, 27.29966742],
    [ 90.975416 , 153.62747853, 25.61277787],
    [ 87.15026913, 149.73781726, 22.34833859],
    [ 81.6589967 , 143.1860214 , 15.28922333],
    [ 92.46044821, 143.53854504, 25.93353437],
    [ 97.94259191, 143.02351557, 27.71018756],
    [102.68770247, 143.53854504, 27.10421213],
    [113.45481757, 147.26240869, 18.25356241],
    [102.73578503, 147.99402534, 25.59918996],
    [ 97.94259191, 147.91388109, 25.94223221],
    [ 92.32389503, 146.39219611, 24.42824311]]]
    gg = np.asarray(gg)
    print (gg.shape)
    gg = gg[0]
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
    v_path = '/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/lZf1RB6l5Gs/00152_aligned.mp4'
    # v_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/0MMSpsvqiG8/00004_crop.mp4'
    cap  =  cv2.VideoCapture(v_path)
    
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        print ('+++++++++++++', ret)
        # if count == 20:
        #     break
        print  (count)
        if ret == True:
            if count < 127 or count > 130:
                count += 1
                continue
            print (count)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )
            preds =  lmark[count]
            get_face_image(preds)
            
            fig = plt.figure(figsize=plt.figaspect(.5))
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(frame)
            ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='w',lw=1)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='w',lw=1) 
            preds = gg
            ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            ax.axis('off') 
            count += 1
            
            # preds = mounth_open2close(preds)
            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(frame)
            # ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='g',lw=1)
            # ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='g',lw=1) 
            ax.axis('off')

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


            plt.show()
            # plt.savefig('./gg.png')

            print ('=======')
        else:
            break

vis()
