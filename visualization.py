import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os
import sys
# from utils import util


def vis():
    v_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_crop.mp4'
    # lmark_path = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain/00j9bKdiOjk/00001_original.npy'
    lmark_path = '/home/cxu-serve/p1/common/grid/align/s1/bbas3a_norm.npy'
    v_path = '/home/cxu-serve/p1/common/grid/align/s1/bbas3a_crop.mp4'
    cap  =  cv2.VideoCapture(v_path)
    lmark = np.load(lmark_path)
    count = 0
    cap  =  cv2.VideoCapture(v_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        # if count == 20:
        #     break
        if ret == True:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB )
            preds = lmark[count]
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
            ax.axis('off')

            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(frame)
            # lmark_rgb = util.plot_landmarks( preds)
            # ax = fig.add_subplot(1, 3, 2)
            # ax.imshow(lmark_rgb)
            # ax.axis('off')

            # ax = fig.add_subplot(1, 3, 3, projection='3d')
            # surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
            # ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
            # ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
            # ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
            # ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
            # ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
            # ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
            # ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
            # ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )

      

            count += 1
            # ax.view_init(elev=90., azim=90.)
            # ax.set_xlim(ax.get_xlim()[::-1])
            # import matplotlib as mpl

            # mpl.use('tkAgg')


            plt.show()
            # plt.savefig('./gg.png')

            print ('=======')
        else:
            break

vis()