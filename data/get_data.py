import os 
import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import util, face_utils
import face_tracker
import cv2
import face_alignment
import fnmatch
import librosa
import shutil
import mmcv
def prepare_data_lrs():
    # root = '/mnt/Data02/lchen63/lrs/'
    root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4'
    path = os.path.join( root , 'pretrain')
    trainset = []
    testset = []
    train_list = sorted(os.listdir(path))
    batch_length = int(0.1 * len(train_list))
    # train_list = train_list[ 4 * batch_length:5 * batch_length ]
    for i in tqdm(range(len(train_list))):
        p_id = train_list[i]
        person_path = os.path.join(path, p_id)
        chunk_txt = sorted(os.listdir(person_path))
        for txt in chunk_txt:
            if txt[-9:] !=  'front.npy':
                continue
            if np.load(os.path.join(path, p_id, txt)).shape[0]> 65:
                if i >  4 * batch_length and i < 5 * batch_length :
                    testset.append( [p_id, txt])
                else:
                    trainset.append( [p_id, txt])
    print (len(trainset))
    print (len(testset))
    with open(os.path.join(root, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(trainset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(root, 'pickle','test_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(testset, handle, protocol=pkl.HIGHEST_PROTOCOL)

def prepare_data_faceforencs_xu():
    path = '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped'

    # landmark_path = os.path.join(path, 'landmarks_seq')

    # video_path = os.path.join(path, 'videos')
    train_list = []
    test_list = []
    for i in range(1000):
        lmark_path = os.path.join(path, 'landmarks_seq', '%03d.npy'%i)

        video_path = os.path.join(path, 'videos', '%03d.mp4'%i)

        print (lmark_path)
        print (video_path)
        if os.path.exists(lmark_path) and os.path.exists(video_path) :
            if i < 800:
                train_list.append([lmark_path, video_path])
            else:
                test_list.append([lmark_path, video_path])
    
    with open(os.path.join('/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube', 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(train_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join('/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube', 'pickle','test_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(test_list, handle, protocol=pkl.HIGHEST_PROTOCOL)


def prepare_data_faceforencs_oppo():
    path = '/mnt/Data/lchen63/faceforensics/original_sequences'

    # landmark_path = os.path.join(path, 'landmarks_seq')

    # video_path = os.path.join(path, 'videos')
    train_list = []
    test_list = []
    for i in range(1000):
        lmark_path = os.path.join(path, 'landmarks', '%03d.npy'%i)

        video_path = os.path.join(path, 'videos', '%03d.mp4'%i)

        print (lmark_path)
        print (video_path)
        if os.path.exists(lmark_path) and os.path.exists(video_path) :
            if i < 800:
                train_list.append([lmark_path, video_path])
            else:
                test_list.append([lmark_path, video_path])
    
    with open(os.path.join(path, 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(train_list, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'pickle','test_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(test_list, handle, protocol=pkl.HIGHEST_PROTOCOL)

def prepare_data_vox2():
    root = '/mnt/Data/lchen63/voxceleb2'
    root_path = os.path.join(root, 'unzip' , 'dev_video' )

    # root_path = '/mnt/Data/lchen63/unzip/test_video'
    identities = sorted(os.listdir( root_path))
    total = len(identities)
    dataset  = []
    for index  in range(total):
        video_ids  = os.listdir( os.path.join( root_path , identities[index]))
        print (index, total)
        for v_id in tqdm(video_ids):
            all_files  =  os.listdir( os.path.join( root_path , identities[index], v_id))
            for ff in all_files:
                if ff[6:] == 'aligned.npy':
                    # try:
                    lmark_path = os.path.join( root_path , identities[index], v_id, ff)
                    rt_path = os.path.join( root_path , identities[index], v_id, ff[:5] + '_aligned_rt.npy')
                    ani_video_path =  os.path.join( root_path , identities[index], v_id, ff[:5] + '_aligned_ani.mp4')
                    video_path =  os.path.join( root_path , identities[index], v_id, ff[:5] + '_aligned.mp4')
                
                    if os.path.exists(video_path) and os.path.exists(ani_video_path) and os.path.exists(lmark_path) and  os.path.exists(rt_path) :
                        # cap_real = cv2.VideoCapture(video_path)
                        # cap_ani = cv2.VideoCapture(ani_video_path)
                        # lmark = np.load(lmark_path)
                        # flage = True
                        # for gg in range(lmark.shape[0]):
                        #     ret1, frame = cap_real.read()
                        #     ret2, frame = cap_ani.read()
                        #     if ret1 == False or ret2 == False:
                        #         print (lmark_path)
                        #         flage = False
                        #         break
                        # if flage:
                        dataset.append([ identities[index], v_id, ff[:5]])
                            
             
    print (len(dataset))
    with open(os.path.join(root, 'pickle','dev_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(dataset, handle, protocol=pkl.HIGHEST_PROTOCOL)
  

def unzip_video():
    path = '/home/cxu-serve/p1/common/grid/zip'
    # zipfiles = os.listdir(path)
    # for f in zipfiles:
    #     if 'mpg_6000' in f:
    #         command = 'tar -xvf ' + os.path.join(path, f) + ' -C ' + path
    #         print (command)
    #         os.system(command)
            # break
    for i in range (2, 35):
        command = 'mv  ' + os.path.join(path , 's' + str(i)) + ' /home/cxu-serve/p1/common/grid/video2'
        # old_path = os.path.join(path , 's' + str(i) ,'video', 'mpg_6000', '*')
        # command = 'mv ' + old_path + ' ' + os.path.join(path , 's' + str(i))
        # print (command)
        os.system(command)
        # command = 'rm -rf ' + os.path.join(path , 's' + str(i) ,'video')
        # print (command)
    


def openrate(lmark1):
    open_pair = []
    for i in range(3):
        open_pair.append([i + 61, 67 - i])
    open_rate1 = []
    for k in range(3):
        open_rate1.append( np.abs(lmark1[open_pair[k][0],:2] - lmark1[open_pair[k][1], :2]))
        
    open_rate1 = np.asarray(open_rate1)
    return open_rate1.mean() 
        

def prepare_data_grid():
    path ='/home/cxu-serve/p1/common/grid'
    # path = "/mnt/Data/lchen63/grid"
    trainset = []
    testset  =[]
    align_path = os.path.join( path , 'align')

    gg = os.listdir(align_path)
    for j in tqdm(range( len(gg))):
        i = gg[j]
        print ('+++++++', i)
        count = 0
        for vid in os.listdir( os.path.join(align_path, i ) ):
            if os.path.exists(os.path.join( align_path ,  i , vid[:-6] + '_front.npy') )  and os.path.exists(os.path.join(path , 'audio' ,  i , vid[:-6]  +'.wav' )) :
                tmp = [i , vid[:-6]]
                lmark_path = os.path.join( align_path ,  i , vid[:-6] + '_front.npy')
                lmark = np.load( lmark_path )
                if lmark.shape[0]< 74:
                    continue
                start_openrate = openrate(lmark[0])
                print (lmark_path)
                if start_openrate < 1.1:
                    count += 1
                    tmp.append(True)    
                else:
                    tmp.append(False)
                end_openrate = openrate(lmark[-1])

                if end_openrate < 1.1:
                    tmp.append(True)
                else:
                    tmp.append(False)
            #     audio_path = os.path.join( path, 'audio' ,  i , vid[:-6] + '.wav') 
            #     sound, _ = librosa.load(audio_path, sr=44100)
            #     lmark_path = os.path.join( align_path ,  i , vid[:-6] + '_front.npy')
            #     lmark1 = np.load(lmark_path)[:,:,:2]
            #     # face_utils.write_video_wpts_wsound( lmark1, sound, 44100, './gg', i +'_' + vid[:-6] +'_front'   , [0.0,256.0], [0.0,256.0])
            #     lmark_path = os.path.join( align_path ,  i , vid[:-6] + '_original.npy')
            #     lmark2 = np.load(lmark_path)[:,:,:2]
            #     # face_utils.write_video_wpts_wsound( lmark2, sound, 44100, './gg', i +'_' + vid[:-6] +'_original'   , [0.0,256.0], [0.0,256.0])
            #     video_path =  os.path.join( align_path ,  i , vid[:-6] + '_crop.mp4')
            #     cap = cv2.VideoCapture(video_path)
            #     real_video = []
            #     while(cap.isOpened()):
            #         ret, frame = cap.read()
            #         if ret == True:
            #             real_video.append(frame)
            #         else:
            #             break
            #     if os.path.exists('./tmp01'):
            #         shutil.rmtree('./tmp01')
            #     os.mkdir("./tmp01")
            #     for ii in range(lmark1.shape[0]):
            #         img = real_video[ii]
            #         print ('_++++++++++++' ,ii , openrate(lmark1[ii]))
            #         for jj in range(68):
            #             x=int(lmark1[ii][jj][1])
            #             y =int(lmark1[ii][jj][0])
            #             cv2.circle(img, (y, x), 1, (0, 0, 255), -1)
            #             x=int(lmark2[ii][jj][1])
            #             y =int(lmark2[ii][jj][0])
            #             cv2.circle(img ,  (y, x), 1, ( 255, 0 , 0), -1)
            #         cv2.imwrite(  './tmp01/%06d.jpg'%ii, img)
            #     mmcv.frames2video('./tmp01', './gg/' + i +'_' + vid[:-6] +'.mp4' )
            #     count += 1 
            # if count == 1:
                # break
        # break
                # for ff in os.listdir( os.path.join(align_path, i )):
                #     if fnmatch.fnmatch(ff, vid[:-6]  + '*diff*'):
                #         break
                if  i in set(["s1","s3","s4","s8","s16","s17","s18","s19","s20","s23","s24","s27","s29","s31","s32","s33","s2","s5","s6","s7","s12","s22","s25","s26","s28","s34"]):
                    trainset.append( tmp )

                else:
                    testset.append( tmp )
            # else:
            #     continue
                
                # print (os.path.join( align_path ,  i , vid[:-6] + '_original.npy'))
        # break
    print  (count)
    print  (len(trainset))
    with open(os.path.join(path, 'pickle','train_audio2lmark_grid.pkl'), 'wb') as handle:
        pkl.dump(trainset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'pickle','test_audio2lmark_grid.pkl'), 'wb') as handle:
        pkl.dump(testset, handle, protocol=pkl.HIGHEST_PROTOCOL)

def grid_check():
    root_path  ='/home/cxu-serve/p1/common/grid'
    _file = open(os.path.join(root_path,  'pickle','test_audio2lmark_grid.pkl'), "rb")
    testset = pkl.load(_file)
    _file.close()

    _file = open(os.path.join(root_path,  'pickle','train_audio2lmark_grid.pkl'), "rb")
    trainset = pkl.load(_file)
    _file.close()
    datalist = testset + trainset
    train_list  = []
    test_list  = []
    for index in range(len(datalist)):
        lmark_path = os.path.join(root_path ,  'align' , datalist[index][0] , datalist[index][1] + '_front.npy') 
        lmark = np.load( lmark_path )
        start_openrate = openrate(lmark[0])
        print (lmark_path)
        if start_openrate < 1.1:
            datalist[index].append(True)    
            print ('++++++++++++++++' , start_openrate )
        else:
            datalist[index].append(False)
            print ('------', start_openrate )
        end_openrate = openrate(lmark[-1])

        if end_openrate < 1.1:
            print ('++-------++' , end_openrate )
            datalist[index].append(True)
        else:
            print ('++-========++' , end_openrate )
            datalist[index].append(False)
        # if datalist[index][0]  in set(["s1","s3","s4","s8","s16","s17","s18","s19","s20","s23","s24","s27","s29","s31","s32","s33","s2","s5","s6","s7","s12","s22","s25","s26","s28","s34"]):
        #     train_list.append(datalist[index])
        # else:
        #     test_list.append(datalist[index])

    # with open(os.path.join(root_path, 'pickle','train_audio2lmark_grid.pkl'), 'wb') as handle:
    #     pkl.dump(train_list, handle, protocol=pkl.HIGHEST_PROTOCOL)   

    # with open(os.path.join(root_path, 'pickle','test_audio2lmark_grid.pkl'), 'wb') as handle:
    #     pkl.dump(test_list, handle, protocol=pkl.HIGHEST_PROTOCOL)     


    
# prepare_standard2()
# grid_check()
prepare_data_vox2()
# prepare_data_grid() 
# prepare_data_faceforencs_oppo()
# prepare_data_lrs()
# unzip_video()
