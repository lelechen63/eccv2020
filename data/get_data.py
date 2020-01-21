import os 
import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import util
def prepare_data_lrs():
    path ='/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'
    trainset = []
    train_list = sorted(os.listdir(path))
    batch_length = int(0.4 * len(train_list))
    train_list = train_list[:batch_length]
    for i in tqdm(range(batch_length)):
        p_id = train_list[i]
        person_path = os.path.join('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain', p_id)
        chunk_txt = sorted(os.listdir(person_path))
        for txt in chunk_txt:
            if txt[-3:] !=  'npy':
                continue
            print (txt)
            if np.load(os.path.join('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain', p_id, txt)).shape[0]> 65:
                trainset.append( [p_id, txt])
    print (len(trainset))
    print (trainset[0])
   
    with open(os.path.join('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4', 'pickle','train_lmark2img.pkl'), 'wb') as handle:
        pkl.dump(trainset, handle, protocol=pkl.HIGHEST_PROTOCOL)


def prepare_data_grid():
    path ='/home/cxu-serve/p1/common/grid'
    trainset = []
    testset  =[]
    lmarks = []
    align_path = os.path.join( path , 'align')
    for i in os.listdir(align_path):
        
        for vid in os.listdir( os.path.join(align_path, i ) ):
            if os.path.exists(os.path.join( align_path ,  i , vid[:-6] + '_original.npy') ) :
                print ( os.path.join(align_path, i, vid[:-6] + '_crop.mp4'  ) )
                lmarks.append( np.load(os.path.join( align_path ,  i , vid[:-6] + '_original.npy')) )
                if  i == 's1' or i == 's2' or i == 's20' or i == 's22':
                    testset.append( [i , vid[:-6]] )
                else:
                    trainset.append( [i , vid[:-6]] )
        # break
    print (len(trainset))
    print (len(testset))
    lmarks = np.asarray(lmarks)
    mean_lmarks = np.mean(lmarks, axis=0)
    mean_lmarks = np.mean(mean_lmarks, axis=0)
    xLim=(0.0, 256.0)
    yLim=(0.0, 256.0)
    xLab = 'x'
    yLab = 'y'
    util.plot_flmarks(mean_lmarks, './gg.png', xLim, yLim, xLab, yLab, figsize=(10, 10) )
    np.save('../basics/grid_mean.npy' , mean_lmarks)

    with open(os.path.join(path, 'pickle','train_audio2lmark_grid.pkl'), 'wb') as handle:
        pkl.dump(trainset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(os.path.join(path, 'pickle','test_audio2lmark_grid.pkl'), 'wb') as handle:
        pkl.dump(testset, handle, protocol=pkl.HIGHEST_PROTOCOL)
    


prepare_data_grid()
