import os 
import pickle as pkl
import numpy as np
from tqdm import tqdm
def prepare_data():
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

prepare_data()
