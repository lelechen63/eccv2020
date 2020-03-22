# import os 
# from scipy.io import wavfile
from utils import face_utils, util
# import numpy as np
# lmark = np.load('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/test/P2AUat93a8Q/00002_original.npy')[73]

# face_utils.get_roi(lmark)
import numpy as np
from utils import util
ani_lmarks = []
lmark = np.load('/home/cxu-serve/p1/common/demo/lisa2_video_original.npy')
rt  =  np.load('/home/cxu-serve/p1/common/demo/00003_aligned_rt.npy')
print (rt.shape,lmark.shape)
for gg in range(rt.shape[0]):
    ani_lmarks.append(util.reverse_rt(lmark[gg], rt[gg]))
np.save('/home/cxu-serve/p1/common/demo/lisa2_crop_video_original.npy', ani_lmarks)
# for i in os.listdir(path):
#     fs, data = wavfile.read(  os.path.join( path , i ))
#     print (data.shape , fs ,)

# train_list = sorted(os.listdir('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'))
# total =  len(train_list)
# for i, id in enumerate(train_list) :
#     if i > 100 and i < 150:
#             print (id)
#     if id == '1G8dQQrlbbA':
        
#         print ( i , total, i/total)
