import os 
from scipy.io import wavfile
path = '/home/cxu-serve/p1/common/grid/audio/s1'
for i in os.listdir(path):
    fs, data = wavfile.read(  os.path.join( path , i ))
    print (data.shape , fs ,)

# train_list = sorted(os.listdir('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'))
# total =  len(train_list)
# for i, id in enumerate(train_list) :
#     if i > 100 and i < 150:
#             print (id)
#     if id == '1G8dQQrlbbA':
        
#         print ( i , total, i/total)