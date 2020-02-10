import os 
from scipy.io import wavfile

def tmp():
    path = '/data/lchen63/vox_obj/vox'
    ff = os.listdir(path)
    for f in ff:
        gg = f.split('__')
        new_path = os.path.join( path , gg[0] , gg[1], gg[2], gg[3])
        old_path = os.path.join(path , f)
        command = 'mv ' + old_path +' ' + new_path
        print (command)
        # os.system(command)
tmp()
# path = '/home/cxu-serve/p1/common/grid/audio/s1'
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
