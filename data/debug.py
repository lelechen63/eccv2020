import os 
train_list = sorted(os.listdir('/home/cxu-serve/p1/common/lrs3/lrs3_v0.4/pretrain'))
total =  len(train_list)
for i, id in enumerate(train_list) :
    if i > 100 and i < 150:
            print (id)
    if id == '1G8dQQrlbbA':
        
        print ( i , total, i/total)