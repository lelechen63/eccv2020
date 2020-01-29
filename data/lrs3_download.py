import os
from face_tracker import _crop_video
from pytube import YouTube
yt_baseurl = 'https://www.youtube.com/watch?v='
import argparse
import shutil
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_id",
                     type=int,
                     default=1)
    
    return parser.parse_args()
config = parse_args()
# root = '/home/cxu-serve/p1/common/lrs3/lrs3_v0.4'
root = '/mnt/Data02/lchen63/lrs/'
train_list = sorted(os.listdir(   os.path.join(root, 'trainval') ))
print (len(train_list))
batch_length = int(0.1 * len(train_list))
for i in range(batch_length * (config.batch_id -1), batch_length * (config.batch_id)):
    if os.path.exists('./tmp%05d'%config.batch_id):
        shutil.rmtree('./tmp%05d'%config.batch_id)
    os.mkdir('./tmp%05d'%config.batch_id)
    p_id = train_list[i]
    # p_id = '1G8dQQrlbbA'
    # extract audio from video 
    person_path = os.path.join(root, 'trainval', p_id)
    chunk_txt = sorted(os.listdir(person_path))
    if len(chunk_txt) == 0:
        continue
    # for txt in chunk_txt:
    #     if txt[-3:] !=  'txt':
    #         os.remove(os.path.join(person_path,txt))

    f = open(os.path.join(person_path,chunk_txt[0][:-3] +'txt'), "r")
    f.readline()
    f.readline()
    line = f.readline()
    url_tile = line.split(' ')[-1]
    yt_url = yt_baseurl + url_tile
    try:
        yt = YouTube(yt_url)
        yt.streams.first().download(output_path = './tmp%05d'%config.batch_id,filename = 'tmp')
        tilename = os.listdir('./tmp%05d'%config.batch_id)[0].split('.')[-1]
    except:
        print ('***********************')
        print (train_list[i], yt_url)
        continue

    for txt in chunk_txt:
        if txt[-4:] != '.txt':
            continue
        txt_path = os.path.join(person_path, txt)
        if os.path.exists( txt_path[:-4] + '_crop.mp4'):
            continue
        print (txt_path)   
        # try:
        f = open(txt_path, "r")
        start_frame = -1
        previous = ''
        counter = 0
        line = f.readline()
        
        while line:
            print (line)
            print ('++', len(line))
            if len(line) == 1:
                if start_frame == -1:
                    line = f.readline()
                    tmp  = f.readline()
                    frame_id = tmp.split(' ')[0]
                    start_frame = float(frame_id)
                else:
                    frame_id = previous.split(' ')[0]
                    end_frame = float(frame_id)
                    break
            previous = line
            line = f.readline()
            counter += 1
        print (start_frame, end_frame)
        start_time =  start_frame / 25.0
        last_time = end_frame / 25.0 - start_time
        print (start_time, last_time)

        #cut video by start time and end time
        command = 'ffmpeg -i ./tmp%05d/'%config.batch_id + 'tmp.' + tilename   + ' -ss {0:.2f}'.format(start_time) +' -strict -2 -t {0:.2f} -filter:v fps=fps=25 -y '.format(last_time)+ txt_path[:-3] + 'mp4'
        print (command)
        os.system(command)
        print ('================== video extracted')

        # #change video fps to 25 fps
        # command = 'ffmpeg -i ' +  txt_path[:-3] + 'mp4' +  ' -strict -2 -t {0:.2f} -y '.format(last_time)+ txt_path[:-3] + 'mp4'
        # print (command)
        # os.system(command)
        # print ('================== video extracted')

        # extract audio
        command = 'ffmpeg -i ' + txt_path[:-3] + 'mp4' + ' -ar 44100 -ac 2 -y  ' + txt_path[:-3] + 'wav'
        print (command)
        os.system(command)

        print ('================== audio extracted')

        _crop_video(txt_path[:-3] + 'mp4', config.batch_id)

        command = 'ffmpeg -framerate 25  -i ./temp%05d'%config.batch_id + '/%05d.png  -vcodec libx264  -vf format=yuv420p -y ' + txt_path[:-4] + '_crop.mp4'
        os.system(command)
        # except:
        #     print ('************************')
        #     continue
        # break
    # break


