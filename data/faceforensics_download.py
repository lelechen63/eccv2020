import os
from face_tracker import _crop_video
from pytube import YouTube
yt_baseurl = 'https://www.youtube.com/watch?v='
import argparse
import tempfile
import shutil
import json
import subprocess
import cv2
import wave
import contextlib
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_id",
                     type=int,
                     default=1)
    
    return parser.parse_args()
config = parse_args()

def create_fps_dict(data_path):
    with open(os.path.join(data_path, 'map.json'), 'r') as f:
        conversion_dict = json.load(f)

    fps_dict = {}
    for key in conversion_dict.keys():
        video_id, num = conversion_dict[key].split(' ')
        with open(os.path.join(data_path, 'downloaded_videos_info', video_id,
                       video_id + '.json'), 'r') as f:
            info_dict = json.load(f)
        fps_dict[key] = info_dict['fps']
        # print ( '=====' ,key , fps_dict[key])
    return fps_dict

def download_original_videos(yt_baseurl, config ):
    data_path = '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube'
    # fps_dict = create_fps_dict(data_path)
    with open(os.path.join(data_path, 'map.json'), 'r') as f:
        maps = json.load(f)
    for key in maps.keys():
        video_id, num = maps[key].split(' ')
        print (key, video_id)
        # if int(key) < 524 :
        #     continue
        # break
        # url_ids = sorted(os.listdir( os.path.join(data_path , 'downloaded_videos_info')  ))

        yt_url = yt_baseurl + video_id
        dirpath = tempfile.mkdtemp()
        
        try:
            yt = YouTube(yt_url)
            yt.streams.filter(subtype='mp4').first().download(output_path = dirpath,filename = 'tmp')
            tilename = os.listdir(dirpath)[0].split('.')[-1]
            save_path = os.path.join(data_path, 'full', key +    '.mp4')
            with open(os.path.join(data_path, 'downloaded_videos_info', video_id,
                        'extracted_sequences',  str(num) + '.json'), 'r') as f:
                frame_info = json.load(f)
                # fps = int(fps_dict[key])
                raw_video = os.path.join(data_path , 'raw' , "videos", key  + '.mp4')
                print (raw_video)
                cap = cv2.VideoCapture( raw_video )
                fps = cap.get(cv2.CAP_PROP_FPS)  
                print (fps)

                start_frame = float(frame_info[0])
                end_frame = float(frame_info[-1])
                start_time =  start_frame / fps
                end_time =  end_frame / fps
                print (start_frame, end_frame, fps, start_time, end_time)
                last_time = end_time - start_time
                #cut video by start time and end time
                command = 'ffmpeg -i ' + dirpath + '/tmp.' + tilename   + ' -ss {0:.2f}'.format(start_time) +' -strict -2 -t {0:.2f} -filter:v fps=fps='.format(last_time) + str(fps) + ' -y '.format(last_time)+ save_path
                print (command)
                os.system(command)

                # extract audio
                command = 'ffmpeg -i ' + save_path + ' -ar 44100 -ac 2 -y  ' + os.path.join(data_path, 'raw/audios', key + '.wav') 
                print (command)
                os.system(command)
        shutil.rmtree(dirpath)
            # break


        except:
            print ('***********************')
            print (yt_url,save_path )
            continue

        # break


# download_original_videos(yt_baseurl, config )

# _crop_video('/u/lchen63/lchen63_data/addition_example/id01822/00003.mp4')

def face_forensics_dataset_crop_preprocess():
    datapath = '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/raw/videos'
    pid = 0
    for i, v_name in enumerate(sorted(os.listdir(datapath))):  
        print (v_name)
        if i < 524:
            continue
        try:
            _crop_video( os.path.join(datapath , v_name), pid )
            command = 'ffmpeg -framerate 25  -i ' +   './temp%05d'%pid + '/%05d.png  -vcodec libx264 -y -vf format=yuv420p ' +  os.path.join(datapath , v_name).replace('raw', 'cropped') 
            os.system(command)
        except:
            print ( '===================')
            continue
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)
def get_audio_duration(filename):
    print (filename)
    with contextlib.closing(wave.open(filename,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration
def double_check():
    audio_path = '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/raw/audios'

    new_video_path = '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped/videos'

    original_video_path = '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/full'

    for i, v_name in enumerate(sorted(os.listdir(new_video_path))):
        if i < 660:
            continue
        new_filename = os.path.join( new_video_path, v_name ) 
        original_filename = os.path.join( original_video_path, v_name ) 
        if os.path.exists(new_filename) and os.path.exists(original_filename):
            audio_time = get_audio_duration(os.path.join( audio_path, v_name[:-4] + '.wav' )  )
            video_time = get_length(os.path.join( new_video_path, v_name )  )
            speed = audio_time / video_time
            # if old_fps != new_fps:
            command = 'ffmpeg -i ' +  new_filename +  ' -filter:v "setpts=' + str(speed)  + '*PTS" '  + os.path.join( ' -y /home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped2/', v_name )
            os.system(command)
            print (command)
            command = 'ffmpeg -i '+os.path.join( '/home/cxu-serve/p1/common/faceforensics/original_sequences/youtube/cropped2/', v_name) + ' -i '+os.path.join(audio_path, v_name[:-4])+'.wav -c:v copy -c:a aac -strict experimental -y ' +new_filename
            os.system(command)
            print (command)


            # vtime1 =  get_length(filename)
            # vtime2 =  get_length(os.path.join( original_video_path, v_name ))
            # print ( filename, vtime1, vtime2)

            # if abs(vtime1 - vtime2) > 0.1:
            #     print ('+++++' , filename, vtime1, vtime2)
        # if i == 10:
        #     break




double_check()
# face_forensics_dataset_crop_preprocess()

