import re
import copy
import pickle
import numpy as np
import argparse
import torch
from dp2model import load_model
from dp2dataloader import SpectrogramParser


def interpolate_features(features, input_rate, output_rate, output_len=None):
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps,
                                             input_timestamps,
                                             features[:, feat])
    return output_features

def parse_audio(audio, audio_parser, model, device):
    audio_spect = audio_parser.parse_audio(audio).contiguous()
    audio_spect = audio_spect.view(1, 1, audio_spect.size(0), audio_spect.size(1))
    audio_spect = audio_spect.to(device)
    input_sizes = torch.IntTensor([audio_spect.size(3)]).int()
    print  (audio_spect.shape)
    print (input_sizes)
    parsed_audio, output_sizes = model(audio_spect, input_sizes)

    # audio (124667, ), audio_spect (1, 1, 161, 780), parsed_audio (1, 390, 29)
    return parsed_audio, output_sizes

def load_audio(audio_path):
    pass


class AudioHandler:
    def __init__(self, args):
        # self.config = config
        # self.audio_feature_type = config['audio_feature_type']
        # self.num_audio_features = config['num_audio_features']
        # self.audio_window_size = config['audio_window_size']
        # self.audio_window_stride = config['audio_window_stride']

        self.args = args
        self.audio_feature_type = self.args.audio_feature_type
        self.num_audio_features = self.args.num_audio_features
        self.audio_window_size = self.args.audio_window_size
        self.audio_window_stride = self.args.audio_window_stride

    def process(self, audio):
        if self.audio_feature_type.lower() == "none":
            return None
        elif self.audio_feature_type.lower() == 'deepspeech':
            return self.convert_to_deepspeech(audio)
        else:
            raise NotImplementedError("Audio features not supported")
    def convert_to_deepspeech(self, audio):

        if type(audio) == dict:
            pass
        else:
            raise ValueError('Wrong type for audio')

        processed_audio = copy.deepcopy(audio)
        for subj in audio.keys():
            for seq in audio[subj].keys():
                print ('process audio: %s - %s'%(subj, seq))

                audio_sample = audio[subj][seq]['audio']
                sample_rate = audio[subj][seq]['sample_rate']

                print (audio_sample.shape)
                print (sample_rate)
                device = torch.device("cuda:0" if args.cuda else "cpu")
                
                model = load_model(device, args.model_path, args.half)
                model.eval()

                audio_parser = SpectrogramParser(model.audio_conf, normalize=True)
                parsed_audio, output_sizes = parse_audio(audio_sample, audio_parser, model, device)
                print (parsed_audio.shape)

                audio_len_s = float(audio_sample.shape[0]) / sample_rate
                num_frames = int(round(audio_len_s * 25))
                print (num_frames)
                print ('+++')
                network_output = interpolate_features(parsed_audio.data[0].cpu().numpy(), 25, 25,
                                                          output_len=num_frames)
                print (network_output.shape)
        return processed_audio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='deepspeech2 transcription')
    parser.add_argument('--cuda', action="store_true", default=True)
    parser.add_argument('--half', action="store_true", default=False)
    parser.add_argument('--model_path', default="/")
    parser.add_argument('--audio_path', default='/')
    parser.add_argument('--audio_feature_type', default='deepspeech')
    parser.add_argument('--num_audio_features', default=29)
    parser.add_argument('--audio_window_size', default=16)
    parser.add_argument('--audio_window_stride', default=1)
    args = parser.parse_args()
    args.model_path = '/u/lchen63/voca/deepspeech_pytorch/models/deepspeech.pth'
    args.audio_path = '/u/lchen63/voca/training_data/raw_audio_fixed.pkl'
    audio_handler = AudioHandler(args)
    _file = open(args.audio_path, 'rb')
    data = pickle._Unpickler(_file)
    data.encoding = 'latin1'
    raw_audio = data.load()
    # raw_audio = pickle.load(open(args.audio_path, 'rb'))
    print (raw_audio.keys())
    processed_audio = audio_handler.convert_to_deepspeech(raw_audio)
    print (processed_audio)