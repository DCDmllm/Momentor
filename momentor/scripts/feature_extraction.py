import argparse
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--video_file_dir', required=True, type=str)
parser.add_argument('--log_file_path', required=True, type=str)
parser.add_argument('--save_dir', required=True, type=str)
parser.add_argument('--device_id', required=True, type=int)
parser.add_argument('--num_sampled_frames', required=False, default=300, type=int)

args = parser.parse_args()
video_file_dir = args.video_file_dir
log_file_path = args.log_file_path
save_dir = args.save_dir
device_id = args.device_id
num_sampled_frames = args.num_sampled_frames

import os, time, pandas as pd, numpy as np, pypeln as pl
import torch
import logging
import math
logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
device = torch.device(f'cuda:{device_id}')

from get_data_utils import *
from import_clip import load_clip

clip_encode_batchsize = 30
clip_feature_dim = 1024
clip_model, clip_processor, clip_encode_image, clip_encode_text = load_clip('cpu', device, torch.float16)

def get_video(params):
    video_name = params['video_name']
    video_path = ytt_get_file_path(video_name)

    video = read_video_file(video_path, decoder='decord', decode_audio=False)
    frames, duration = read_video_frames(
        video, num_samples=num_sampled_frames, centered=False
    )
    frames = frames.transpose(0, 1)
    
    step_params = {
        'video_path' : video_path,
        'duration' : duration,
        'frames' : frames,
    }
    return dict(list(params.items()) + list(step_params.items()))

@torch.no_grad()
def encode_video(params):
    video_name = params['video_name']
    frames = params['frames']
    feature = torch.empty(num_sampled_frames, clip_feature_dim, dtype=torch.float16)
    
    for b in range(math.ceil(frames.shape[0]/clip_encode_batchsize)):
        inputs = clip_processor(
            images=frames[b*clip_encode_batchsize:(b+1)*clip_encode_batchsize], return_tensors="pt"
        )
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16).to(device)
        feature[b*clip_encode_batchsize:(b+1)*clip_encode_batchsize] = clip_model.vision_model(**inputs).pooler_output
    save_content = {'duration' : params['duration'], 'feature' : feature}
    np.save(os.path.join(save_dir, f'{video_name}.npy'), save_content)
    return None

ytt_video_dir_format = video_file_dir + '{}.mp4'
def ytt_get_file_path(video_name):
    return ytt_video_dir_format.format(video_name)
valid_video_names = [file_name[:file_name.find('.')] for file_name in os.listdir(video_file_dir)]

input_params = []
for video_name in valid_video_names:
    if not os.path.exists(os.path.join(save_dir, f'{video_name}.npy')):
        input_params.append({
            'video_name' : video_name
        })

def print_info(func, process_id):
    def new_func(params):
        global start
        video_id = params['video_index']
        logging.info(f'Video {video_id} process {process_id} starts at {time.time() - start}.')
        ret = func(params)
        logging.info(f'Video {video_id} process {process_id} finishes at {time.time() - start}.')
        return ret
    return new_func

pipe = [get_video, encode_video]
pipe = [print_info(method, i) for i, method in enumerate(pipe)]

stage = input_params
for method in pipe:
    stage = pl.thread.map(method, stage, workers=1, maxsize=1)
res = list(stage)