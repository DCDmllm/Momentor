import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument('--source_path', required=True, type=str)
parser.add_argument('--video_path', required=True, type=str)
args = parser.parse_args()

source_path = args.source_path
video_path = args.video_path

import json, numpy as np, copy
from tqdm import tqdm

print('Loading data.')

with open(source_path, 'r') as f:
    packed_data = json.load(f)
    
print('Start downloading.')
    
video_names = list(packed_data.keys())
youtube_video_format = 'https://www.youtube.com/watch?v={}'
video_path_format = os.path.join(video_path, '{}.mp4')

for video_name in video_names:
    try:
        url = youtube_video_format.format(video_name)
        file_path = video_path_format.format(video_name)
        if os.path.exists(file_path):
            continue
        os.system('yt-dlp -o ' + file_path + ' -f 134 ' + url)
        print(Back.YELLOW + f'Downloading of Video {video_name} has finished.' + Style.RESET_ALL)
    except:
        print(Back.RED + f'Downloading of Video {video_name} has failed.' + Style.RESET_ALL)
        
print('Finished.')