import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source_path', required=True, type=str)
parser.add_argument('--target_path', required=True, type=str)
args = parser.parse_args()

source_path = args.source_path
target_path = args.target_path

import json, numpy as np, copy
from tqdm import tqdm

print('Loading data.')

with open(source_path, 'r') as f:
    gesm_data = json.load(f)
    
# Data Conversion

moment_template_pool = [
    "{:.2f}s-{:.2f}s",
    "{:.2f}s~{:.2f}s",
    "{:.2f}s to {:.2f}s",
    "{:.0f}s-{:.0f}s",
    "{:.0f}s~{:.0f}s",
    "{:.0f}s to {:.0f}s",
    "{:.2f} seconds-{:.2f} seconds",
    "{:.2f} seconds~{:.2f} seconds",
    "{:.2f} seconds to {:.2f} seconds",
    "{:.0f} seconds-{:.0f} seconds",
    "{:.0f} seconds~{:.0f} seconds",
    "{:.0f} seconds to {:.0f} seconds",
]

def format_moment(moment, moment_template=None):
    if moment_template is None:
        moment_template = np.random.choice(moment_template_pool)
    return moment_template.format(*moment)

converted_data = []
for video_name in tqdm(gesm_data, desc='Converting data'):
    moment_template = np.random.choice(moment_template_pool)
    data = ''
    for moment, caption in zip(gesm_data[video_name]['timestamps'], gesm_data[video_name]['captions']):
        data += f'{format_moment(moment, moment_template)}: {caption} '
    data = data[:-1]
    converted_data.append({
        'id' : video_name,
        'data_type' : 'event_sequence_decoding',
        'data' : data,
    })
    
with open(target_path, 'w') as f:
    json.dump(converted_data, f)
    
print('Finished.')