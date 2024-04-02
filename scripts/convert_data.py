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
    packed_data = json.load(f)
    
# Data Filtering (Optional)

print('Filtering data.')

thresholds = {
    'segment_caption_data': 0.145,
    'instance_segment_caption_data': 0.146,
    'instance_caption_data': 0.126,
    'appearance_data': 0.129,
    'qa_data': 0.177,
    'instance_qa_data': 0.159,
    'cross_segment_qa_data': 0.211,
    'hypo_scene_data': 0.888,
    'comp_ret_data': 0.195,
    'segment_locate_data': 1.067
}

filtered_data = {}

for video_name in packed_data:
    filtered_data[video_name] = {}
    for data_type in packed_data[video_name]:
        filtered_data[video_name][data_type] = []
        for data in packed_data[video_name][data_type]:
            if 'match_score' in data and data['match_score'] > thresholds[data_type]:
                filtered_data[video_name][data_type].append(data)
            elif 'clip_similarity' in data and data['clip_similarity'] > thresholds[data_type]:
                filtered_data[video_name][data_type].append(data)
packed_data = filtered_data

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

def format_moments(moments, moment_template=None):
    if moment_template is None:
        moment_template = np.random.choice(moment_template_pool)
    moment_text = ''
    for i, m in enumerate(moments):
        if i == 0:
            moment_text = moment_text + moment_template.format(*m)
        elif i == len(moments) - 1:
            moment_text = moment_text + ' and ' + moment_template.format(*m)
        else:
            moment_text = moment_text + ', ' + moment_template.format(*m)
    return moment_text

def variable_transfer(var_dict):
    transferred_variables = {}
    for key in var_dict:
        if key == 'moment':
            if isinstance(var_dict[key][0], list):
                transferred_variables[key] = format_moments(var_dict[key])
            else:
                transferred_variables[key] = format_moment(var_dict[key])
        elif key == 'SOURCE_CLIP':
            transferred_variables[key] = format_moment(var_dict[key])
        elif key == 'click_position':
            transferred_variables[key] = f'[{var_dict[key][0]}s, <{round(var_dict[key][1][0], 2)}, {round(var_dict[key][1][1], 2)}>]'
        else:
            transferred_variables[key] = var_dict[key]
    return transferred_variables

converted_data = []
for video_name in tqdm(packed_data, desc='Converting data'):
    for data_type in packed_data[video_name]:
        for data in packed_data[video_name][data_type]:
            if isinstance(data['conversations'], dict):
                data['conversations'] = [data['conversations']]
            transferred_variables = variable_transfer(data['variables'])
            for conv in data['conversations']:
                for role in conv:
                    conv[role] = conv[role].format(**transferred_variables)
            converted_data.append({
                'id' : data['id'],
                'data_type' : data['data_type'],
                'conversations' : data['conversations'],
            })
with open(target_path, 'w') as f:
    json.dump(converted_data, f)
    
print('Finished.')