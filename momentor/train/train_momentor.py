import copy, os, json
from dataclasses import dataclass, field
import re, numpy as np
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from momentor.train.momentor_trainer import MomentorTrainer
from momentor import video_conversation as conversation_lib
from momentor.model.momentor import MomentorLlamaForCausalLM, MomentorConfig
import torch.distributed as dist
from momentor.constants import *
from tqdm import tqdm
from nltk import sent_tokenize

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    num_temporal_tokens: int = field(default=300)

@dataclass
class DataArguments:
    data_dir: str = ''
    feature_dir: str = ''
    num_sampled_frames: int = field(default=300)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    initialize: Optional[str] = field(default='default')
    model_max_length: int = field(default=2048)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    dist.barrier()
    state_dict = trainer.model.state_dict()
    trainer._save(output_dir, state_dict=state_dict)  # noqa
    
def preprocess_multimodal(
        sources: Sequence[str],
        cur_token_len: int,
) -> Dict:
    video_token_len = cur_token_len
    for source in sources:
        for i, sentence in enumerate(source):
            if i == 0:
                sentence['User'] = DEFAULT_VIDEO_TOKEN + '\n' + sentence['User']
            replace_token = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
            sentence["User"] = sentence["User"].replace(DEFAULT_VIDEO_TOKEN, replace_token)
    return sources

def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"User": conv.roles[0], "Assistant": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        conv.messages = []
        for j, conv_turn in enumerate(source):
            if len(set(conv_turn.keys()).intersection(roles.keys())) != 2:
                continue
            conv.append_message(conv.roles[0], conv_turn['User'])
            conv.append_message(conv.roles[1], conv_turn['Assistant'])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class YTTDataset(object):
    def __init__(self, instruction_data, feature_dir):
        self.instruction_data = instruction_data
        self.feature_dir = feature_dir
        
    def get_num_instructions(self):
        return len(self.instruction_data)
    
    def get_video_info(self, video_name):
        return np.load(os.path.join(self.feature_dir, f'{video_name}'.npy), allow_pickle=True).tolist()
    
    def get_instruction(self, index):
        assert index < self.get_num_instructions()
        return self.instruction_data[index]

class InstructionLoader(Dataset):
    def __init__(
        self, 
        ytt_dataset, 
        tokenizer: transformers.PreTrainedTokenizer, 
        video_token_len: int,
    ):
        self.ytt_dataset = ytt_dataset
        self.tokenizer = tokenizer
        self.video_token_len = video_token_len
        self.num_instructions = self.ytt_dataset.get_num_instructions()
    
    def __len__(self):
        return self.num_instructions
    
    def __getitem__(self, index):
        source = self.ytt_dataset.get_instruction(index)
        video_name = source['id']
        video_info = self.ytt_dataset.get_video_info(video_name)

        duration = video_info['duration']
        feature = video_info['feature']
        
        temporal_input_locations = []
        temporal_output_locations = []
        
        processed_source = preprocess_multimodal([copy.deepcopy(source['conversations'])], self.video_token_len)[0]
        if source['data_type'] == 'event_sequence_decoding':
            for t in source['timestamps']:
                temporal_output_locations.append(t[0] / duration)
                temporal_output_locations.append(t[1] / duration)
        else:
            for s in processed_source:
                for role in ['User', 'Assistant']:
                    if role == 'User':
                        TEMPORAL_TOKEN = TEMPORAL_INPUT_TOKEN
                        temporal_locations = temporal_input_locations
                    else:
                        TEMPORAL_TOKEN = TEMPORAL_OUTPUT_TOKEN
                        temporal_locations = temporal_output_locations
                    if s[role].find('{SOURCE_CLIP}') != -1:
                        s[role] = s[role].replace('{SOURCE_CLIP}', f'{TEMPORAL_TOKEN} to {TEMPORAL_TOKEN}')
                        start, end = source['SOURCE_CLIP']
                        temporal_locations.append(start/duration)
                        temporal_locations.append(end/duration)
                    if s[role].find('{moment}') != -1:
                        if isinstance(source['moment'][0], list):
                            temporal_expression = ''
                            for i, moment in enumerate(source['moment']):
                                start, end = moment
                                if i > 0:
                                    if i == len(source['moment']) - 1:
                                        temporal_expression = temporal_expression + ' and '
                                    else:
                                        temporal_expression = temporal_expression + ', '
                                temporal_expression = temporal_expression + f'{TEMPORAL_TOKEN} to {TEMPORAL_TOKEN}'
                                temporal_locations.append(start/duration)
                                temporal_locations.append(end/duration)
                            s[role] = s[role].replace('{moment}', temporal_expression)
                        else:
                            s[role] = s[role].replace('{moment}', f'{TEMPORAL_TOKEN} to {TEMPORAL_TOKEN}')
                            start, end = source['moment']
                            temporal_locations.append(start/duration)
                            temporal_locations.append(end/duration)
                    if s[role].find('{instance_class}') != -1:
                        s[role] = s[role].replace('{instance_class}', source['instance_class'])
                    if s[role].find('{content}') != -1:
                        s[role] = s[role].replace('{content}', source['content'])
                    if s[role].find('{click_position}') != -1:
                        s[role] = s[role].replace(
                            '{click_position}',
                            '<{:.2f}, {:.2f}>'.format(*source['click_position'][1]) + TEMPORAL_TOKEN
                        )
                        time_location = source['click_position'][0]
                        temporal_locations.append(time_location/duration)
        
        data_dict = preprocess_v1([processed_source], self.tokenizer)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        data_dict["video"] = feature
        data_dict['temporal_input_locations'] = temporal_input_locations
        data_dict['temporal_output_locations'] = temporal_output_locations
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'video' in instances[0]:
            features = [instance['video'].clone() for instance in instances]
            if all(x is not None and x.shape == features[0].shape for x in features):
                batch['video_spatio_temporal_features'] = torch.stack(features)
            else:
                batch['video_spatio_temporal_features'] = features
        batch['temporal_input_locations'] = [
            instance['temporal_input_locations'] if 'temporal_input_locations' in instance else None for instance in instances
        ]
        batch['temporal_output_locations'] = [
            instance['temporal_output_locations'] if 'temporal_output_locations' in instance else None for instance in instances
        ]
        return batch

def make_supervised_data_module(
    ytt_dataset, 
    tokenizer: transformers.PreTrainedTokenizer, 
    video_token_len: int,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = InstructionLoader(
        ytt_dataset, 
        tokenizer,
        video_token_len,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

with open(data_args.data_dir, 'r') as f:
    instruction_data = json.load(f)

qa_types = ['qa_data', 'instance_qa_data', 'cross_segment_qa_data']
packed_instruction_data = []
for video_name in tqdm(instruction_data):
    for key in instruction_data[video_name]:
        for dialogue in instruction_data[video_name][key]:
            text_dialogue = dialogue['dialogue']
            backbone = dialogue['prototype']['backbone']
            if key in qa_types:
                backbone[0]['User'] = re.sub(r'(' + re.escape(sent_tokenize(text_dialogue[0]['User'])[0]) + r') +', '', backbone[0]['User'])
            packed_instruction_data.append({
                'id' : video_name,
                'data_type' : key,
                'moment' : dialogue['prototype']['variables'].get('moment', None),
                'click_position' : dialogue['prototype']['variables'].get('click_position', None),
                'instance_class' : dialogue['prototype']['variables'].get('instance_class', None),
                'SOURCE_CLIP' : dialogue['prototype']['variables'].get('SOURCE_CLIP', None),
                'content' : dialogue['prototype']['variables'].get('content', None),
                'conversations' : backbone,
            })

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=False,
)
conversation_lib.default_conversation = conversation_lib.conv_templates["momentor_v1"]

model = MomentorLlamaForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
)
model.model.requires_grad_(False)
model.config.use_cache = False

model_vision_dict = model.get_model().initialize_vision_modules(
    pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
)
vision_config = model_vision_dict['vision_config']

model.initialize_vision_tokenizer(
    tokenizer=tokenizer,
    device=training_args.device,
    cache_dir=training_args.cache_dir,
    pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
)
model.initialize_temporal_tokens(tokenizer=tokenizer, num_temporal_tokens=model_args.num_temporal_tokens)

for p in model.get_model().mm_projector.parameters():
    p.requires_grad = True
    
model.model.temporal_input_embeddings = torch.nn.Embedding(model_args.num_temporal_tokens, model.model.embed_tokens.weight.shape[1])
model.model.temporal_output_embeddings = torch.nn.Linear(model.model.embed_tokens.weight.shape[1], model_args.num_temporal_tokens, bias=False)
model._init_weights(model.model.temporal_input_embeddings)
model._init_weights(model.model.temporal_output_embeddings)

ytt_dataset = YTTDataset(instruction_data, data_args.feature_dir)
mixed_dataloader = InstructionLoader(ytt_dataset, tokenizer, video_token_len=DataArguments.num_sampled_frames)
data_module = make_supervised_data_module(
    ytt_dataset, 
    tokenizer, 
    video_token_len=DataArguments.num_sampled_frames,
)

trainer = MomentorTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
trainer.train()
trainer.save_state()
safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)