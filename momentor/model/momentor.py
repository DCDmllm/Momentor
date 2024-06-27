from typing import List, Optional, Tuple, Union
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from momentor.constants import *
import numpy as np

def position_transfer(position, num_temporal_tokens):
    position = np.clip(position, 0, 1)
    embed_position = position * (num_temporal_tokens - 1)
    floor_position = math.floor(embed_position)
    ceil_position = math.ceil(embed_position)
    ratio = embed_position - floor_position
    return floor_position, ceil_position, ratio

def token_transfer(position, temporal_embed_tokens, return_position=False):
    position = np.clip(position, 0, 1)
    floor_position, ceil_position, ratio = position_transfer(position, temporal_embed_tokens.shape[0])
    ret_feature = temporal_embed_tokens[floor_position] * (1 - ratio) + temporal_embed_tokens[ceil_position] * ratio
    if return_position:
        return ret_feature, floor_position, ceil_position, ratio
    else:
        return ret_feature

def reparam(weight, reparam_mat):
    reparam_weight = reparam_mat.to(weight.dtype).to(weight.device) @ weight
    return weight + reparam_weight - reparam_weight.detach()

class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1024
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None

class MomentorConfig(LlamaConfig):
    model_type = "Momentor"
    return_dict = True
    output_attentions = True

class MomentorLlamaModel(LlamaModel):
    config_class = MomentorConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(MomentorLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_config = VisionConfig()

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            video_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            temporal_input_locations: Optional[list] = None,
            temporal_output_locations: Optional[list] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        temporal_input_embeddings = reparam(self.temporal_input_embeddings.weight, self.reparam_mat)
        temporal_output_embeddings = reparam(self.temporal_output_embeddings.weight, self.reparam_mat)
        
        if inputs_embeds is None:
            inputs_embeds = F.embedding(input_ids, torch.cat(
                [self.embed_tokens.weight, temporal_input_embeddings.to(self.embed_tokens.weight.device)], 0
            ))

        if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:
            video_features = self.mm_projector(video_spatio_temporal_features) + (temporal_input_embeddings + temporal_output_embeddings)/2
            dummy_video_features = torch.zeros(video_features.shape[1], 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_video_features = self.mm_projector(dummy_video_features)

            new_input_embeds = []
            cur_video_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features.to(cur_input_embeds.device)).sum() # Add Gradient?
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue
                
                if (cur_input_ids == self.vision_config.vid_start_token).sum() != (cur_input_ids == self.vision_config.vid_end_token).sum():
                    raise ValueError("The number of video start tokens and video end tokens should be the same.")
                video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                for video_start_token_pos in video_start_tokens:
                    cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                    num_patches = cur_video_features.shape[0]
                    if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                        raise ValueError("The video end token should follow the video start token.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:video_start_token_pos].detach(),
                            cur_input_embeds[video_start_token_pos:video_start_token_pos + 1],
                            cur_video_features, 
                            cur_input_embeds[video_start_token_pos + num_patches + 1:video_start_token_pos + num_patches + 2],
                            cur_input_embeds[video_start_token_pos + num_patches + 2:].detach(),
                        ), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:video_start_token_pos + 1],
                            cur_video_features,
                            cur_input_embeds[video_start_token_pos + num_patches + 1:]
                        ), dim=0)
                    cur_video_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
                
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        
        if (input_ids.shape[1] != 1 or self.training) and temporal_input_locations is not None and temporal_output_locations is not None:
            new_input_embeds = []
            
            for cur_video_idx, (cur_input_ids, cur_input_embeds) in enumerate(zip(input_ids, inputs_embeds)):
                cur_temporal_input_locations = temporal_input_locations[cur_video_idx]
                cur_temporal_output_locations = temporal_output_locations[cur_video_idx]
                cur_new_input_embeds = inputs_embeds[cur_video_idx].clone()
                if (cur_input_ids == self.vision_config.temporal_input_token_id).sum() \
                    + (cur_input_ids == self.vision_config.temporal_output_token_id).sum() == 0:
                    new_input_embeds.append(cur_input_embeds)
                else:
                    if (cur_input_ids == self.vision_config.temporal_input_token_id).sum() > len(cur_temporal_input_locations) \
                        or (cur_input_ids == self.vision_config.temporal_output_token_id).sum() > len(cur_temporal_output_locations):
                        raise ValueError("The number of temporal tokens and input temporal location features should be the same.")
                    temporal_input_token_indices = torch.where(cur_input_ids == self.vision_config.temporal_input_token_id)[0]
                    temporal_output_token_indices = torch.where(cur_input_ids == self.vision_config.temporal_output_token_id)[0]
                    for i, index in enumerate(temporal_input_token_indices):
                        cur_temporal_location_feature = token_transfer(cur_temporal_input_locations[i], temporal_input_embeddings)
                        cur_new_input_embeds[index] = cur_temporal_location_feature
                    for i, index in enumerate(temporal_output_token_indices):
                        cur_temporal_location_feature = token_transfer(cur_temporal_output_locations[i], temporal_input_embeddings)
                        cur_new_input_embeds[index] = cur_temporal_location_feature
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
        
        return super(MomentorLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class MomentorLlamaForCausalLM(LlamaForCausalLM):
    config_class = MomentorConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MomentorLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            temporal_input_locations: Optional[list] = None,
            temporal_output_locations: Optional[list] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        vision_config = self.get_model().vision_config
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features,
            temporal_input_locations=temporal_input_locations,
            temporal_output_locations=temporal_output_locations
        )
        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            logits = None
            temporal_output_embeddings = reparam(self.model.temporal_output_embeddings.weight, self.model.reparam_mat)
            for cur_video_idx, cur_labels in enumerate(labels):
                cur_temporal_output_locations = temporal_output_locations[cur_video_idx]
                
                cur_logits = F.linear(hidden_states[cur_video_idx:cur_video_idx+1], torch.cat(
                    [self.lm_head.weight, temporal_output_embeddings.to(self.lm_head.weight.device)], 0
                ))
                shift_logits = cur_logits[..., :-1, :].contiguous()
                shift_logits = shift_logits.view(-1, self.vocab_size + self.num_temporal_tokens)
                
                temporal_output_token_indices = torch.where(cur_labels == vision_config.temporal_output_token_id)[0]
                if len(temporal_output_token_indices) > 0:
                    cur_onehot_labels = torch.zeros(*cur_labels.shape, self.vocab_size + self.num_temporal_tokens)
                    for i in range(cur_labels.shape[0]):
                        if cur_labels[i] != -100:
                            cur_onehot_labels[i, cur_labels[i]] = 1
                    for i in range(temporal_output_token_indices.shape[0]):
                        floor_position, ceil_position, ratio = position_transfer(cur_temporal_output_locations[i], self.num_temporal_tokens)
                        cur_onehot_labels[[temporal_output_token_indices[i]]][vision_config.temporal_output_token_id] = 0
                        cur_onehot_labels[[temporal_output_token_indices[i]]][self.vocab_size + floor_position] = 1 - ratio
                        cur_onehot_labels[[temporal_output_token_indices[i]]][self.vocab_size + ceil_position] = ratio
                    shift_labels = cur_onehot_labels[1:].contiguous()
                    shift_labels = shift_labels.view(-1, self.vocab_size + self.num_temporal_tokens)
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss_mask = (cur_labels[1:] != -100)
                    origin_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
                    cur_loss = origin_loss[loss_mask.to(origin_loss.device)].mean()
                    loss = cur_loss if loss is None else loss + cur_loss
                else:
                    shift_labels = cur_labels[..., 1:].contiguous()
                    shit_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    
                    loss = F.cross_entropy(shift_logits, shift_labels) if loss is None else loss + F.cross_entropy(shift_logits, shift_labels)
            logits = cur_logits if logits is None else torch.cat([logits, cur_logits], 0)
        else:
            logits = F.linear(hidden_states, torch.cat([
                self.lm_head.weight, self.model.temporal_output_embeddings.weight.to(self.lm_head.weight.device)], 0
            ))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, tokenizer, device, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
        for p in self.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in self.get_output_embeddings().parameters():
            p.requires_grad = False

        if pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            assert num_new_tokens == 2
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(
                    f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                    f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    
    def initialize_temporal_tokens(self, tokenizer, num_temporal_tokens):
        vision_config = self.get_model().vision_config
        self.num_temporal_tokens = num_temporal_tokens
        self.temporal_tokens = [TEMPORAL_TOKEN_FORMAT.format(i) for i in range(num_temporal_tokens)]
        num_new_tokens = tokenizer.add_tokens([TEMPORAL_INPUT_TOKEN, TEMPORAL_OUTPUT_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        vision_config.temporal_input_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_INPUT_TOKEN])[0]
        vision_config.temporal_output_token_id = tokenizer.convert_tokens_to_ids([TEMPORAL_OUTPUT_TOKEN])[0]
        num_new_tokens = tokenizer.add_tokens(self.temporal_tokens, special_tokens=True)
        index_vec = torch.arange(num_temporal_tokens)
        self.model.reparam_mat = 2.**(-(index_vec[:, None] - index_vec[None]).abs())

AutoConfig.register("Momentor", MomentorConfig)
AutoModelForCausalLM.register(MomentorConfig, MomentorLlamaForCausalLM)