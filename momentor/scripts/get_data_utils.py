import pandas as pd
import json
import numpy as np
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.data.encoded_video_pyav import thwc_to_cthw, _pyav_decode_stream
import torch
import os, math

def get_index_frame(video, indices):
    frame_idxs = indices

    try:
        outputs = video._av_reader.get_batch(frame_idxs)
    except Exception as e:
        print(f"Failed to decode video with Decord: {video._video_name}. {e}")
        raise e

    video = outputs

    if video is not None:
        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video.asnumpy())
        video = video.to(torch.float32)
        video = thwc_to_cthw(video)

    return video

def get_sec_frame(video, seconds):
    if isinstance(seconds, float):
        indices = [min(round(video._fps * seconds), len(video._av_reader)-1)]
    else:
        indices = [min(round(video._fps * sec), len(video._av_reader)-1) for sec in seconds]
    frame_idxs = indices

    try:
        outputs = video._av_reader.get_batch(frame_idxs)
    except Exception as e:
        print(f"Failed to decode video with Decord: {video._video_name}. {e}")
        raise e

    video = outputs

    if video is not None:
        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video.asnumpy())
        video = video.to(torch.float32)
        video = thwc_to_cthw(video)

    return video

def get_clip_audio(video, start_sec, end_sec, target_sample_rate=16000):
    sample_rate = video._av_reader._AVReader__audio_reader.sample_rate
    num_audio_signals = video._av_reader._AVReader__audio_reader.shape[1]
    
    start_idx = int(sample_rate * start_sec)
    end_idx = math.ceil(sample_rate * end_sec)
    start_idx, end_idx = min(start_idx, num_audio_signals-1), min(end_idx, num_audio_signals-1)
    
    frame_idxs = np.linspace(start_idx, end_idx, int(target_sample_rate*(end_sec - start_sec))).astype('int64')
    
    audio_arr = torch.Tensor(video._av_reader._AVReader__audio_reader._array[0, frame_idxs])
    audio = audio_arr.to(torch.float32)
    return audio

def read_video_file(video_path, decoder='decord', decode_audio=False):
    video = EncodedVideo.from_path(video_path, decoder=decoder, decode_audio=decode_audio)
    return video

def read_video_frames(video, sampled_seconds=None, num_samples=None, start=None, end=None, centered=True, decode_audio=False, return_sampled_seconds=False):
    duration = video.duration
    if sampled_seconds is None:
        if start is None:
            start = 0
        if end is None:
            end = video.duration
        duration = end - start
        if num_samples is None:
            num_samples = round(duration)
        if centered:
            clip_len = duration / num_samples
            sampled_seconds = np.arange(num_samples) * clip_len + 0.5 * clip_len + start
        else:
            clip_len = duration / (num_samples-1)
            sampled_seconds = np.arange(num_samples) * clip_len + start
    video_data = get_sec_frame(video, sampled_seconds)
    if return_sampled_seconds:
        return video_data, sampled_seconds, duration
    else:
        return video_data, duration