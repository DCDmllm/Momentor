# Training Momentor
We train Momentor on our Moment-10M dataset. Please follow the instructions below to train Momentor.

### Prepare LLaVA and Video-ChatGPT weights

- Download the LLaVA-Lightening-7B weights from [mmaaz60/LLaVA-Lightening-7B-v1-1](https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1).
- Download the Video-ChatGPT-7B weights from [mbzuai-oryx/Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).


## Prepare Dataset

**1. Follow the instructions in [rowanz/merlot_reserve](https://github.com/rowanz/merlot_reserve) to download videos from YTTemporal-1B.** 

**2. Use CLIP to extract frame features from the videos.**

```shell
python momentor/scripts/feature_extraction.py \
        --video_file_dir <path to the directory of the YTTemporal videos> \
        --log_file_path <path to the log file>\
        --save_dir <path to where you want to save the video features>\
        --device_id <cuda device id>\
```

## Train Momentor

```shell
torchrun --nproc_per_node=8 --master_port 29001 momentor/train/train_mem.py \
    --model_name_or_path <path to LLaVA-7B-Lightening-v-1-1 model> \
    --pretrain_mm_mlp_adapter <path to the parameters of pretrained multimodal adapter> \
    --output_dir <path to save checkpoints> \
    --data_dir <path to the instruction data file> \
    --feature_dir <path to the encoded video features> \
    --model_max_length 2048 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --gradient_checkpointing True \
    --bf16 True \
    --tf32 True
```
