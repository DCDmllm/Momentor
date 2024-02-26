# Momentor

The official repository of paper [Momentor: Advancing Video Large Language Model with Fine-Grained Temporal Reasoning](http://arxiv.org/abs/2402.11435).

Code and dataset will be available soon.


## Momentor Overview

Momentor is a Video-LLM designed for fine-grained comprehension and localization in videos. It is composed of a frame encoder, a linear projection layer, a Temporal Perception Module (TPM), and a Large Language Model (LLM). We carefully design the Temporal Perception Module (TPM) to improve fine-grained temporal modeling and representation. Architecture and training of Momentor are shown in the following figure.

<img src="images/Momentor.jpg"  width="100%">


## Moment-10M

We present Moment-10M, a large-scale video instruction dataset with segment-level annotation. We use videos from YTTemporal-1B to construct Moment-10M. We propose an automatic data generation engine to extract instance and event information from these videos and generate segment-level instruction following data. We meticulously design 5 single-segment tasks and 3 cross-segment tasks, which enables Video-LLMs perform comprehensive segment-level reasoning.

<img src="images/data_generation_engine.jpg"  width="100%">

Moment-10M will be released soon.


## Acknowledgment
Thanks to the open source of the following projects:
+ [LLaMA](https://github.com/facebookresearch/llama)
+ [Vicuna](https://github.com/lm-sys/FastChat)
+ [LLaVA](https://github.com/haotian-liu/LLaVA)
+ [BLIP2](https://huggingface.co/Salesforce/blip2-opt-2.7b-coco)
+ [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)
+ [LAVIS](https://github.com/salesforce/LAVIS)
+ [Tag2Text](https://github.com/xinyu1205/recognize-anything)
+ [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 
+ [PySceneDetect](https://www.scenedetect.com/download)
+ [merlot_reserve](https://rowanzellers.com/merlotreserve)