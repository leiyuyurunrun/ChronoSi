import torch
from torch.utils.data import Dataset
import pandas as pd
import transformers 
import numpy as np
from typing import Dict, Optional, Sequence, List
import copy
import conversation as conversation_lib

special_tokens = {
    "vector_start": "<v_start>",
    "vector_end": "<v_end>",
    "vector_patch": "<v_patch>",
    "vector": "<vector>"
    # "class_start": "<class_start>",
    # "class_end": "<class_end>"
    }
def produce_dataset(per_usage=3,save_path='data.json'):
    vectors = np.loadtxt('data/combined_vectors.csv', delimiter=',', dtype=float)
    labels = np.loadtxt('data/combined_labels.csv', delimiter=',', dtype=str)
    # tokenizer = tokenizer
    per_usage = per_usage

    def build_prompt(idx,classes):
        # Prepare human prompt with known pairs, excluding the last vector-class pair
        human_prompt=''
        for c in  classes[:-1]:  # Exclude the last pair#######改
            human_prompt += (f"Vector:{special_tokens['vector']}-"
                            f"Class: {c}, ")

        # Add query for the new vector (last vector in the list)
        unique_eles = list(set(classes))
        human_prompt += (f"Given these {len(unique_eles)} Classes, ")
        for i,ele in enumerate(unique_eles):
            human_prompt += (f" Class {i+1}:{ele}, ")
            
        human_prompt += (f"\nPredict the Class for this new Vector:"
                        f"{special_tokens['vector']}. \n Your answer template: Based on the given Vector-Class pairs, the predicted Class for the Vector is Class:\n")

        # Prepare GPT response for the final vector-class pair
        response = ("Based on the given Vector-Class pairs, the predicted Class for the Vector"
                    " is Class:"
                    f"{classes[-1]}.\n")
        human_prompt_dict={
            'from':'human',
            'value':human_prompt
        }
        gpt_prompt_dict={
            'from':'gpt',
            'value':response
        }
        sources={
            'id':idx,
            'conversations':[human_prompt_dict,gpt_prompt_dict]
        }
        return sources

    # Ensure new tokens are added to tokenizer
    # tokenizer.add_tokens(list(special_tokens.values()))
    json_content=[]
    for idx in range(0,len(labels) // per_usage):
        start_idx = idx * per_usage
        end_idx = start_idx + per_usage

        # 获取对应的 vectors 和 classes 子集
        vectors_subset = vectors[start_idx:end_idx]
        classes_subset = labels[start_idx:end_idx]
        sources=build_prompt(idx, classes_subset)
        sources['vector_data']=vectors_subset.tolist()#这里转化为列表了，读出的时候记得转化为ndarray
        json_content.append(sources)
    with open(save_path, 'w') as f:
        json.dump(json_content, f, indent=4)
    

import os
work_space=os.getcwd()
import sys
sys.path.append(work_space)

import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import LightningModule, Trainer, seed_everything

from chronosight_pl import ChronoSightPL


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.callback import Callback

IGNORE_INDEX = -100
# id还有不明白的点

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=False)  # 这里本来应该放到TrainingArguments表示需不需要tune，
    graph_tower: Optional[str] = field(default=None)
    graph_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)  # 是否已经有pretrain的linear
    use_graph_start_end: bool = field(default=False)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_graph: bool = False
    sep_graph_conv_front: bool = False
    graph_token_len: int = 0
    graph_content: Optional[str] = field(default=None)
    graph_data_path: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    sample_used_per_instance:int = field(default=3)


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_graph_mlp_adapter: bool = field(default=False) # eval的时候
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    strategy: str = field(
        default='fsdp'
    )
    real_batch_size: int = field(default=1)

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool =False

    gpus: Optional[str] = field(default='6,7')
    resume: Optional[str] = field(default=None)

    adam_epsilon: float = field(default=1e-8)
    warmup_steps:int = field(default=1000)
    num_workers:int = field(default=16)

    bf16: bool = field(default=False) 
    fp16: bool = field(default=False) 
    output_dir: str = field(default='./checkpoints/graphchat-gt-graphmatch-7b') 
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default='no')
    save_strategy: str = field(default='steps')
    save_steps: int = field(default=2400)
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default='cosine')
    logging_steps: int = field(default=1)
    tf32: bool = field(default=True) 
    gradient_checkpointing: bool = field(default=True)
    report_to: str = field(default='wandb')


def main():
    save_path='data.json'
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(',')]
    
    batch_size = training_args.real_batch_size
    devices = training_args.gpus
    num_devices = len(devices)
    gradient_accumulation_steps = max(1,batch_size // (training_args.per_device_train_batch_size*num_devices))

    #分词器使用vicuna的
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=training_args.cache_dir,
    #         model_max_length=training_args.model_max_length,
    #         padding_side="right",
    #         use_fast=False
    #     )

    # if model_args.version == "v1":
    #     tokenizer.pad_token = tokenizer.unk_token   #如果我的模型有pad_token，这一句不需要也行。
        # conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    # else: 
    #     raise ValueError
    produce_dataset(3,save_path)

if __name__ == "__main__":
    main()
