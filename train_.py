
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
import conversation as conversation_lib
import torch

import transformers
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from lazy_ import ChronoSightDataset


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

    gpus: Optional[str] = field(default='0,1')
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


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # 这里是自定义collate动态填充的时候
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'vector_data' in instances[0]:
            # graph_node_reps = [instance['graph_node'] for instance in instances]
            # edge_index_reps = [instance['graph_edge'] for instance in instances]
            # target_node_reps = [instance['target_node'] for instance in instances]
            vector_data_batch = [instance['vector_data'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            #     batch['images'] = torch.stack(images)
            # else:
            #     batch['images'] = images
        # batch['graph_node_reps'] = graph_node_reps
        # batch['edge_index_reps'] = edge_index_reps
        # batch['edge_index_reps'] = target_node_reps
        batch['vector_data'] = vector_data_batch

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ChronoSightDataset(tokenizer=tokenizer,
                                per_usage=data_args.sample_used_per_instance,
                                )
    print("++++train_dataset len :", len(train_dataset))
    # print(train_dataset[0])
    # print(f'++++++data type in dataset is :{type(train_dataset[0])}')
    print('-----------')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=training_args.per_device_train_batch_size,
                                  num_workers=training_args.num_workers,
                                  collate_fn=data_collator,
                                  prefetch_factor=4,
                                  pin_memory=True)
    return train_dataloader, None


def train():
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False
        )

    if model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token   #如果我的模型有pad_token，这一句不需要也行。
        # conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
    else: 
        raise ValueError

    model = ChronoSightPL(training_args, model_args, data_args, tokenizer)##########

    train_dataloader, _ = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args, training_args=training_args)
    # 这里控制存储，只是创建了回调对象，用于保存，并不是加载的
    checkpoint_callback = ModelCheckpoint(
            dirpath=training_args.output_dir,
            filename=model_args.model_save_name,
            monitor="loss",
            save_top_k=1,
            save_last=True,
        )

    # fsdp是一种数据并行化技术，用于在分布式训练环境中提高模型训练的效率和可扩展性。
    if training_args.strategy == 'fsdp': 
        strategy = FSDPStrategy(
        auto_wrap_policy={LlamaDecoderLayer},
        activation_checkpointing_policy={LlamaDecoderLayer},
        state_dict_type="full",
        limit_all_gathers=True,
        cpu_offload=False,
        # **kwargs
        )
    else: 
        strategy = training_args.strategy

    wandb_logger = WandbLogger(save_dir=training_args.output_dir, project="ChronoTime", offline=True, name=model_args.model_save_name)
    model_precision = ('16' if training_args.fp16 else ('bf16' if training_args.bf16 else '32'))
    # print('************* epoch:', training_args.num_train_epochs)
    # 只要在模型类中，继承lightingmodule然后写tranining_step方法，就可以调用trainer训练了
    trainer = Trainer(default_root_dir=training_args.output_dir, max_epochs=int(training_args.num_train_epochs), 
                    accumulate_grad_batches=gradient_accumulation_steps,
                    accelerator="gpu", devices=devices, 
                    strategy=strategy,
                    logger = wandb_logger, 
                    precision=model_precision,
                    callbacks=[checkpoint_callback])
    resume = training_args.resume

    # for name, param in model.named_parameters():
    #     print(name, param.dtype)
    # model.to(dtype=torch.float16)

    trainer.fit(model, train_dataloader, ckpt_path=resume)

    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                    output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
