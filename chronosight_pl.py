import os
from chronosight import ChronoSightCausalLLM
import torch
import logging
from torch import nn
from lightning.pytorch import LightningModule
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizer, get_cosine_schedule_with_warmup
from transformers import AdamW
import transformers


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class ChronoSightPL(LightningModule):
    def __init__(self, training_args, model_args, data_args, tokenizer, **kwargs):
        super(ChronoSightPL, self).__init__()
        
        # 将参数存储为类属性
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        self.tokenizer = tokenizer
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        bnb_model_from_pretrained_args = {}
        # 配置模型的量化参数
        if training_args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_int8_training
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                )
            ))

        # 初始化ChronoSightCausalLLM模型
        self.model = ChronoSightCausalLLM.from_pretrained(model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
        self.model.config.use_cache = False
        if model_args.freeze_backbone:
            self.model.model.requires_grad_(False)

        if training_args.bits in [4, 8]:
            self.model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            self.model = prepare_model_for_int8_training(self.model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        
        # 判断是否需要添加LoRA层
        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            logging.warning("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        # 这里涉及model的地方我都取消了更远一步得到model,如get_model等应该不会有错
        self.model.config.tune_graph_mlp_adapter = training_args.tune_graph_mlp_adapter = model_args.tune_graph_mlp_adapter
         
        # 初始化线性层参数（initialize_module 方法）
        self.model.get_model().initialize_module(model_args.pretrain_graph_mlp_adapter,per_usage=data_args.sample_used_per_instance)## 对比的这里有一个返回，我这里先写了

        # 判断是否需要调节graph MLP adapter
        if model_args.tune_graph_mlp_adapter:
            # 使 projector 的参数需要梯度
            self.model.requires_grad_(False)
            for param in self.model.get_model().graph_projector.parameters():
                param.requires_grad = True
        self.model.config.freeze_graph_mlp_adapter = training_args.freeze_graph_mlp_adapter
        if training_args.freeze_graph_mlp_adapter:
            # 冻结 projector 的参数
            for param in self.model.get_model().projector.parameters():
                param.requires_grad = False
        self.model.initialize_tokenizer(tokenizer=self.tokenizer, device='cuda',
                                        tune_graph_mlp_adapter=training_args.tune_graph_mlp_adapter,
                                        pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter)
        if training_args.bits in [4, 8]:#这里是进不来的，因为是16
            self.model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.deivce)##########没这个属性啊，这是怎么进去的
        
        if training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)
                            
            print('**************************  require_grad parameters nums: #', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            tuned_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    tuned_params.append(name)
            print(f'#######  tuned_params:{tuned_params}')
            
            
    def training_step(self, batch, batch_idx):
        bs = len(batch["input_ids"])
        # 在这里写调用model传播。
        loss_dict = self.model(**batch)
        loss = loss_dict['loss']
        
        log_dict = {f'loss': loss.item()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss
    # def training_step(self, batch, batch_idx):
    #     # 前向传播并计算损失
    #     outputs = self.model(
    #         input_ids=batch['input_ids'],
    #         attention_mask=batch['attention_mask'],
    #         labels=batch['labels']
    #     )
    #     loss = outputs.loss

    #     # 返回loss供训练过程使用
    #     self.log("train_loss", loss)
    #     return loss

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # no_decay = ["bias", "LayerNorm.weight"]
        # if IS_STAGE2:
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()], "lr_scale": [1e-5, 1e-4]
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learning_rate)

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.training_args.warmup_steps,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
