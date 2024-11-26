import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast



# 现在我觉得是不需要初始化嵌入的，因为最后还会替换掉，所以intialize_tokenizer完全没有必要。


class ChronoSightConfig(LlamaConfig):
    model_type = "ChronoSight"
        
class ChronoSightLLM(LlamaModel):
    config_class = ChronoSightConfig
    
    def __init__(self, config: ChronoSightConfig):
        super(ChronoSightLLM, self).__init__(config)
        
        # 添加线性层projector，用于对<vector>进行线性变换
        self.graph_projector = nn.Linear(config.graph_hidden_size, config.hidden_size)
        if hasattr(config, "per_usage"):
            self.per_usage=config.per_usage
        
        # # 特殊token标记
        # self.special_tokens = {
        #     "vector_start": "<v_start>",
        #     "vector_end": "<v_end>",
        #     "vector_patch": "<v_patch>"
        # }
        
        # # 将特殊token ID存储，方便forward过程中替换和处理
        # self.vector_start_id = config.tokenizer.convert_tokens_to_ids(self.special_tokens["vector_start"])
        # self.vector_end_id = config.tokenizer.convert_tokens_to_ids(self.special_tokens["vector_end"])
    
    def initialize_module(self, pretrain_graph_mlp_adapter=None, device='cuda',per_usage=3):
        self.per_usage = per_usage
        if pretrain_graph_mlp_adapter is None:
            # 初始化 projector 的权重
            nn.init.normal_(self.graph_projector.weight.to(device), mean=0.0, std=0.02)
            if self.graph_projector.bias is not None:
                nn.init.zeros_(self.graph_projector.bias.to(device))
        else:
            # 加载预训练适配器的权重
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location=device)
            self.graph_projector.load_state_dict({k.split('.')[-1]: v.to(device) for k, v in graph_projector_weights.items()})
    
    # def initialize_module(self, pretrain_graph_mlp_adapter=None):
    #     if pretrain_graph_mlp_adapter is None:
    #         # 当没有提供预训练适配器时，初始化 projector 的权重
    #         nn.init.normal_(self.graph_projector.weight, mean=0.0, std=0.02)
    #         if self.graph_projector.bias is not None:
    #             nn.init.zeros_(self.graph_projector.bias)
    #     else:
    #         # 加载预训练适配器的权重
    #         graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
    #         self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})
    
    def forward(self, input_ids, attention_mask=None, vector_data=None, **kwargs):
        device = input_ids.device  # 确定当前设备
        inputs_embeds = self.embed_tokens(input_ids).to(device)
        # print(f'++++++++the shape of inputs_embeds is : {inputs_embeds.size()}')
        vector_start_positions = (input_ids == self.vector_start_id).nonzero(as_tuple=True)
        vector_end_positions = (input_ids == self.vector_end_id).nonzero(as_tuple=True)
        # print(f'+++++size of input_ids is : {input_ids.size()}')
        # print(f'+++++shape of input_ids is : {input_ids.shape}')
        # print(f'type of input_ids is : {type(input_ids)}')
        # print(f'data is : {data}')
        # print(f'++++type of data is : {type(data)}')
        # print(f'shape of data is : {data.size()}')
        # 遍历每个样本
        batch_size = input_ids.size(0)
        # print(f'====data in forward:{data}')
        # print(f'----batch size:{batch_size}')
        # print(f'----type of vector_data  in up forward is: {type(vector_data)}')
        for batch_index in range(batch_size):
            batch_start_positions = vector_start_positions[1][vector_start_positions[0] == batch_index]
            batch_end_positions = vector_end_positions[1][vector_end_positions[0] == batch_index]
            # print(f'++++++batch_start_positions is : {batch_start_positions}')
            # print(f'++++++batch_end_positions is : {batch_end_positions}')
            
            for data_index, (start_pos, end_pos) in enumerate(zip(batch_start_positions, batch_end_positions)):
                # vector_embeds = inputs_embeds[batch_index, start_pos + 1:end_pos]没用到
                if data_index<self.per_usage:
                    # print(f'shape of data in batch:{vector_data[batch_index][data_index].shape}')
                    # print(f'data:{vector_data[batch_index][data_index]}')
                    # print(f'tyep 0of data in batch:{type(vector_data[batch_index][data_index][0])}')
                    transformed_vector = self.graph_projector(vector_data[batch_index][data_index].to(device)).unsqueeze(0)# 当批处理中，为了在batch中迎合处理，一定需要
                # print(f'++++++transformed_vector size(0) is : {transformed_vector.size(0)}')
                # print(f'++++++transformed_vector size() is : {transformed_vector.size()}')
                # print(f'++inputs_embeds size1 is : {inputs_embeds[batch_index,:].size()}')
                # print(f'++inputs_embeds size2 is : {inputs_embeds[batch_index,start_pos + 2:end_pos-1].size()}')
                
                if transformed_vector.size(0) == (end_pos - start_pos - 1):
                    inputs_embeds[batch_index, start_pos + 1:end_pos] = transformed_vector
                else:
                    # print(f'+++++end_pos is : {end_pos},start_pos is : {start_pos}')
                    raise ValueError("Transformed vector and target segment size do not match.")
        
        return super(ChronoSightLLM, self).forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)


    # def forward(self, input_ids, attention_mask=None, data=None, **kwargs):# inputs_embeds 从一开始就是一个 [batch_size, sequence_length, embedding_dim] 的张量因此不需要堆叠
    #     # 获取嵌入
    #     inputs_embeds = self.embed_tokens(input_ids)
        
    #     # 查找 <v_start> 和 <v_end> 标记位置
    #     vector_start_positions = (input_ids == self.vector_start_id).nonzero(as_tuple=True)
    #     vector_end_positions = (input_ids == self.vector_end_id).nonzero(as_tuple=True)

    #     # 使用 batch 维度遍历每个样本
    #     batch_size = input_ids.size(0)
    #     for batch_index in range(batch_size):
    #         # 获取当前样本中的 <v_start> 和 <v_end> 位置
    #         batch_start_positions = vector_start_positions[1][vector_start_positions[0] == batch_index]
    #         batch_end_positions = vector_end_positions[1][vector_end_positions[0] == batch_index]

    #         # 遍历每一对 <v_start> 和 <v_end>，并用 data 中对应的嵌入替换
    #         for data_index, (start_pos, end_pos) in enumerate(zip(batch_start_positions, batch_end_positions)):
    #             # 获取对应的原始向量嵌入，并应用线性变换
    #             vector_embeds = inputs_embeds[batch_index, start_pos + 1:end_pos]
    #             transformed_vector = self.graph_projector(data[batch_index][data_index])  # 使用对应 data 的第 data_index 项

    #             # 确保 transformed_vector 的形状与替换片段一致
    #             if transformed_vector.size(0) == (end_pos - start_pos - 1):
    #                 inputs_embeds[batch_index, start_pos + 1:end_pos] = transformed_vector
    #             else:
    #                 raise ValueError("Transformed vector and target segment size do not match.")

    #     # 继续模型的前向传播
    #     return super(ChronoSightLLM, self).forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)


class ChronoSightCausalLLM(LlamaForCausalLM):# 对于和上面的并不是继承，所以方法当然没办法用
    config_class = ChronoSightConfig
    def __init__(self,  config: ChronoSightConfig):
        super(LlamaForCausalLM, self).__init__(config)
        # 使用MyLLM作为基础模型
        self.model = ChronoSightLLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    def get_model(self):
        return self.model
    
    # 这个是generate的时候才用的，牛逼啊，是因为这个，导致我一直找，还换名字以为被占用了
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 如果提供了 past_key_values，这意味着我们正在进行生成的后续步骤，而不是第一步。在这种情况下，我们只需要最后一个标记ID，因为我们将使用它来预测下一个标记。
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
                "vector_data": [kwargs.get("vector_data", None)],
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        return model_inputs
    
    def initialize_tokenizer(self, tokenizer, device, tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
    # 添加新 tokens 到词汇表中,因为进来只知道这几个，<vector>只在外面用
        self.special_tokens = {"vector_start": "<v_start>", "vector_end": "<v_end>", "vector_patch": "<v_patch>"}
        tokenizer.add_tokens(list(self.special_tokens.values()), special_tokens=True)
        
        self.resize_token_embeddings(len(tokenizer))
        self.get_model().vector_start_id, self.get_model().vector_end_id = tokenizer.convert_tokens_to_ids(
            [self.special_tokens["vector_start"], self.special_tokens["vector_end"]]
        )
        
        embedding_layer = self.get_input_embeddings()
        
        # 计算已有 token 的平均嵌入，并将新 tokens 的嵌入设置为平均嵌入
        current_embeddings = embedding_layer.weight.data.to(device)
        mean_embedding = torch.mean(current_embeddings, dim=0, keepdim=True)
        
        for token in self.special_tokens.values():
            token_id = tokenizer.convert_tokens_to_ids(token)
            embedding_layer.weight.data[token_id] = mean_embedding.to(device)
        
        if tune_graph_mlp_adapter:
            self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False
        
        if pretrain_graph_mlp_adapter:
            mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location=device)
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight'].to(device)
            # Ensure input embeddings are in the correct device
            input_embeddings = embedding_layer.weight.data.to(device)
            input_embeddings[-len(self.special_tokens):] = embed_tokens_weight

    
    
    
    
    # def initialize_tokenizer(self, tokenizer, device,
    #                                 tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
    #     # 添加新tokens到词汇表中
    #     self.special_tokens = {
    #         "vector_start": "<v_start>",
    #         "vector_end": "<v_end>",
    #         "vector_patch": "<v_patch>"
    #     }
    #     tokenizer.add_tokens(list(self.special_tokens.values()),special_tokens=True)# 转化为string的列表

        
    #     self.resize_token_embeddings(len(tokenizer))
    #     vector_start_id,vector_end_id = tokenizer.convert_tokens_to_ids([self.special_tokens["vector_start"],self.special_tokens["vector_end"]])
    #     self.get_model().vector_start_id = vector_start_id
    #     self.get_model().vector_end_id = vector_end_id
        
    #     # 获取嵌入层
    #     embedding_layer = self.get_input_embeddings()
        
    #     # 计算已有token的平均嵌入
    #     current_embeddings = embedding_layer.weight.data
    #     mean_embedding = torch.mean(current_embeddings, dim=0, keepdim=True)

    #     # 初始化新tokens的嵌入为平均嵌入
    #     for token in self.special_tokens:
    #         token_id = tokenizer.convert_tokens_to_ids(token)
    #         embedding_layer.weight.data[token_id] = mean_embedding
    #     if tune_graph_mlp_adapter:
    #         self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
    #         for p in self.get_input_embeddings().parameters():
    #             p.requires_grad = True
    #         for p in self.get_output_embeddings().parameters():
    #             p.requires_grad = False
    #     # 在有了预训练的之后，新加入的token值映射为了第一阶段的l
    #     if pretrain_graph_mlp_adapter:
    #         # 为啥还要映射到cpu上，没事干吗
    #         mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
    #         embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
    #         assert num_new_tokens == 2
    #         if input_embeddings.shape == embed_tokens_weight.shape:
    #             input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
    #         elif embed_tokens_weight.shape[0] == num_new_tokens:
    #             input_embeddings[-num_new_tokens:] = embed_tokens_weight
    #         else:
    #             raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
            
            
    
    
    def forward(self, input_ids, attention_mask=None, labels=None, vector_data= None,**kwargs):
        # 通过MyLLM模型计算输出
        # print(f'----vector_data in causal forward is:{type(vector_data)}')
        # print(f'----input_ids in causal forward is:{type(input_ids)}')
        # print(f'----input_ids shape in causal forward is:{input_ids.shape}')
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,vector_data = vector_data, **kwargs)
        
        # 提取transformers标准输出
        hidden_states = output[0]##############这里可能有问题，
        # lm_head的的输出是vocab_size
        logits = self.lm_head(hidden_states)
        
        # 如果传入了labels，用于计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift the logits and labels for the causal language model loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=output.past_key_values)


from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("ChronoSight", ChronoSightConfig)  # 注册自定义配置
AutoModelForCausalLM.register(ChronoSightConfig, ChronoSightCausalLLM)  # 注册自定义模型

# 下面是实例化，通过config初始化模型，但是这个时候会遵循Myconfig中的自定义参数，现在我这里是空的，
# config = MyConfig.from_pretrained("path_to_pretrained_model")
# my_model = ChronoSightCausalLLM(config)
#另外就是直接加载模型，这个时候config就会是模型目录下的config.json文件。在模型中，这个一直叫做config直接传唤，在模型外，是通过model.config传唤