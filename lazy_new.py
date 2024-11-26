import torch
from torch.utils.data import Dataset
import pandas as pd
import transformers 
import numpy as np
from typing import Dict, Optional, Sequence, List
import copy
import conversation as conversation_lib,PromptGeneratorSingleton

import random
IGNORE_INDEX=-100
special_tokens = {
    "vector_start": "<v_start>",
    "vector_end": "<v_end>",
    "vector_patch": "<v_patch>",
    "vector": "<vector>"
    # "class_start": "<class_start>",
    # "class_end": "<class_end>"
}
class ChronoSightDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, prompt_generator,per_usage=3):
        """
        Args:
            csv_file (str): Path to the CSV file containing the vectors and classes.
            tokenizer: Tokenizer to convert text to token IDs.
            max_vectors (int): Maximum number of <vector><class> pairs to include in each example.
        """
        self.vectors = np.loadtxt('data/combined_vectors.csv', delimiter=',', dtype=float)
        self.labels = np.loadtxt('data/combined_labels.csv', delimiter=',', dtype=str)
        self.tokenizer = tokenizer
        self.per_usage = per_usage
        self.prompt_generator = prompt_generator

        # Ensure new tokens are added to tokenizer
        self.tokenizer.add_tokens(list(special_tokens.values()))# 觉得这里没用

    
    # def _build_prompt(self,idx,classes):
    #     # Prepare human prompt with known pairs, excluding the last vector-class pair
    #     # human_prompt = ("Please act as a classification agent to perform a few-shot learning task.\n"
    #     #                 "Here are some Vector-Class pairs for reference:\n")
    #     human_prompt=""
    #     for c in  classes[:-1]:  # Exclude the last pair#######改
    #         human_prompt += (f"Vector:{special_tokens['vector']}-"
    #                         f"Class: {c}, ")

    #     # Add query for the new vector (last vector in the list)
    #     unique_eles = list(set(classes))
    #     human_prompt += (f"Given these {len(unique_eles)} Classes, ")
    #     for i,ele in enumerate(unique_eles):
    #         human_prompt += (f" Class {i+1}:{ele}, ")
            
    #     human_prompt += (f"\nPredict the Class for this new Vector:"
    #                     f"{special_tokens['vector']}.\n Your answer template: Based on the given Vector-Class pairs, the predicted Class for the Vector is Class:\n")

    #     # Prepare GPT response for the final vector-class pair
    #     response = (f"Based on the given Vector-Class pairs, the predicted Class for the Vector "
    #                 "is Class:"
    #                 f"{classes[-1]}.\n")
        
        
    #     templates = [template_1_logic, template_2_logic, template_3_logic, 
    #                 template_4_logic, template_5_logic, template_6_logic]
    #     chosen_template = random.choice(:6)
    #     human_prompt,response=template(idx, classes)
        
    #     human_prompt_dict={
    #         'from':'human',
    #         'value':human_prompt
    #     }
    #     gpt_prompt_dict={
    #         'from':'gpt',
    #         'value':response
    #     }
    #     sources={
    #         'id':idx,
    #         'conversations':[human_prompt_dict,gpt_prompt_dict]
    #     }
    #     return sources
    def __len__(self):
         return len(self.labels) // self.per_usage

    def __getitem__(self, idx):# 我现在在怀疑这里的浮点数能否进去，之前GraphGPT的话这里依然是以token的空值占位的，进去模型后才用换掉，换掉的方式是图传播，本来数据集不会只有inputids和labels,还会有别的属性，比如graph进去再换掉

       
        # Limit to max_vectors pairs
        start_idx = idx * self.per_usage
        end_idx = start_idx + self.per_usage

        # 获取对应的 vectors 和 classes 子集
        vectors = self.vectors[start_idx:end_idx]
        classes = self.labels[start_idx:end_idx]
        
        sources=self.prompt_generator.generate_prompt(idx, classes)
        
        sources=[sources]
        sources = preprocess_vector(copy.deepcopy([e["conversations"] for e in sources]))# 这里只是把角色放到了value中，我没有把graph都放到头部，其次我已经添加了patch
        data_dict=preprocess(sources, self.tokenizer)        
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict['vector_data']=torch.tensor(vectors)

        return data_dict
        # 本来这里还要加一个额外的地方的向量，比如graph_data。然后直接放到dict中，data_dict = dict(input_ids=data_dict["input_ids"][0],
                            #  labels=data_dict["labels"][0])



def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    ):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
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

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

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

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    ):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
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
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

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

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

# 这里是把前面的ids mask掉，生成任务一般不是mask后面吗，为什么这里是mask前面，是把human的
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
    """这里说的很清晰了
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
def preprocess_vector(sources: Sequence[str]):
    for source in sources:# 这里是单方面说的角色信息放到value中了
            for sentence in source:
                # print(sentence)
                replace_token = special_tokens["vector_start"] + special_tokens["vector_patch"] + special_tokens["vector_end"]
                sentence['value'] = sentence['value'].replace(special_tokens["vector"], replace_token)
    return sources
    

