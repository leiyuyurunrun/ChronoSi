import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from conversation import conv_templates, SeparatorStyle
from utils import disable_torch_init

from chronosight import ChronoSightCausalLLM

from model_utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy

import os


from tqdm import tqdm
import json
import os.path as osp

import ray

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'




def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# def prepare_query(instruct_item): 


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    print(f'chunk-size:{chunk_size}------prompt file len :{len(prompt_file)}')
    ans_handles = []
    #######
    ground_end_id=min(args.end_id,len(prompt_file))
    split_list = list(range(args.start_id, ground_end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    print(f'split list:{split_list}-----num_gpus:{num_gpus}')
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []
    # print('==================')
    # print(ans_handles[:3])
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading tokenizer======')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading tokenizer===')
    #  这次是通过config.json来实例化了，没有那么多参数类了。
    print('start loading model=====')
    model =ChronoSightCausalLLM.from_pretrained(args.model_name, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading model=======')
    
    
    
    special_tokens = { "vector_start":"<v_start>", "vector_end": "<v_end>", "vector_patch": "<v_patch>","vector":"<vector>"}
    tokenizer.add_tokens(list(special_tokens.values()), special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))
    model.get_model().vector_start_id, model.get_model().vector_end_id = tokenizer.convert_tokens_to_ids(
                                                                        [special_tokens["vector_start"], special_tokens["vector_end"]])
    # 但是这样在主函数中这样改模型的参数有点奇怪，to do 最好自己之后改到init中

    print(f'total: {len(prompt_file)}')
    
    res_data=[]
    for idx, instruct_item in tqdm(enumerate(prompt_file)):## to do 一会试着把json存储了。
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
 
        qs = instruct_item["conversations"][0]["value"]
        data=instruct_item["vector_data"]
        
        conv_mode = "chronosight_v1"
        replace_token = special_tokens["vector_start"] + special_tokens["vector_patch"] + special_tokens["vector_end"]
        qs = qs.replace(special_tokens["vector"], replace_token)

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        vector_data=torch.tensor(data,dtype=torch.float16).cuda()
        # print(f'in eval =======shape of vector_data:{vector_data.shape}')
        # print(f'in eval ====type of vector_data:{type(vector_data)}')
        # print(f' in eval ==========type of vector_data0:{type(vector_data[0])}')
        # print(f'in eval ========type of vector_data00:{type(vector_data[0][0])}')
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # print(f'=======vector data type in eval is {type(vector_data)}')

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                vector_data=vector_data,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria])

        # print('==after generate in eval======')
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        human_qs=instruct_item["conversations"][0]
        from_generate={'from': 'my_gpt','value': outputs}
        conversation=[human_qs, from_generate]
        res_data.append({"id": instruct_item["id"], "conversation":conversation}.copy())
        with open(osp.join(args.output_res_path, 'my_data_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
            json.dump(res_data, fout, indent=4)
    return res_data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20567)

    args = parser.parse_args()
    return args


# 现在你可以使用 args 对象访问命令行参数
# 例如：
# model_name = args.model_name
# start_id = args.start_id



if __name__ == "__main__":
    args = parse_arguments()
    ray.init()
    run_eval(args, args.num_gpus)
# 确实这里没有使用pl，所以也没办法使用eval_step,但是这里走的又是generate跟那个不是一回路。


# protobuf             4.22.3