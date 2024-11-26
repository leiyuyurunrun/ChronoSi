# to fill in the following path to run the first stage of our GraphGPT!
model_path=backbone/vicuna-7b-v1.5-16k
instruct_ds=pre_resources/dataset/graph-matching/train_instruct_graphmatch.json
graph_data_path=pre_resources/dataset/All_pyg_graph_data/graph_data_all.pt
pretra_gnn=clip_gt_arxiv
# 一会有一个判断，因此pretra_gnn一定得是这样。
output_model=./checkpoints

/data/Matrix/common/miniconda3/envs/ChronoTime/bin/python train.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True \
    --bf16 False \
    --output_dir ${output_model} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --real_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --fp16 False
