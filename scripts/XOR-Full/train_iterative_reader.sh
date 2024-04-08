#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=168:00:00
#SBATCH --mem=512G
#SBATCH --partition=deeplearn
#SBATCH -A punim2015
#SBATCH --gres=gpu:A100:4
#SBATCH -q gpgpudeeplearn
#SBATCH --constraint=dlg5
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

srun python train.py --distributed_port 43960 \
  --output_dir ./checkpoints/XOR-Full/class-xor-full \
  --model_name_or_path checkpoints/XOR-Full/class-nq/checkpoint-best \
  --save_steps 500 \
  --task XOR-Full \
  --train_dir data/XOR-Full \
  --train_path xor_train_full.jsonl \
  --corpus_file all_w100.tsv \
  --query_file xor_train_full.jsonl \
  --eval_query_file xor_dev_full_v1_1.jsonl \
  --tf32 True \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 512 \
  --gradient_accumulation_steps 1 \
  --negatives_x_device \
  --grad_cache \
  --refresh_passages \
  --refresh_intervals 1000 \
  --separate_joint_encoding \
  --de_avg_pooling \
  --gradient_checkpointing \
  --gc_chunk_size 8 \
  --retriever_weight 8 \
  --multi_task \
  --ddp_find_unused_parameters False \
  --train_n_passages 100 \
  --max_query_length 50 \
  --max_passage_length 200 \
  --max_query_passage_length 250 \
  --max_answer_length 50 \
  --learning_rate 5e-5 \
  --max_steps 6000 \
  --num_train_epochs 1 \
  --distillation_start_steps 0 \
  --weight_decay 0.01 \
  --dataloader_num_workers 2 \
  --print_steps 20
