#!/bin/bash
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=168:00:00
#SBATCH --mem=800G
#SBATCH --partition=feit-gpu-a100
#SBATCH -A punim2015
#SBATCH --gres=gpu:A100:4
#SBATCH -q feit
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

srun python train.py --distributed_port 23333 \
  --output_dir ./checkpoints/XOR-Retrieve/class-stage-2 \
  --model_name_or_path checkpoints/class-stage-1/checkpoint-best \
  --save_steps 500 \
  --train_dir data/XOR-Retrieve \
  --train_path mss.ICL.query.en.anchor.txt \
  --corpus_file psgs_w100.tsv \
  --query_file mss.ICL.query.en.anchor.txt \
  --tf32 True \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 128 \
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
  --max_steps 16000 \
  --num_train_epochs 1 \
  --distillation_start_steps 0 \
  --weight_decay 0.01 \
  --dataloader_num_workers 2 \
  --print_steps 20
