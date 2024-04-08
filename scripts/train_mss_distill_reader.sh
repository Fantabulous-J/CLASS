#!/bin/bash
#
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=240:00:00
#SBATCH --mem=800G
#SBATCH --partition=deeplearn
#SBATCH -A punim2015
#SBATCH --gres=gpu:A100:4
#SBATCH -q gpgpudeeplearn
#SBATCH --constraint=dlg5
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

srun python train.py --distributed_port 43960 \
  --output_dir ./checkpoints/class-stage-1 \
  --model_name_or_path google/mt5-large \
  --teacher_model_name_or_path neulab/reatt-large-nq \
  --save_steps 1000 \
  --train_dir data/XOR-Retrieve \
  --train_path mss.parallel.train.json \
  --corpus_file psgs_w100.tsv \
  --query_file mss.parallel.train.txt \
  --tf32 True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --negatives_x_device \
  --separate_joint_encoding \
  --de_avg_pooling \
  --gradient_checkpointing \
  --cross_lingual_alignment \
  --english_distill \
  --parallel_queries \
  --retriever_weight 8 \
  --grad_cache \
  --gc_chunk_size 8 \
  --multi_task \
  --ddp_find_unused_parameters False \
  --train_n_passages 100 \
  --max_query_length 50 \
  --max_passage_length 200 \
  --max_query_passage_length 250 \
  --max_answer_length 50 \
  --learning_rate 1e-4 \
  --max_steps 64000 \
  --num_train_epochs 1 \
  --distillation_start_steps 0 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --dataloader_num_workers 2 \
  --print_steps 20
