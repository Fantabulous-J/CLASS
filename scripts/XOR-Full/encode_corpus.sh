MODEL_PATH=checkpoints/XOR-Full/class-xor-full/checkpoint-best
DATA_PATH=data/XOR-Full
SHARD_IDX=0
NUM_SHARD=16

mkdir -p ${MODEL_PATH}/encoding
python encode.py \
  --model_name_or_path ${MODEL_PATH} \
  --output_dir ${MODEL_PATH} \
  --train_dir ${DATA_PATH} \
  --tf32 True \
  --corpus_file all_w100.tsv \
  --separate_joint_encoding \
  --max_passage_length 200 \
  --per_device_eval_batch_size 512 \
  --encode_shard_index ${SHARD_IDX} \
  --encode_num_shard ${NUM_SHARD} \
  --dataloader_num_workers 4 \
  --encoded_save_path ${MODEL_PATH}/encoding/embedding_split${SHARD_IDX}.pt
