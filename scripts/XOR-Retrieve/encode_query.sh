MODEL_PATH=checkpoints/XOR-Retrieve/class-xor-retrieve/checkpoint-best
DATA_PATH=data/XOR-Retrieve

mkdir -p ${MODEL_PATH}/encoding
python encode.py \
  --model_name_or_path ${MODEL_PATH} \
  --output_dir ${MODEL_PATH} \
  --train_dir ${DATA_PATH} \
  --tf32 True \
  --normalize_text \
  --encode_is_qry \
  --query_file xor_dev_retrieve_eng_span_v1_1.jsonl \
  --separate_joint_encoding \
  --max_query_length 50 \
  --per_device_eval_batch_size 512 \
  --dataloader_num_workers 4 \
  --encoded_save_path ${MODEL_PATH}/encoding/query_embedding.pt
