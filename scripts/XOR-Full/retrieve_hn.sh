MODEL_PATH=checkpoints/XOR-Full/class-xor-full/checkpoint-best

python retriever.py \
  --query_embeddings ${MODEL_PATH}/encoding/query_embedding.pt \
  --passage_embeddings ${MODEL_PATH}/encoding/'embedding_split*.pt' \
  --depth 100 \
  --batch_size 5000 \
  --save_jsonl \
  --search_then_merge \
  --save_ranking_to ${MODEL_PATH}/dev_xor_retrieve_pids.jsonl \
  --use_gpu
