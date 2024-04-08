MODEL_PATH=checkpoints/XOR-Retrieve/class-xor-retrieve/checkpoint-best

python retriever.py \
  --query_embeddings ${MODEL_PATH}/encoding/query_embedding.pt \
  --passage_embeddings ${MODEL_PATH}/encoding/'embedding_split*.pt' \
  --depth 100 \
  --batch_size 5000 \
  --search_then_merge \
  --save_ranking_to ${MODEL_PATH}/dev_xor_retrieve_results.json \
  --use_gpu
