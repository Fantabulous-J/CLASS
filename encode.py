import logging
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from arguments import ModelArguments, DataArguments, BiEncoderTrainingArguments as TrainingArguments
from dataloader import EncodeDataset, EncodeCollator, GenericDataLoader
from model import BiEncoder

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    if training_args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    if training_args.separate_joint_encoding:
        from model import RRForConditionalGeneration
        model = RRForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
        # model.encoder.copy_relative_attention_bias()
        # model.decoder.copy_relative_attention_bias()
    else:
        model = BiEncoder.build(
            model_args,
            data_args,
            training_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )

    text_max_length = data_args.max_query_length if data_args.encode_is_qry else data_args.max_passage_length

    corpus = None
    queries = None
    encode_num_shard = data_args.encode_num_shard
    encode_shard_index = data_args.encode_shard_index
    if data_args.encode_is_qry:
        queries = GenericDataLoader(data_args.train_dir, corpus_file=data_args.corpus_file,
                                    query_file=data_args.query_file).load_queries()
        start, end = 0, len(queries)

    else:
        corpus = GenericDataLoader(data_args.train_dir, corpus_file=data_args.corpus_file,
                                   query_file=data_args.query_file).load_corpus()
        from utils import group_corpus_by_langs
        corpus = group_corpus_by_langs(corpus)[data_args.lang]
        shard_size = len(corpus) // encode_num_shard
        start = encode_shard_index * shard_size
        end = (encode_shard_index + 1) * shard_size if encode_shard_index + 1 != encode_num_shard else len(corpus)

    encode_dataset = EncodeDataset(queries if data_args.encode_is_qry else corpus, tokenizer,
                                   max_length=text_max_length, is_query=data_args.encode_is_qry,
                                   start=start, end=end, normalize_text=data_args.normalize_text,
                                   lower_case=data_args.lower_case,
                                   separate_joint_encoding=training_args.separate_joint_encoding,
                                   add_lang_token=data_args.add_lang_token,
                                   eval_mode=True,
                                   task=data_args.task)
    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size * training_args.n_gpu,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            separate_joint_encoding=training_args.separate_joint_encoding,
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model = model.to(training_args.device)
    if training_args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.eval()

    logger.info(f'Generate passage embeddings from {start} to {end}')

    if training_args.fp16:
        autocast_dtype = torch.float16
    elif training_args.bf16:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None
    autocast_enabled = autocast_dtype is not None

    for (batch_ids, batch) in tqdm(encode_loader):
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    if training_args.separate_joint_encoding:
                        query_vector = model(query=batch, only_encoding=True).query_vector
                        # print(query_vector[0])
                        # exit(-1)
                    else:
                        if training_args.n_gpu > 1:
                            query_vector = model(query=batch.data, only_query=True).query_vector
                        else:
                            query_vector = model.encode_query(query=batch)
                    encoded.append(query_vector.cpu())
                else:
                    if training_args.separate_joint_encoding:
                        passage_vector = model(passage=batch, only_encoding=True).passage_vector
                        # print(passage_vector[0], batch_ids[0])
                        # exit(-1)
                    else:
                        if training_args.n_gpu > 1:
                            passage_vector = model(passage=batch.data, only_passage=True).passage_vector
                        else:
                            passage_vector = model.encode_passage(passage=batch)
                    encoded.append(passage_vector.cpu())

    encoded = torch.cat(encoded)
    torch.save((encoded, lookup_indices), data_args.encoded_save_path)


if __name__ == "__main__":
    main()
