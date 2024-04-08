import json
import logging
import os
import sys
from statistics import mean

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from torch.utils.data import DataLoader, SequentialSampler

from arguments import DataArguments, DistilModelArguments, BiEncoderTrainingArguments
from dataloader import ReaderDataset, ReaderCollator, GenericDataLoader
from model import RRForConditionalGeneration

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((DistilModelArguments, DataArguments, BiEncoderTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: DistilModelArguments
        data_args: DataArguments
        training_args: BiEncoderTrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)

    set_seed(training_args.seed)

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
        use_fast=False
    )
    config.n_passages = data_args.train_n_passages
    if not training_args.de_avg_pooling:
        config.retriever_head = 6
    model = RRForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )

    model.to(training_args.device)
    model.eval()

    data_dir = data_args.train_dir
    if os.path.exists(data_args.train_path):
        train_path = data_args.train_path
    else:
        train_path = os.path.join(data_dir, data_args.train_path)

    corpus = GenericDataLoader(data_dir, corpus_file=data_args.corpus_file).load_corpus()
    queries = GenericDataLoader(data_dir, query_file=data_args.query_file).load_queries()

    output_path = os.path.join(model_args.model_name_or_path, data_args.output_path)

    eval_dataset = ReaderDataset(
        queries=queries,
        corpus=corpus,
        tokenizer=tokenizer,
        train_path=train_path,
        data_args=data_args,
    )

    predictions = {}

    data_collator = ReaderCollator(
        tokenizer,
        max_query_length=data_args.max_query_length,
        max_passage_length=data_args.max_passage_length,
        max_query_passage_length=data_args.max_query_passage_length,
        max_answer_length=data_args.max_answer_length,
        separate_joint_encoding=training_args.separate_joint_encoding,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.eval_batch_size,
        sampler=SequentialSampler(eval_dataset),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )

    autocast_dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            if training_args.separate_joint_encoding:
                qids, reader_inputs, batch_answer, batch_pids = batch
            else:
                qids, encoded_query, encoded_passage, reader_inputs, batch_answer, batch_pids = batch

            bsz, seq_len = reader_inputs["input_ids"].size()
            input_ids = reader_inputs["input_ids"].view(bsz // data_args.train_n_passages,
                                                        data_args.train_n_passages, seq_len)
            if training_args.separate_joint_encoding:
                model_inputs = {
                    "input_ids": input_ids.to(training_args.device),
                    "attention_mask": reader_inputs["attention_mask"].to(training_args.device),
                    "independent_mask": reader_inputs["independent_mask"].to(training_args.device),
                }
            else:
                model_inputs = {
                    "input_ids": input_ids.to(training_args.device),
                    "attention_mask": reader_inputs["attention_mask"].to(training_args.device)
                }

            with torch.cuda.amp.autocast(enabled=training_args.fp16 or training_args.bf16, dtype=autocast_dtype):
                outputs = model.generate(
                    **model_inputs,
                    max_length=data_args.max_answer_length,
                    num_beams=1,
                )
            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                predictions[qids[k]] = ans

            torch.cuda.empty_cache()

    with open(output_path, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
