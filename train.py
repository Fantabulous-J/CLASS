import copy
import logging
import os
import sys

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import DataArguments, DistilModelArguments, BiEncoderTrainingArguments
from dataloader import ReaderDataset, ReaderCollator, GenericDataLoader
from model import RRForConditionalGeneration
from trainer import ReaderTrainer
from utils import load_dictionaries

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
        training_args.fp16 or training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("Data Parameters %s", data_args)

    set_seed(training_args.seed)

    if training_args.tf32:
        import torch
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
    initialise_from_smaller_model = False
    if config.num_layers != 24:
        config.num_layers = config.num_layers * 2
        initialise_from_smaller_model = True
    config.retriever_layer = config.num_layers // 2

    model = RRForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )
    if initialise_from_smaller_model:
        for idx in range(config.num_layers // 2, config.num_layers):
            model.encoder.block[idx] = copy.deepcopy(model.encoder.block[idx - config.num_layers // 2])
            for module in model.encoder.block[idx].modules():
                if hasattr(module, 'layer_index'):
                    module.layer_index = idx
                if hasattr(module, 'relative_attention_bias'):
                    delattr(module, 'relative_attention_bias')

    teacher_tokenizer = None
    if training_args.english_distill:
        config = AutoConfig.from_pretrained(
            model_args.teacher_config_name if model_args.teacher_config_name else model_args.teacher_model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
        config.n_passages = data_args.train_n_passages
        if not hasattr(config, 'retriever_head'):
            config.retriever_head = 6

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            model_args.teacher_tokenizer_name if model_args.teacher_tokenizer_name else model_args.teacher_model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False
        )
        teacher_model = RRForConditionalGeneration.from_pretrained(
            model_args.teacher_model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
        teacher_model.to(training_args.device)
        teacher_model.eval()

    train_datasets = []
    tasks = ['XOR-Retrieve']
    for _ in tasks:
        data_dir = data_args.train_dir
        train_path = os.path.join(data_dir, data_args.train_path)

        queries, corpus = None, None
        if training_args.load_corpus:
            corpus = GenericDataLoader(data_dir, corpus_file=data_args.corpus_file).load_corpus()
            queries = GenericDataLoader(data_dir, query_file=data_args.query_file).load_queries()

        train_dataset = ReaderDataset(
            queries=queries,
            corpus=corpus,
            tokenizer=tokenizer,
            train_path=train_path,
            data_args=data_args,
        )
        train_datasets.append(train_dataset)

    dictionary = None
    if training_args.cross_lingual_alignment and not training_args.parallel_queries:
        dictionary = load_dictionaries()
    data_collator = ReaderCollator(
        tokenizer,
        max_query_length=data_args.max_query_length,
        max_passage_length=data_args.max_passage_length,
        max_query_passage_length=data_args.max_query_passage_length,
        max_answer_length=data_args.max_answer_length,
        separate_joint_encoding=training_args.separate_joint_encoding,
        parallel_queries=training_args.parallel_queries,
        dictionaries=dictionary,
        teacher_tokenizer=teacher_tokenizer,
    )

    if training_args.english_distill:
        from trainer import ReaderDistillTrainer
        trainer = ReaderDistillTrainer(
            model=model,
            teacher_model=teacher_model,
            train_dataset=train_datasets,
            data_collator=data_collator,
            training_args=training_args,
            data_args=data_args,
            tokenizer=tokenizer
        )
    else:
        trainer = ReaderTrainer(
            model=model,
            train_dataset=train_datasets,
            data_collator=data_collator,
            training_args=training_args,
            data_args=data_args,
            tokenizer=tokenizer
        )

    for dataset in train_datasets:
        dataset.trainer = trainer

    if os.path.exists(os.path.join(os.path.join(training_args.output_dir, "checkpoint-best"), "pytorch_model.bin")) \
            and training_args.eval_on_test:
        trainer.test()
    else:
        trainer.train()
        trainer.test()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
