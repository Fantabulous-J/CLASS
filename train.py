import copy
import json
import logging
import os
import random
import sys
import time

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from arguments import DataArguments, DistilModelArguments, BiEncoderTrainingArguments
from dataloader import ReaderDataset, ReaderCollator, GenericDataLoader
from model import T5ForConditionalGeneration
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

    # if training_args.output_dir == "./checkpoints/xor-retrieve-english-span-mt5-nq-dpr-pretrain-xor-full":
    #     model_path = "checkpoints/xor-retrieve-english-span-mt5-mss-distill-unsupervised-parallel-augmentation-" \
    #                  "ICL-mss-target-anchor-pretrain-nq-xor-full"
    #     if not os.path.exists(os.path.join(model_path, "checkpoint-best/pytorch_model.bin")):
    #         training_args.output_dir = model_path
    #         model_args.model_name_or_path = "checkpoints/xor-retrieve-english-span-mt5-mss-distill-unsupervised-" \
    #                                         "parallel-augmentation-ICL-mss-target-anchor-pretrain-nq/checkpoint-best"

    # if training_args.output_dir == "./checkpoints/xor-retrieve-english-span-mt5-mss-distill-unsupervised-parallel-" \
    #                                "augmentation-ICL-mss-target-anchor-pretrain-nq-trans-nq":
    #     data_args.train_path = "mss_en_trans.csv"
    #     data_args.query_file = "mss_en_trans.csv"
    #     training_args.save_steps = 500
    #     training_args.refresh_intervals = 1000
    #     training_args.max_steps = 8000

    # if training_args.output_dir == "./checkpoints/xor-retrieve-english-span-mt5-ICL-mss-en-anchor-pretrain":
    #     model_path = "checkpoints/xor-retrieve-english-span-mt5-mss-distill-unsupervised-parallel-augmentation-" \
    #                  "ICL-mss-target-anchor-pretrain-nq-xor-full/nq-ALLT"
    #     if not os.path.exists(os.path.join(model_path, "checkpoint-best/pytorch_model.bin")):
    #         training_args.output_dir = model_path
    #         model_args.model_name_or_path = "checkpoints/xor-retrieve-english-span-mt5-mss-distill-unsupervised-" \
    #                                         "parallel-augmentation-ICL-mss-target-anchor-pretrain-nq/checkpoint-best"
    #         data_args.task = "XOR-Full"
    #         data_args.train_dir = "data/XOR-Full"
    #         data_args.train_path = "xor_nq_train_full.jsonl"
    #         data_args.corpus_file = "all_w100.tsv"
    #         data_args.query_file = "xor_nq_train_full.jsonl"
    #         data_args.eval_query_file = "xor_dev_full_v1_1.jsonl"
    #         training_args.refresh_intervals = 1633
    #         training_args.max_steps = 12000
    #         training_args.use_mcontriever = False
    #         training_args.self_retrieve_steps = 0
    # if model_args.model_name_or_path == "checkpoints/de-experiments/xor-retrieve-english-span-mt5-mss-distill-" \
    #                                     "unsupervised-parallel-augmentation-ICL-mss-en-anchor-pretrain/reader-retrain/checkpoint-best":
    #     model_args.model_name_or_path = "checkpoints/de-experiments/xor-retrieve-english-span-mt5-mss-distill-" \
    #                                     "unsupervised-parallel-augmentation-ICL-mss-en-anchor-pretrain/checkpoint-best"
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

    # if training_args.refresh_passages:
    #     while not os.path.exists(os.path.join(model_args.model_name_or_path, "pytorch_model.bin")):
    #         time.sleep(60)

    if training_args.tf32:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    config.barlow_twins = training_args.barlow_twins

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False
    )
    config.n_passages = data_args.train_n_passages
    config.add_bias = training_args.add_bias
    if not training_args.de_avg_pooling:
        config.retriever_head = 6
    if training_args.separate_joint_encoding:
        from model import RRForConditionalGeneration
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
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )

    if training_args.only_reader:
        assert training_args.separate_joint_encoding, \
            "Only reader training is only supported with separate joint encoding"
        if training_args.de_avg_pooling:
            stage1_model_path = "checkpoints/de-experiments/xor-retrieve-english-span-mt5-mss-multilingual-distill-" \
                                "unsupervised-parallel-augmentation/checkpoint-best"
        else:
            stage1_model_path = "checkpoints/xor-retrieve-english-span-mt5-mss-distill-unsupervised-parallel-" \
                                "augmentation/checkpoint-best"
        stage1_model = RRForConditionalGeneration.from_pretrained(
            stage1_model_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
        model.decoder = copy.deepcopy(stage1_model.decoder)
        model.lm_head = copy.deepcopy(stage1_model.lm_head)
        model.decoder.embed_tokens = model.shared
        for name, param in model.named_parameters():
            if 'encoder.final_layer_norm' in name:
                continue
            if 'encoder' in name and eval(name.split(".")[2]) < config.retriever_layer:
                param.requires_grad = False
            if 'encoder' in name and eval(name.split(".")[2]) == config.retriever_layer:
                if training_args.de_avg_pooling:
                    if 'SelfAttention.query_proj' in name:
                        param.requires_grad = False
                    if 'SelfAttention.passage_proj' in name:
                        param.requires_grad = False
                    if 'SelfAttention.layer_norm' in name:
                        param.requires_grad = False
                else:
                    if 'SelfAttention.q.weight' in name:
                        param.requires_grad = False
                    if 'SelfAttention.k.weight' in name:
                        param.requires_grad = False
            if 'shared' in name:
                param.requires_grad = False

        # if training_args.local_rank in [-1, 0]:
        #     for name, param in model.named_parameters():
        #         print(name, param.requires_grad)

    teacher_tokenizer = None
    if training_args.english_distill or training_args.multilingual_distill:
        config = AutoConfig.from_pretrained(
            model_args.teacher_config_name if model_args.teacher_config_name else model_args.teacher_model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
        config.barlow_twins = training_args.barlow_twins
        config.n_passages = data_args.train_n_passages
        if not hasattr(config, 'retriever_head'):
            config.retriever_head = 6

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            model_args.teacher_tokenizer_name if model_args.teacher_tokenizer_name else model_args.teacher_model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False
        )
        from model import RRForConditionalGeneration
        teacher_model = RRForConditionalGeneration.from_pretrained(
            model_args.teacher_model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir
        )
        teacher_model.to(training_args.device)
        teacher_model.eval()

    if training_args.homogeneous_batch:
        data_dir = data_args.train_dir
        train_path = os.path.join(data_dir, data_args.train_path)
        queries, corpus = None, None
        if training_args.load_corpus:
            corpus = GenericDataLoader(data_dir, corpus_file=data_args.corpus_file).load_corpus()
            queries = GenericDataLoader(data_dir, query_file=data_args.query_file).load_queries()
        queries_per_language = {}
        for qid, query in queries.items():
            if query[-1] not in queries_per_language:
                queries_per_language[query[-1]] = {}
            queries_per_language[query[-1]][qid] = query
        examples_per_language = {}
        corpus_per_language = {}
        for jsonline in tqdm(open(train_path)):
            example = json.loads(jsonline)
            lang = example['qid'][-2:]
            if lang not in examples_per_language:
                examples_per_language[lang] = []
            examples_per_language[lang].append(jsonline)
            if lang not in corpus_per_language:
                corpus_per_language[lang] = {}
            for pid in example['pids']:
                if pid in corpus_per_language[lang]:
                    continue
                corpus_per_language[lang][pid] = corpus[pid]
        train_datasets = []
        for lang, queries in queries_per_language.items():
            num_examples = len(examples_per_language[lang])
            examples = sorted(list(set(examples_per_language[lang])))
            epoch_rnd = random.Random(training_args.seed)
            epoch_rnd.shuffle(examples)
            assert len(examples) == len(queries), \
                f"Number of examples {len(examples)} does not match number of queries {len(queries)}"
            train_dataset = ReaderDataset(
                queries=queries,
                corpus=corpus_per_language[lang],
                tokenizer=tokenizer,
                train_path=examples,
                num_examples=num_examples,
                data_args=data_args,
            )
            train_datasets.append(train_dataset)
    else:
        train_datasets = []
        tasks = ['XOR-Retrieve']
        for _ in tasks:
            data_dir = data_args.train_dir
            train_path = os.path.join(data_dir, data_args.train_path)
            # while not os.path.exists(train_path):
            #     time.sleep(360)

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
        multilingual_distill=training_args.multilingual_distill,
        dictionaries=dictionary,
        teacher_tokenizer=teacher_tokenizer,
    )

    if training_args.english_distill or training_args.multilingual_distill:
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
    elif training_args.wikidata:
        from trainer import ReaderWikidataTrainer
        trainer = ReaderWikidataTrainer(
            model=model,
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
        trainer.train()  # TODO: resume training
        trainer.test()
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
